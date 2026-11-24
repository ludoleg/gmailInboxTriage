"""Exclusive GTX 1080 Ti GPU lease manager coordinating inkling-ctl and Ollama."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


class GpuLeaseManager:
    """Manages exclusive GPU access between Ollama and inkling-server.service.

    Uses only `inkling-ctl start`, `inkling-ctl stop`, and `inkling-ctl unload-ollama`
    so ~/.hermes/config.yaml and hermes-gateway.service are never modified.
    Snapshots the initial inkling-server.service state on entry and restores it on exit.
    """

    def __init__(
        self,
        inkling_ctl_path: Path,
        inkling_base_url: str = "http://127.0.0.1:8001",
        restore_policy: Literal["restore", "keep_inkling", "idle"] = "restore",
        unit_name: str = "inkling-server.service",
        command_timeout_s: float = 45.0,
    ) -> None:
        self.inkling_ctl_path = Path(inkling_ctl_path)
        self.inkling_base_url = inkling_base_url.rstrip("/")
        self.restore_policy = restore_policy
        self.unit_name = unit_name
        self.command_timeout_s = command_timeout_s

        self.initial_inkling_active: bool = False
        self._current_owner: Literal["none", "ollama", "inkling"] = "none"
        self.gpu_switches: int = 0

    def _run_cmd(self, args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        logger.debug("Running command: %s", " ".join(args))
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_s,
            check=check,
        )

    def is_inkling_service_active(self) -> bool:
        """Check whether inkling-server.service is currently active via systemd --user."""
        try:
            proc = self._run_cmd(["systemctl", "--user", "is-active", self.unit_name])
            return proc.returncode == 0 and proc.stdout.strip() == "active"
        except (OSError, subprocess.SubprocessError):
            return False

    def is_inkling_healthy(self, timeout: float = 2.0) -> bool:
        """Check whether inkling_engine responds 200 on /health."""
        try:
            resp = httpx.get(f"{self.inkling_base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def __enter__(self) -> GpuLeaseManager:
        self.initial_inkling_active = self.is_inkling_service_active()
        if self.initial_inkling_active:
            self._current_owner = "inkling"
        logger.info(
            "GPU lease entered (initial inkling-server active=%s, policy=%s)",
            self.initial_inkling_active,
            self.restore_policy,
        )
        return self

    def acquire_ollama(self) -> None:
        """Ensure inkling-server.service is stopped so Ollama owns all GPU VRAM."""
        if self._current_owner == "ollama":
            return

        if self.is_inkling_service_active() or self.is_inkling_healthy(timeout=1.0):
            if not self.inkling_ctl_path.exists():
                raise RuntimeError(f"inkling-ctl not found at {self.inkling_ctl_path}")
            logger.info("Stopping inkling-server.service to free VRAM for Ollama...")
            proc = self._run_cmd([str(self.inkling_ctl_path), "stop"])
            if proc.returncode != 0:
                logger.warning("inkling-ctl stop returned %d: %s", proc.returncode, proc.stderr)
            # Wait briefly for port 8001 to stop responding
            for _ in range(10):
                if not self.is_inkling_healthy(timeout=0.5):
                    break
                time.sleep(0.3)
            self.gpu_switches += 1

        self._current_owner = "ollama"

    def unload_ollama(self) -> None:
        """Ask inkling-ctl to unload any resident Ollama models from VRAM."""
        if self.inkling_ctl_path.exists():
            self._run_cmd([str(self.inkling_ctl_path), "unload-ollama"])

    def acquire_inkling(self) -> bool:
        """Start inkling-server.service via inkling-ctl (unloads Ollama first).

        Returns True if inkling_engine is healthy on /health, False otherwise.
        """
        if self.is_inkling_healthy(timeout=1.5):
            self._current_owner = "inkling"
            return True

        if not self.inkling_ctl_path.exists():
            logger.warning("inkling-ctl not found at %s; cannot start Inkling.", self.inkling_ctl_path)
            return False

        logger.info("Starting inkling-server.service via inkling-ctl start...")
        try:
            proc = self._run_cmd([str(self.inkling_ctl_path), "start"])
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("Failed to execute inkling-ctl start: %s", exc)
            return False

        if proc.returncode != 0:
            logger.error("inkling-ctl start failed (%d): %s", proc.returncode, proc.stderr.strip())
            return False

        for _ in range(20):
            if self.is_inkling_healthy(timeout=1.0):
                self._current_owner = "inkling"
                self.gpu_switches += 1
                return True
            time.sleep(0.5)

        logger.error("inkling-server.service started but /health did not respond in time.")
        return False

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.restore_policy == "keep_inkling":
                if not self.is_inkling_healthy(timeout=1.0):
                    self.acquire_inkling()
            elif self.restore_policy == "idle":
                if self.is_inkling_service_active() or self.is_inkling_healthy(timeout=1.0):
                    if self.inkling_ctl_path.exists():
                        self._run_cmd([str(self.inkling_ctl_path), "stop"])
                self.unload_ollama()
            else:  # "restore"
                if self.initial_inkling_active:
                    if not self.is_inkling_healthy(timeout=1.0):
                        logger.info("Restoring inkling-server.service to pre-run active state...")
                        self.acquire_inkling()
                else:
                    if self.is_inkling_service_active() or self.is_inkling_healthy(timeout=1.0):
                        logger.info("Stopping inkling-server.service to restore pre-run idle state...")
                        if self.inkling_ctl_path.exists():
                            self._run_cmd([str(self.inkling_ctl_path), "stop"])
                    self.unload_ollama()
        except Exception as restore_exc:  # noqa: BLE001
            logger.error("Error while restoring GPU state: %s", restore_exc)
