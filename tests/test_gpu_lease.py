"""Unit tests for GpuLeaseManager GPU exclusivity and state restoration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from inbox_triage.llm.gpu_lease import GpuLeaseManager


class FakeGpuEnvironment:
    """Simulates systemctl --user, inkling-ctl, and /health endpoint state."""

    def __init__(self, *, initial_active: bool, start_succeeds: bool = True, healthy_after_start: int = 1) -> None:
        self.active = initial_active
        self.healthy = initial_active
        self.start_succeeds = start_succeeds
        self.healthy_after_start = healthy_after_start
        self._polls_since_start = 0
        self.commands: list[list[str]] = []

    def run_cmd(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(args))
        if args[:3] == ["systemctl", "--user", "is-active"]:
            if self.active:
                return subprocess.CompletedProcess(args, 0, stdout="active\n", stderr="")
            return subprocess.CompletedProcess(args, 3, stdout="inactive\n", stderr="")

        subcmd = args[-1]
        if subcmd == "stop":
            self.active = False
            self.healthy = False
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if subcmd == "start":
            if not self.start_succeeds:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="failed to start")
            self.active = True
            self._polls_since_start = 0
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if subcmd == "unload-ollama":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command invoked: {args}")

    def httpx_get(self, url: str, **kwargs: Any) -> httpx.Response:
        assert url.endswith("/health")
        if self.active and not self.healthy:
            self._polls_since_start += 1
            if self._polls_since_start >= self.healthy_after_start:
                self.healthy = True
        if self.healthy:
            return httpx.Response(200, json={"status": "ok"})
        raise httpx.ConnectError("Connection refused")

    def assert_never_touched_hermes(self) -> None:
        for cmd in self.commands:
            assert "use-inkling" not in cmd, f"Forbidden command use-inkling called: {cmd}"
            assert "use-ollama" not in cmd, f"Forbidden command use-ollama called: {cmd}"


def _make_ctl(tmp_path: Path) -> Path:
    ctl = tmp_path / "inkling-ctl"
    ctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return ctl


def test_zero_switch_when_inkling_already_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=True)
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
        assert lease.initial_inkling_active is True
        assert lease.acquire_inkling() is True
        assert lease.gpu_switches == 0

    env.assert_never_touched_hermes()
    ctl_calls = [c for c in env.commands if str(ctl) in c]
    assert ctl_calls == []
    assert env.active is True


def test_zero_switch_when_ollama_acquired_from_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=False)
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
        assert lease.initial_inkling_active is False
        lease.acquire_ollama()
        assert lease.gpu_switches == 0

    env.assert_never_touched_hermes()
    # On exit from initial_active=False, only unload-ollama is called (never stop/start)
    ctl_subcmds = [c[-1] for c in env.commands if str(ctl) in c]
    assert "stop" not in ctl_subcmds
    assert "start" not in ctl_subcmds
    assert "unload-ollama" in ctl_subcmds


def test_acquire_ollama_stops_active_inkling_and_restores_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=True, healthy_after_start=2)
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
        lease.acquire_ollama()
        assert env.active is False
        assert lease.gpu_switches == 1
        # Calling acquire_ollama again is a no-op
        lease.acquire_ollama()
        assert lease.gpu_switches == 1

    # Restored to initial active state
    env.assert_never_touched_hermes()
    assert env.active is True
    assert env.healthy is True
    ctl_subcmds = [c[-1] for c in env.commands if str(ctl) in c]
    assert ctl_subcmds == ["stop", "start"]


def test_acquire_inkling_starts_service_and_restores_idle_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=False, healthy_after_start=2)
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
        assert lease.acquire_inkling() is True
        assert env.active is True
        assert env.healthy is True
        assert lease.gpu_switches == 1

    # Restored back to inactive + unloaded Ollama
    env.assert_never_touched_hermes()
    assert env.active is False
    ctl_subcmds = [c[-1] for c in env.commands if str(ctl) in c]
    assert ctl_subcmds == ["start", "stop", "unload-ollama"]


def test_restores_active_inkling_even_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=True)
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="boom"):
        with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
            lease.acquire_ollama()
            assert env.active is False
            raise RuntimeError("boom")

    env.assert_never_touched_hermes()
    assert env.active is True
    assert env.healthy is True


def test_restores_idle_state_even_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=False)
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    with pytest.raises(ValueError, match="crash"):
        with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
            assert lease.acquire_inkling() is True
            assert env.active is True
            raise ValueError("crash")

    env.assert_never_touched_hermes()
    assert env.active is False


def test_restore_policy_keep_inkling_and_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctl = _make_ctl(tmp_path)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    # keep_inkling: starts Inkling on exit even if initially inactive
    env_keep = FakeGpuEnvironment(initial_active=False)
    monkeypatch.setattr(subprocess, "run", env_keep.run_cmd)
    monkeypatch.setattr(httpx, "get", env_keep.httpx_get)
    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="keep_inkling") as lease:
        lease.acquire_ollama()
    env_keep.assert_never_touched_hermes()
    assert env_keep.active is True

    # idle: stops Inkling and unloads Ollama on exit even if initially active
    env_idle = FakeGpuEnvironment(initial_active=True)
    monkeypatch.setattr(subprocess, "run", env_idle.run_cmd)
    monkeypatch.setattr(httpx, "get", env_idle.httpx_get)
    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="idle") as lease:
        assert lease.acquire_inkling() is True
    env_idle.assert_never_touched_hermes()
    assert env_idle.active is False
    ctl_subcmds = [c[-1] for c in env_idle.commands if str(ctl) in c]
    assert ctl_subcmds == ["stop", "unload-ollama"]


def test_acquire_inkling_returns_false_when_ctl_missing_or_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeGpuEnvironment(initial_active=False, start_succeeds=False)
    monkeypatch.setattr(subprocess, "run", env.run_cmd)
    monkeypatch.setattr(httpx, "get", env.httpx_get)
    monkeypatch.setattr("inbox_triage.llm.gpu_lease.time.sleep", lambda _: None)

    # Missing inkling-ctl
    missing_ctl = tmp_path / "does-not-exist"
    with GpuLeaseManager(inkling_ctl_path=missing_ctl, restore_policy="restore") as lease:
        assert lease.acquire_inkling() is False

    # Existing inkling-ctl that returns exit code 1
    ctl = _make_ctl(tmp_path)
    with GpuLeaseManager(inkling_ctl_path=ctl, restore_policy="restore") as lease:
        assert lease.acquire_inkling() is False
