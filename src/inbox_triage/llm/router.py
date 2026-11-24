"""Unified google-genai router spanning Inkling-2-Mini, Ollama, and Cloud Gemini 4."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, TypeVar

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from inbox_triage.config import Settings
from inbox_triage.llm.local_transport import LocalGenAITransport, extract_json_text
from inbox_triage.models import ModelCallStats

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelRouter:
    """Routes structured generation calls across local Inkling, local Ollama, and Cloud Gemini.

    All three backends are invoked through the official `google-genai` SDK (`genai.Client`),
    returning typed Pydantic objects via `response.parsed` (with validation fallback/retry).
    """

    def __init__(
        self,
        settings: Settings,
        ollama_transport: LocalGenAITransport | None = None,
        inkling_transport: LocalGenAITransport | None = None,
        cloud_client: genai.Client | None = None,
    ) -> None:
        self.settings = settings
        self._ollama_transport = ollama_transport or LocalGenAITransport(
            backend_url=settings.ollama_base_url,
            dialect="ollama",
            timeout=300.0,
        )
        self._inkling_transport = inkling_transport or LocalGenAITransport(
            backend_url=settings.inkling_base_url,
            dialect="inkling",
            timeout=300.0,
        )

        self.ollama_client = genai.Client(
            api_key="local-ollama",
            http_options=types.HttpOptions(
                base_url="http://local.ollama",
                api_version="v1beta",
                httpx_client=httpx.Client(transport=self._ollama_transport, timeout=300.0),
            ),
        )
        self.inkling_client = genai.Client(
            api_key="local-inkling",
            http_options=types.HttpOptions(
                base_url="http://local.inkling",
                api_version="v1beta",
                httpx_client=httpx.Client(transport=self._inkling_transport, timeout=300.0),
            ),
        )
        self.cloud_client: genai.Client | None = cloud_client
        if self.cloud_client is None and settings.gemini_api_key:
            self.cloud_client = genai.Client(api_key=settings.gemini_api_key)

    def _get_client(self, backend: Literal["inkling", "ollama", "cloud"]) -> genai.Client:
        if backend == "inkling":
            return self.inkling_client
        if backend == "ollama":
            return self.ollama_client
        if self.cloud_client is None:
            raise RuntimeError("Cloud Gemini requested but GEMINI_API_KEY is not configured.")
        return self.cloud_client

    def generate_structured(
        self,
        *,
        backend: Literal["inkling", "ollama", "cloud"],
        model: str,
        stage: str,
        system_prompt: str,
        contents: Any,
        response_schema: type[T],
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        think: bool | str | None = None,
        reasoning_effort: str | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        max_retries: int = 2,
    ) -> tuple[T, ModelCallStats]:
        """Call a model via google-genai and return a validated Pydantic instance + call stats."""
        client = self._get_client(backend)
        custom_headers: dict[str, str] = {}
        if backend == "ollama":
            if think is not None:
                custom_headers["X-Local-Think"] = (
                    "true" if think is True else "false" if think is False else str(think)
                )
            if num_ctx is not None:
                custom_headers["X-Local-Num-Ctx"] = str(num_ctx)
            if keep_alive is not None:
                custom_headers["X-Local-Keep-Alive"] = keep_alive
        elif backend == "inkling":
            if reasoning_effort is not None:
                custom_headers["X-Local-Reasoning-Effort"] = reasoning_effort

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            effective_system = system_prompt
            if attempt > 1:
                effective_system = (
                    f"{system_prompt}\n\nIMPORTANT: Your previous output failed schema validation "
                    f"({last_err}). Output ONLY a strictly valid JSON object matching every required field."
                )

            cfg_kwargs: dict[str, Any] = {
                "system_instruction": effective_system,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
            if custom_headers:
                cfg_kwargs["http_options"] = types.HttpOptions(headers=custom_headers)

            config = types.GenerateContentConfig(**cfg_kwargs)
            t0 = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                parsed: T | None = None
                if response.parsed is not None and isinstance(response.parsed, response_schema):
                    parsed = response.parsed
                elif response.text:
                    cleaned = extract_json_text(response.text)
                    parsed = response_schema.model_validate_json(cleaned)

                if parsed is None:
                    raise ValueError("Model returned empty structured output.")

                usage = response.usage_metadata
                stats = ModelCallStats(
                    backend=backend,
                    model=model,
                    stage=stage,
                    prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                    completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                    latency_ms=round(latency_ms, 1),
                )
                return parsed, stats
            except (ValidationError, ValueError, RuntimeError, httpx.HTTPError) as exc:
                last_err = exc
                logger.warning(
                    "[%s/%s] Attempt %d/%d failed for %s: %s",
                    backend,
                    model,
                    attempt,
                    max_retries,
                    stage,
                    exc,
                )

        raise RuntimeError(
            f"Structured generation failed after {max_retries} attempts "
            f"(backend={backend}, model={model}, stage={stage}): {last_err}"
        ) from last_err

    def unload_ollama_model(self, model: str) -> None:
        """Explicitly evict a model from Ollama VRAM using keep_alive=0."""
        try:
            httpx.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={"model": model, "messages": [], "keep_alive": 0, "stream": False},
                timeout=15.0,
            )
            logger.debug("Unloaded Ollama model %s from VRAM.", model)
        except httpx.HTTPError as exc:
            logger.warning("Failed to unload Ollama model %s: %s", model, exc)

    def list_ollama_models(self) -> list[str]:
        """Query Ollama /api/tags and return installed model names."""
        resp = httpx.get(f"{self.settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if "name" in m]
