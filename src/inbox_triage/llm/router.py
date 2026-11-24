"""Pure google-genai router supporting Gemini API (key) and Vertex AI (IAM)."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from inbox_triage.config import Settings
from inbox_triage.models import ModelCallStats

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_HTTP_TIMEOUT_MS = 60_000  # 60 seconds per GenAI request


class GenAIRouter:
    """Routes structured generation calls through the official google-genai SDK.

    Supports two backends via `USE_VERTEXAI`:
      - `USE_VERTEXAI=false` (default): Gemini Developer API via `GEMINI_API_KEY`.
      - `USE_VERTEXAI=true`: Vertex AI via GCP IAM / Service Account (`project` + `location`).
    """

    def __init__(
        self,
        settings: Settings,
        client: genai.Client | None = None,
        timeout_ms: int = DEFAULT_HTTP_TIMEOUT_MS,
    ) -> None:
        self.settings = settings
        self.backend_name: Literal["vertexai", "gemini_api"] = (
            "vertexai" if settings.use_vertexai else "gemini_api"
        )
        http_opts = types.HttpOptions(timeout=timeout_ms)
        if client is not None:
            self.client = client
        elif settings.use_vertexai:
            self.client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.vertex_location,
                http_options=http_opts,
            )
        else:
            if not settings.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is required when USE_VERTEXAI=false. "
                    "Set GEMINI_API_KEY or enable USE_VERTEXAI=true."
                )
            self.client = genai.Client(api_key=settings.gemini_api_key, http_options=http_opts)

    def generate_structured(
        self,
        *,
        model: str,
        stage: str,
        system_prompt: str,
        contents: Any,
        response_schema: type[T],
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        max_retries: int = 3,
    ) -> tuple[T, ModelCallStats]:
        """Call a Gemini model and return a validated Pydantic object + call telemetry."""
        last_err: Exception | None = None
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            effective_system = system_prompt
            if attempt > 1 and isinstance(last_err, (ValidationError, ValueError)):
                effective_system = (
                    f"{system_prompt}\n\nIMPORTANT: Your previous output failed schema validation "
                    f"({last_err}). Output ONLY a strictly valid JSON object matching every required field."
                )

            config = types.GenerateContentConfig(
                system_instruction=effective_system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            )

            t0 = time.perf_counter()
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                parsed: T | None = None
                if response.parsed is not None and isinstance(response.parsed, response_schema):
                    parsed = response.parsed
                elif response.text:
                    parsed = response_schema.model_validate_json(response.text.strip())

                if parsed is None:
                    raise ValueError("Model returned empty structured output.")

                usage = response.usage_metadata
                stats = ModelCallStats(
                    backend=self.backend_name,
                    model=model,
                    stage=stage,
                    prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                    completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                    latency_ms=round(latency_ms, 1),
                )
                return parsed, stats
            except errors.APIError as exc:
                last_err = exc
                status_code = getattr(exc, "code", None)
                if status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                    logger.warning(
                        "[%s/%s] Retryable API error %s on %s (attempt %d/%d); sleeping %.1fs",
                        self.backend_name,
                        model,
                        status_code,
                        stage,
                        attempt,
                        max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning(
                        "[%s/%s] Network/timeout error on %s (attempt %d/%d): %s; retrying in %.1fs",
                        self.backend_name,
                        model,
                        stage,
                        attempt,
                        max_retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise
            except (ValidationError, ValueError) as exc:
                last_err = exc
                logger.warning(
                    "[%s/%s] Validation attempt %d/%d failed on %s: %s",
                    self.backend_name,
                    model,
                    attempt,
                    max_retries,
                    stage,
                    exc,
                )

        raise RuntimeError(
            f"Structured generation failed after {max_retries} attempts "
            f"(backend={self.backend_name}, model={model}, stage={stage}): {last_err}"
        ) from last_err
