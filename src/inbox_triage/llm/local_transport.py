"""In-process httpx.BaseTransport bridging google-genai SDK calls to local Ollama & inkling_engine."""

from __future__ import annotations

import json
import re
from typing import Any, Literal
from urllib.parse import unquote

import httpx

_MODEL_PATH_RE = re.compile(r"/models/([^/:]+(?::[^/:]+)?):generateContent$")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_GEMINI_TYPE_MAP: dict[str, str] = {
    "OBJECT": "object",
    "ARRAY": "array",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "NULL": "null",
}


def normalize_gemini_schema(schema: Any) -> Any:
    """Recursively convert Gemini OpenAPI schema conventions to standard JSON Schema."""
    if isinstance(schema, list):
        return [normalize_gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = {}
    for key, val in schema.items():
        if key == "propertyOrdering":
            continue
        if key == "type" and isinstance(val, str):
            out["type"] = _GEMINI_TYPE_MAP.get(val.upper(), val.lower())
        elif key == "nullable" and val is True:
            # Handled after loop by expanding type to [type, "null"] if present
            continue
        else:
            out[key] = normalize_gemini_schema(val)

    if schema.get("nullable") is True and "type" in out and isinstance(out["type"], str):
        out["type"] = [out["type"], "null"]
    return out


def extract_json_text(raw: str) -> str:
    """Strip Markdown fences or surrounding commentary to isolate valid JSON."""
    text = raw.strip()
    if not text:
        return text

    # Direct valid JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Fenced code block
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            text = candidate

    # Fallback: locate outermost {...} or [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return raw.strip()


def _parse_think_header(raw_value: str) -> bool | str:
    lower = raw_value.strip().lower()
    if lower in ("false", "0", "no", "off", "none"):
        return False
    if lower in ("true", "1", "yes", "on"):
        return True
    return lower


class LocalGenAITransport(httpx.BaseTransport):
    """Translates google-genai REST requests to local Ollama or inkling_engine /api/chat.

    Supported per-request control headers (via google-genai HttpOptions.headers):
      - X-Local-Think: false | true | low | medium | xhigh (Ollama dialect)
      - X-Local-Reasoning-Effort: none | minimal | low | medium | high | xhigh (Inkling dialect)
      - X-Local-Num-Ctx: integer context size (Ollama dialect)
      - X-Local-Keep-Alive: duration string or "0" (Ollama dialect)
    """

    def __init__(
        self,
        backend_url: str,
        dialect: Literal["ollama", "inkling"] = "ollama",
        timeout: float = 300.0,
        upstream_client: httpx.Client | None = None,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.dialect = dialect
        self._owns_client = upstream_client is None
        self._client = upstream_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _extract_model(self, url_path: str) -> str:
        decoded_path = unquote(url_path)
        match = _MODEL_PATH_RE.search(decoded_path)
        if not match:
            raise ValueError(f"Unsupported google-genai path for local transport: {decoded_path}")
        return match.group(1)

    def _build_backend_payload(
        self,
        model: str,
        gemini_req: dict[str, Any],
        headers: httpx.Headers,
    ) -> tuple[dict[str, Any], bool]:
        messages: list[dict[str, Any]] = []

        # 1. System instruction
        sys_parts = (gemini_req.get("systemInstruction") or {}).get("parts") or []
        sys_text = "\n".join(p.get("text", "") for p in sys_parts if p.get("text")).strip()

        gen_cfg = gemini_req.get("generationConfig") or {}
        raw_schema = gen_cfg.get("responseJsonSchema") or gen_cfg.get("responseSchema")
        expects_json = (
            gen_cfg.get("responseMimeType") == "application/json" or raw_schema is not None
        )
        normalized_schema = normalize_gemini_schema(raw_schema) if raw_schema is not None else None

        # For inkling_engine (no sampler-level JSON grammar), append stable JSON schema contract
        # to the system message so KV prefix caching still hits across calls in the same stage.
        if self.dialect == "inkling" and expects_json:
            schema_clause = (
                "\n\nRespond ONLY with a valid JSON object matching this exact JSON Schema "
                "(no markdown fences, no commentary):\n"
                + json.dumps(normalized_schema, separators=(",", ":"), sort_keys=True)
                if normalized_schema is not None
                else "\n\nRespond ONLY with a valid JSON object (no markdown fences, no commentary)."
            )
            sys_text = (sys_text + schema_clause).strip()

        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        # 2. Conversation contents
        for content in gemini_req.get("contents") or []:
            role = content.get("role", "user")
            chat_role = "assistant" if role == "model" else "user"
            text_chunks: list[str] = []
            media_b64: list[str] = []

            for part in content.get("parts") or []:
                if "text" in part and part["text"] is not None:
                    text_chunks.append(str(part["text"]))
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    media_b64.append(inline["data"])

            msg: dict[str, Any] = {"role": chat_role, "content": "\n".join(text_chunks)}
            if media_b64 and self.dialect == "ollama":
                msg["images"] = media_b64
            messages.append(msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        options: dict[str, Any] = {}
        if "temperature" in gen_cfg and gen_cfg["temperature"] is not None:
            options["temperature"] = float(gen_cfg["temperature"])
        if "maxOutputTokens" in gen_cfg and gen_cfg["maxOutputTokens"] is not None:
            options["num_predict"] = int(gen_cfg["maxOutputTokens"])

        if self.dialect == "ollama":
            if normalized_schema is not None:
                payload["format"] = normalized_schema
            elif expects_json:
                payload["format"] = "json"

            if "x-local-num-ctx" in headers:
                options["num_ctx"] = int(headers["x-local-num-ctx"])
            if "x-local-keep-alive" in headers:
                raw_ka = headers["x-local-keep-alive"].strip()
                payload["keep_alive"] = 0 if raw_ka == "0" else raw_ka
            if "x-local-think" in headers:
                payload["think"] = _parse_think_header(headers["x-local-think"])
        else:
            # Inkling dialect (/home/ludo/projects/InklingOptimization/src/server.h)
            if "x-local-reasoning-effort" in headers:
                payload["reasoning_effort"] = headers["x-local-reasoning-effort"].strip().lower()

        if options:
            payload["options"] = options

        return payload, expects_json

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        model = self._extract_model(request.url.path)
        gemini_req = json.loads(request.content.decode("utf-8")) if request.content else {}
        backend_payload, expects_json = self._build_backend_payload(
            model=model,
            gemini_req=gemini_req,
            headers=request.headers,
        )

        upstream_resp = self._client.post(
            f"{self.backend_url}/api/chat",
            json=backend_payload,
        )

        if upstream_resp.status_code >= 400:
            err_body = {
                "error": {
                    "code": upstream_resp.status_code,
                    "message": f"Local {self.dialect} backend error: {upstream_resp.text}",
                    "status": "INTERNAL",
                }
            }
            return httpx.Response(
                status_code=upstream_resp.status_code,
                json=err_body,
                request=request,
            )

        data = upstream_resp.json()
        message = data.get("message") or {}
        content_text = str(message.get("content") or "")
        if expects_json:
            content_text = extract_json_text(content_text)

        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)

        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": content_text}],
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens,
            },
            "modelVersion": model,
        }
        return httpx.Response(status_code=200, json=gemini_resp, request=request)
