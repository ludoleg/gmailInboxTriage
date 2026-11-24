"""Unit tests for LocalGenAITransport and ModelRouter using real google.genai.Client instances."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from inbox_triage.config import Settings
from inbox_triage.llm.local_transport import (
    LocalGenAITransport,
    extract_json_text,
    normalize_gemini_schema,
)
from inbox_triage.llm.router import ModelRouter
from inbox_triage.models import MultimodalFinding, Stage1Classification, TriageCategory


def _make_genai_client(
    dialect: str,
    handler: Any,
    backend_url: str = "http://127.0.0.1:9999",
) -> tuple[genai.Client, LocalGenAITransport]:
    upstream = httpx.Client(transport=httpx.MockTransport(handler))
    transport = LocalGenAITransport(
        backend_url=backend_url,
        dialect=dialect,  # type: ignore[arg-type]
        upstream_client=upstream,
    )
    client = genai.Client(
        api_key=f"local-{dialect}",
        http_options=types.HttpOptions(
            base_url=f"http://local.{dialect}",
            api_version="v1beta",
            httpx_client=httpx.Client(transport=transport, timeout=30.0),
        ),
    )
    return client, transport


# --- Schema normalization & JSON extraction unit tests ---


def test_normalize_gemini_schema_converts_uppercase_types_and_nullable() -> None:
    gemini_schema = {
        "type": "OBJECT",
        "propertyOrdering": ["category", "note", "tags"],
        "properties": {
            "category": {"type": "STRING", "enum": ["URGENT", "FYI"]},
            "count": {"type": "INTEGER"},
            "score": {"type": "NUMBER"},
            "flag": {"type": "BOOLEAN"},
            "note": {"type": "STRING", "nullable": True},
            "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["category"],
    }

    normalized = normalize_gemini_schema(gemini_schema)

    assert normalized["type"] == "object"
    assert "propertyOrdering" not in normalized
    props = normalized["properties"]
    assert props["category"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["score"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert props["note"]["type"] == ["string", "null"]
    assert "nullable" not in props["note"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('Here is the JSON:\n```JSON\n{"a": 2}\n```\nDone.', '{"a": 2}'),
        ('Sure! {"a": 3} hope that helps.', '{"a": 3}'),
        ('Result: [1, 2, 3] trailing', "[1, 2, 3]"),
        ("", ""),
    ],
)
def test_extract_json_text_handles_fences_and_prose(raw: str, expected: str) -> None:
    assert extract_json_text(raw) == expected


# --- Real google.genai.Client -> LocalGenAITransport (Inkling dialect) ---


def test_inkling_dialect_reasoning_effort_schema_in_prompt_and_fence_stripping() -> None:
    captured: list[dict[str, Any]] = []

    def mock_inkling(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://127.0.0.1:8001/api/chat")
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        fenced_json = (
            "Here is the classification:\n"
            "```json\n"
            + json.dumps(
                {
                    "category": "URGENT",
                    "confidence": 0.94,
                    "one_line_summary": "Server outage requires immediate action.",
                    "needs_tech_extract": True,
                    "needs_deep_analysis": True,
                    "reason": "Production outage mentioned.",
                }
            )
            + "\n```"
        )
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": fenced_json},
                "prompt_eval_count": 210,
                "eval_count": 48,
            },
        )

    client, _ = _make_genai_client("inkling", mock_inkling, backend_url="http://127.0.0.1:8001")
    response = client.models.generate_content(
        model="inkling-2-mini",
        contents="Subject: Prod DB down",
        config=types.GenerateContentConfig(
            system_instruction="You are a triage classifier.",
            temperature=0.0,
            max_output_tokens=512,
            response_mime_type="application/json",
            response_schema=Stage1Classification,
            http_options=types.HttpOptions(headers={"X-Local-Reasoning-Effort": "low"}),
        ),
    )

    assert len(captured) == 1
    req = captured[0]
    assert req["model"] == "inkling-2-mini"
    assert req["stream"] is False
    assert req["reasoning_effort"] == "low"
    assert req["options"] == {"temperature": 0.0, "num_predict": 512}
    # Inkling dialect does NOT send Ollama's `format`, `think`, or `keep_alive`
    assert "format" not in req
    assert "think" not in req
    assert "keep_alive" not in req

    # System prompt must include the embedded lowercase JSON schema contract
    sys_msg = req["messages"][0]
    assert sys_msg["role"] == "system"
    assert "You are a triage classifier." in sys_msg["content"]
    assert '"one_line_summary"' in sys_msg["content"]
    assert '"type":"object"' in sys_msg["content"]

    # google-genai must automatically populate response.parsed after fence stripping!
    parsed = response.parsed
    assert isinstance(parsed, Stage1Classification)
    assert parsed.category == TriageCategory.URGENT
    assert parsed.confidence == pytest.approx(0.94)
    assert parsed.needs_tech_extract is True
    assert parsed.needs_deep_analysis is True

    # Usage metadata mapped from prompt_eval_count / eval_count
    assert response.usage_metadata is not None
    assert response.usage_metadata.prompt_token_count == 210
    assert response.usage_metadata.candidates_token_count == 48
    assert response.usage_metadata.total_token_count == 258


def test_inkling_dialect_does_not_attach_images_field(tiny_png_bytes: bytes) -> None:
    captured: list[dict[str, Any]] = []

    def mock_inkling(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client, _ = _make_genai_client("inkling", mock_inkling, backend_url="http://127.0.0.1:8001")
    client.models.generate_content(
        model="inkling-2-mini",
        contents=["Analyze", types.Part.from_bytes(data=tiny_png_bytes, mime_type="image/png")],
    )

    assert len(captured) == 1
    user_msg = captured[0]["messages"][-1]
    assert "images" not in user_msg


# --- Real google.genai.Client -> LocalGenAITransport (Ollama dialect) ---


def test_ollama_dialect_schema_format_headers_and_multimodal_images(tiny_png_bytes: bytes) -> None:
    captured: list[dict[str, Any]] = []

    def mock_ollama(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://127.0.0.1:11434/api/chat")
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "filename": "receipt.png",
                            "summary": "CloudHost invoice #INV-9921 for $480.00.",
                            "extracted_key_facts": ["Amount: $480.00", "Due today"],
                            "urgency_signal": "high",
                        }
                    ),
                },
                "prompt_eval_count": 320,
                "eval_count": 64,
            },
        )

    client, _ = _make_genai_client("ollama", mock_ollama, backend_url="http://127.0.0.1:11434")
    response = client.models.generate_content(
        model="gemma4:12b",
        contents=[
            "Inspect receipt.png",
            types.Part.from_bytes(data=tiny_png_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            system_instruction="Multimodal analyst.",
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=MultimodalFinding,
            http_options=types.HttpOptions(
                headers={
                    "X-Local-Think": "true",
                    "X-Local-Num-Ctx": "16384",
                    "X-Local-Keep-Alive": "0",
                }
            ),
        ),
    )

    assert len(captured) == 1
    req = captured[0]
    assert req["model"] == "gemma4:12b"
    assert req["stream"] is False
    assert req["think"] is True
    assert req["keep_alive"] == 0
    assert req["options"]["num_ctx"] == 16384
    assert req["options"]["num_predict"] == 1024
    assert req["options"]["temperature"] == pytest.approx(0.1)

    # Ollama `format` receives lowercase JSON Schema (no uppercase Gemini types)
    fmt = req["format"]
    assert isinstance(fmt, dict)
    assert fmt["type"] == "object"
    assert fmt["properties"]["filename"]["type"] == "string"
    assert fmt["properties"]["extracted_key_facts"]["type"] == "array"

    # System prompt does NOT get the inkling schema suffix
    sys_msg = req["messages"][0]
    assert sys_msg == {"role": "system", "content": "Multimodal analyst."}

    # User message contains base64-encoded image matching tiny_png_bytes
    user_msg = req["messages"][1]
    assert user_msg["role"] == "user"
    assert "Inspect receipt.png" in user_msg["content"]
    assert "images" in user_msg
    assert len(user_msg["images"]) == 1
    assert base64.b64decode(user_msg["images"][0]) == tiny_png_bytes

    # google-genai populates typed Pydantic object
    parsed = response.parsed
    assert isinstance(parsed, MultimodalFinding)
    assert parsed.filename == "receipt.png"
    assert parsed.urgency_signal == "high"
    assert "Amount: $480.00" in parsed.extracted_key_facts


@pytest.mark.parametrize(
    ("header_val", "expected"),
    [
        ("false", False),
        ("true", True),
        ("medium", "medium"),
        ("low", "low"),
    ],
)
def test_ollama_think_header_parsing(header_val: str, expected: bool | str) -> None:
    captured: list[dict[str, Any]] = []

    def mock_ollama(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"message": {"content": "hello"}})

    client, _ = _make_genai_client("ollama", mock_ollama, backend_url="http://127.0.0.1:11434")
    client.models.generate_content(
        model="gemma4:e2b",
        contents="ping",
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(headers={"X-Local-Think": header_val})
        ),
    )

    assert captured[0]["think"] == expected


# --- Error handling & ModelRouter retry ---


def test_local_transport_maps_upstream_http_error_to_genai_api_error() -> None:
    def mock_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    client, _ = _make_genai_client("ollama", mock_error, backend_url="http://127.0.0.1:11434")
    with pytest.raises(genai_errors.APIError) as exc_info:
        client.models.generate_content(model="gemma4:12b", contents="hello")

    assert exc_info.value.code == 503
    assert "model loading" in str(exc_info.value)


def test_local_transport_rejects_unsupported_path() -> None:
    transport = LocalGenAITransport(
        backend_url="http://127.0.0.1:8001",
        dialect="inkling",
        upstream_client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200))),
    )
    req = httpx.Request("GET", "http://local.inkling/v1beta/models")
    with pytest.raises(ValueError, match="Unsupported google-genai path"):
        transport.handle_request(req)


def test_model_router_retries_invalid_json_with_repair_prompt(test_settings: Settings) -> None:
    calls: list[dict[str, Any]] = []

    def flaky_inkling(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(200, json={"message": {"content": "not valid json at all"}})
        valid = {
            "category": "FYI",
            "confidence": 0.85,
            "one_line_summary": "Informational update.",
            "needs_tech_extract": False,
            "needs_deep_analysis": False,
            "reason": "No action needed.",
        }
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(valid)}, "prompt_eval_count": 50, "eval_count": 20},
        )

    transport = LocalGenAITransport(
        backend_url="http://127.0.0.1:8001",
        dialect="inkling",
        upstream_client=httpx.Client(transport=httpx.MockTransport(flaky_inkling)),
    )
    router = ModelRouter(settings=test_settings, inkling_transport=transport)

    result, stats = router.generate_structured(
        backend="inkling",
        model="inkling-2-mini",
        stage="stage1_triage",
        system_prompt="Classify email.",
        contents="Hello FYI",
        response_schema=Stage1Classification,
        reasoning_effort="none",
        max_retries=2,
    )

    assert len(calls) == 2
    assert "failed schema validation" in calls[1]["messages"][0]["content"]
    assert result.category == TriageCategory.FYI
    assert stats.backend == "inkling"
    assert stats.prompt_tokens == 50
    assert stats.completion_tokens == 20
