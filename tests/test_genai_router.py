"""Unit tests for GenAIRouter (Vertex AI / Gemini API initialization, parsing, and retries)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from inbox_triage.config import Settings
from inbox_triage.llm.router import DEFAULT_HTTP_TIMEOUT_MS, GenAIRouter
from inbox_triage.models import Stage1Classification, TriageCategory


class _SimpleSchema(BaseModel):
    label: str
    score: float


def _fake_response(
    *,
    parsed: BaseModel | None = None,
    text: str | None = None,
    prompt_tokens: int = 120,
    completion_tokens: int = 45,
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed=parsed,
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
    )


class TestGenAIRouterInit:
    def test_init_vertexai_mode(self, test_settings: Settings) -> None:
        vertex_settings = test_settings.model_copy(
            update={
                "use_vertexai": True,
                "gcp_project_id": "gmail-inbox-triage-515705",
                "vertex_location": "global",
                "gemini_api_key": None,
            }
        )
        fake_client = MagicMock()
        with patch("inbox_triage.llm.router.genai.Client", return_value=fake_client) as mock_cls:
            router = GenAIRouter(vertex_settings)

        mock_cls.assert_called_once_with(
            vertexai=True,
            project="gmail-inbox-triage-515705",
            location="global",
            http_options=types.HttpOptions(timeout=DEFAULT_HTTP_TIMEOUT_MS),
        )
        assert DEFAULT_HTTP_TIMEOUT_MS == 60_000
        assert router.backend_name == "vertexai"
        assert router.client is fake_client

    def test_init_gemini_api_mode(self, test_settings: Settings) -> None:
        api_settings = test_settings.model_copy(
            update={"use_vertexai": False, "gemini_api_key": "AIza-test-secret"}
        )
        fake_client = MagicMock()
        with patch("inbox_triage.llm.router.genai.Client", return_value=fake_client) as mock_cls:
            router = GenAIRouter(api_settings)

        mock_cls.assert_called_once_with(
            api_key="AIza-test-secret",
            http_options=types.HttpOptions(timeout=60_000),
        )
        assert router.backend_name == "gemini_api"
        assert router.client is fake_client

    def test_init_missing_api_key_raises_runtime_error(self, test_settings: Settings) -> None:
        bad_settings = test_settings.model_copy(
            update={"use_vertexai": False, "gemini_api_key": None}
        )
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GenAIRouter(bad_settings)

    def test_init_with_injected_client_skips_sdk_constructor(self, test_settings: Settings) -> None:
        injected = MagicMock()
        with patch("inbox_triage.llm.router.genai.Client") as mock_cls:
            router = GenAIRouter(test_settings, client=injected)
        mock_cls.assert_not_called()
        assert router.client is injected


class TestGenAIRouterStructuredGeneration:
    def test_generate_structured_uses_response_parsed(self, test_settings: Settings) -> None:
        expected = Stage1Classification(
            category=TriageCategory.URGENT,
            confidence=0.96,
            one_line_summary="Hosting invoice overdue.",
            needs_tech_extract=False,
            needs_deep_analysis=True,
            reason="Explicit deadline today at 5 PM PT.",
        )
        client = MagicMock()
        client.models.generate_content.return_value = _fake_response(
            parsed=expected,
            text=None,
            prompt_tokens=210,
            completion_tokens=64,
        )

        router = GenAIRouter(test_settings, client=client)
        result, stats = router.generate_structured(
            model="gemini-3.5-flash-lite",
            stage="stage1_fast_triage",
            system_prompt="Classify email.",
            contents="Subject: Urgent",
            response_schema=Stage1Classification,
            temperature=0.0,
            max_output_tokens=1024,
        )

        assert result == expected
        assert stats.backend == "gemini_api"
        assert stats.model == "gemini-3.5-flash-lite"
        assert stats.stage == "stage1_fast_triage"
        assert stats.prompt_tokens == 210
        assert stats.completion_tokens == 64
        assert stats.latency_ms >= 0.0

        call_kwargs = client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash-lite"
        assert call_kwargs["contents"] == "Subject: Urgent"
        cfg = call_kwargs["config"]
        assert cfg.system_instruction == "Classify email."
        assert cfg.temperature == 0.0
        assert cfg.max_output_tokens == 1024
        assert cfg.response_mime_type == "application/json"
        assert cfg.response_schema is Stage1Classification

    def test_generate_structured_falls_back_to_response_text_json(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _fake_response(
            parsed=None,
            text='  {"label": "ok", "score": 0.88}  \n',
            prompt_tokens=50,
            completion_tokens=20,
        )
        vertex_settings = test_settings.model_copy(update={"use_vertexai": True})
        router = GenAIRouter(vertex_settings, client=client)

        result, stats = router.generate_structured(
            model="gemini-4-carbon",
            stage="stage2_test",
            system_prompt="Return JSON.",
            contents="hello",
            response_schema=_SimpleSchema,
        )

        assert result == _SimpleSchema(label="ok", score=0.88)
        assert stats.backend == "vertexai"

    def test_generate_structured_retries_on_validation_error_with_feedback(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = [
            _fake_response(parsed=None, text='{"label": "missing-score"}'),
            _fake_response(parsed=None, text='{"label": "fixed", "score": 0.91}'),
        ]
        router = GenAIRouter(test_settings, client=client)

        result, _ = router.generate_structured(
            model="gemini-4-carbon",
            stage="stage2_test",
            system_prompt="Base prompt.",
            contents="payload",
            response_schema=_SimpleSchema,
            max_retries=3,
        )

        assert result == _SimpleSchema(label="fixed", score=0.91)
        assert client.models.generate_content.call_count == 2
        first_cfg = client.models.generate_content.call_args_list[0].kwargs["config"]
        second_cfg = client.models.generate_content.call_args_list[1].kwargs["config"]
        assert first_cfg.system_instruction == "Base prompt."
        assert "failed schema validation" in second_cfg.system_instruction
        assert "Base prompt." in second_cfg.system_instruction

    def test_generate_structured_raises_runtime_error_when_validation_exhausted(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _fake_response(parsed=None, text="not-json")
        router = GenAIRouter(test_settings, client=client)

        with pytest.raises(RuntimeError, match="Structured generation failed after 2 attempts") as exc_info:
            router.generate_structured(
                model="gemini-3.5-flash-lite",
                stage="stage1",
                system_prompt="Prompt",
                contents="Body",
                response_schema=_SimpleSchema,
                max_retries=2,
            )

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert client.models.generate_content.call_count == 2

    def test_generate_structured_retries_empty_output(self, test_settings: Settings) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = [
            _fake_response(parsed=None, text=""),
            _fake_response(parsed=_SimpleSchema(label="recovered", score=1.0)),
        ]
        router = GenAIRouter(test_settings, client=client)

        result, _ = router.generate_structured(
            model="gemini-4-neon",
            stage="stage3",
            system_prompt="Synthesize.",
            contents="cards",
            response_schema=_SimpleSchema,
            max_retries=2,
        )

        assert result.label == "recovered"
        assert client.models.generate_content.call_count == 2

    def test_generate_structured_exponential_backoff_on_429_and_503(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = [
            errors.APIError(429, {"error": {"message": "Rate limited"}}),
            errors.APIError(503, {"error": {"message": "Service unavailable"}}),
            _fake_response(parsed=_SimpleSchema(label="success", score=0.99)),
        ]
        router = GenAIRouter(test_settings, client=client)

        with patch("inbox_triage.llm.router.time.sleep") as mock_sleep:
            result, _ = router.generate_structured(
                model="gemini-4-carbon",
                stage="stage2",
                system_prompt="Analyze.",
                contents="content",
                response_schema=_SimpleSchema,
                max_retries=3,
            )

        assert result == _SimpleSchema(label="success", score=0.99)
        assert client.models.generate_content.call_count == 3
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 2.0]

    def test_generate_structured_reraises_retryable_error_after_exhaustion(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = errors.APIError(
            503, {"error": {"message": "Overloaded"}}
        )
        router = GenAIRouter(test_settings, client=client)

        with patch("inbox_triage.llm.router.time.sleep") as mock_sleep:
            with pytest.raises(errors.APIError) as exc_info:
                router.generate_structured(
                    model="gemini-4-neon",
                    stage="stage3",
                    system_prompt="Brief.",
                    contents="cards",
                    response_schema=_SimpleSchema,
                    max_retries=3,
                )

        assert exc_info.value.code == 503
        assert client.models.generate_content.call_count == 3
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 2.0]

    def test_generate_structured_raises_immediately_on_non_retryable_400(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = errors.APIError(
            400, {"error": {"message": "Invalid argument"}}
        )
        router = GenAIRouter(test_settings, client=client)

        with patch("inbox_triage.llm.router.time.sleep") as mock_sleep:
            with pytest.raises(errors.APIError) as exc_info:
                router.generate_structured(
                    model="gemini-4-carbon",
                    stage="stage2",
                    system_prompt="Analyze.",
                    contents="bad payload",
                    response_schema=_SimpleSchema,
                    max_retries=3,
                )

        assert exc_info.value.code == 400
        assert client.models.generate_content.call_count == 1
        mock_sleep.assert_not_called()

    def test_generate_structured_retries_on_httpx_timeout_and_transport_error(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = [
            httpx.ReadTimeout("socket read timed out"),
            httpx.ConnectError("connection reset"),
            _fake_response(parsed=_SimpleSchema(label="recovered-net", score=0.97)),
        ]
        router = GenAIRouter(test_settings, client=client)

        with patch("inbox_triage.llm.router.time.sleep") as mock_sleep:
            result, _ = router.generate_structured(
                model="gemini-4-carbon",
                stage="stage2_deep_multimodal",
                system_prompt="Analyze.",
                contents="content",
                response_schema=_SimpleSchema,
                max_retries=3,
            )

        assert result == _SimpleSchema(label="recovered-net", score=0.97)
        assert client.models.generate_content.call_count == 3
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 2.0]

    def test_generate_structured_reraises_timeout_after_exhaustion(
        self, test_settings: Settings
    ) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = httpx.ReadTimeout("timed out")
        router = GenAIRouter(test_settings, client=client)

        with patch("inbox_triage.llm.router.time.sleep") as mock_sleep:
            with pytest.raises(httpx.ReadTimeout):
                router.generate_structured(
                    model="gemini-4-neon",
                    stage="stage3_executive_synthesis",
                    system_prompt="Brief.",
                    contents="cards",
                    response_schema=_SimpleSchema,
                    max_retries=2,
                )

        assert client.models.generate_content.call_count == 2
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0]
