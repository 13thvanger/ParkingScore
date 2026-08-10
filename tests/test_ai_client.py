from types import SimpleNamespace

import httpx
import pytest

from parking_score.ai_client import (
    AIClient,
    AIError,
    AITransientError,
    parse_assessment,
)
from parking_score.config import Settings
from parking_score.criteria import CriteriaSet
from parking_score.image_processor import PreparedImage


def _settings(**overrides) -> Settings:
    values = {
        "ftp_host": "example",
        "ftp_port": 21,
        "ftp_user": "user",
        "ftp_password": "password",
        "ai_api_key": "key",
        "ai_requests_per_minute": 0,
        "ai_retry_base_seconds": 0,
        "ai_retry_max_seconds": 0,
        "ai_retry_jitter_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _assess(client: AIClient):
    observation = SimpleNamespace(
        plate="O716MP48",
        place="test address",
        camera="camera-1",
        plate_box=None,
    )
    criteria = CriteriaSet(("criterion",), "criteria-v1")
    image = PreparedImage("data:image/jpeg;base64,AA==", 1, 1, 1)
    return client.assess(observation, criteria, image)


def test_parse_assessment_accepts_fenced_json() -> None:
    result = parse_assessment(
        '```json\n{"probability": 81.6, "criteria": [], "comment": "ok"}\n```'
    )

    assert result.probability == 82
    assert result.comment == "ok"


def test_parse_assessment_rejects_out_of_range_probability() -> None:
    with pytest.raises(AIError):
        parse_assessment('{"probability": 101, "criteria": []}')


def test_client_accepts_content_as_text_parts() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": '{"probability":73,"criteria":[]}',
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )
    client = AIClient(_settings(), transport=transport)
    try:
        assert _assess(client).probability == 73
    finally:
        client.close()


def test_client_reports_non_text_response_shape_as_transient() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": "hidden",
                            "tool_calls": [],
                        },
                        "finish_reason": "length",
                    }
                ]
            },
        )
    )
    client = AIClient(_settings(ai_request_retries=1), transport=transport)
    try:
        with pytest.raises(AITransientError, match="content_type=NoneType"):
            _assess(client)
    finally:
        client.close()


def test_client_retries_429_and_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="busy")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"probability":64,"criteria":[]}'
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    client = AIClient(_settings(ai_request_retries=2), httpx.MockTransport(handler))
    try:
        assert _assess(client).probability == 64
        assert calls == 2
    finally:
        client.close()
