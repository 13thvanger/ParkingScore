import json
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


def _criteria() -> CriteriaSet:
    return CriteriaSet(("criterion",), "sha256:" + "1" * 64)


def _response(send_probability: int = 73, criterion_id: str = "LEGACY001") -> str:
    return json.dumps(
        {
            "schema_version": 2,
            "send_probability": send_probability,
            "lawn_probability": 80,
            "evidence_quality_probability": 70,
            "target_identity_probability": 99,
            "criteria": [
                {
                    "id": criterion_id,
                    "category": "decision",
                    "probability": 75,
                    "satisfied": True,
                    "evidence": "visible",
                }
            ],
            "comment": "ok",
        }
    )


def _assess(client: AIClient):
    observation = SimpleNamespace(
        plate="O716MP48",
        place="test address",
        camera="camera-1",
        plate_box=None,
    )
    image = PreparedImage("data:image/jpeg;base64,AA==", 1, 1, 1)
    return client.assess(observation, _criteria(), image)


def test_parse_assessment_accepts_fenced_v2_json() -> None:
    result = parse_assessment(f"```json\n{_response(82)}\n```", _criteria())

    assert result.send_probability == 82
    assert result.lawn_probability == 80
    assert result.evidence_quality_probability == 70
    assert result.target_identity_probability == 99
    assert result.comment == "ok"


def test_parse_assessment_rejects_missing_or_out_of_range_probability() -> None:
    value = json.loads(_response())
    value.pop("lawn_probability")
    with pytest.raises(AIError, match="lawn_probability"):
        parse_assessment(json.dumps(value), _criteria())

    value = json.loads(_response())
    value["send_probability"] = 101
    with pytest.raises(AIError, match="between 0 and 100"):
        parse_assessment(json.dumps(value), _criteria())


def test_parse_assessment_rejects_unknown_duplicate_and_missing_ids() -> None:
    with pytest.raises(AIError, match="unknown criterion"):
        parse_assessment(_response(criterion_id="OTHER"), _criteria())

    value = json.loads(_response())
    value["criteria"].append(dict(value["criteria"][0]))
    with pytest.raises(AIError, match="duplicate criterion"):
        parse_assessment(json.dumps(value), _criteria())

    value["criteria"] = []
    with pytest.raises(AIError, match="missing criteria"):
        parse_assessment(json.dumps(value), _criteria())


def test_client_accepts_content_as_text_parts() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": _response()}]
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
                        "message": {"content": _response(64)},
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


def test_prompt_names_target_dimensions_and_stable_criteria_id() -> None:
    request_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _response()}}]},
        )

    client = AIClient(_settings(), httpx.MockTransport(handler))
    try:
        _assess(client)
    finally:
        client.close()

    prompt = request_body["messages"][1]["content"][0]["text"]
    assert "O716MP48" in prompt
    assert "TARGET" in prompt
    assert "send_probability" in prompt
    assert "[decision:LEGACY001]" in prompt
    assert "lawn — только на lawn_probability" in prompt
    assert "evidence — только на evidence_quality_probability" in prompt
    assert "identity — только на target_identity_probability" in prompt
    assert "не среднее арифметическое" in prompt
    assert "блики и отражения" in prompt
    assert "хороший отдельный признак не должен компенсировать" in prompt
    assert "видно менее одной четверти целевого автомобиля" in prompt
    assert "считается не отрицательным, а неустановимым" in prompt
    assert "key" not in json.dumps(client.request_parameters)
