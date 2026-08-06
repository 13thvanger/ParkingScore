import pytest

from parking_score.ai_client import AIError, parse_assessment


def test_parse_assessment_accepts_fenced_json() -> None:
    result = parse_assessment(
        '```json\n{"probability": 81.6, "criteria": [], "comment": "ok"}\n```'
    )

    assert result.probability == 82
    assert result.comment == "ok"


def test_parse_assessment_rejects_out_of_range_probability() -> None:
    with pytest.raises(AIError):
        parse_assessment('{"probability": 101, "criteria": []}')
