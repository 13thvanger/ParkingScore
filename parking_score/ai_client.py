from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .config import Settings
from .criteria import CriteriaSet
from .image_processor import PreparedImage
from .models import Assessment, Observation

logger = logging.getLogger(__name__)


class AIError(RuntimeError):
    """Raised when the AI service cannot produce a valid assessment."""


class AITransientError(AIError):
    """Raised when an assessment should be retried without permanent failure."""


class _UnusableMessageError(AIError):
    def __init__(self, message: str, finish_reason: str | None = None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AIError("AI response does not contain a JSON object")


def _probability(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIError(f"AI response {field} must be a number")
    if not 0 <= float(value) <= 100:
        raise AIError(f"AI response {field} must be between 0 and 100")
    return round(float(value))


def parse_assessment(content: str, criteria: CriteriaSet) -> Assessment:
    value = _extract_json_object(content)
    if value.get("schema_version") != 2:
        raise AIError("AI response schema_version must be 2")
    send_probability = _probability(
        value.get("send_probability"), "send_probability"
    )
    lawn_probability = _probability(
        value.get("lawn_probability"), "lawn_probability"
    )
    evidence_probability = _probability(
        value.get("evidence_quality_probability"),
        "evidence_quality_probability",
    )
    identity_probability = _probability(
        value.get("target_identity_probability"),
        "target_identity_probability",
    )

    details = value.get("criteria")
    if not isinstance(details, list) or not all(
        isinstance(item, dict) for item in details
    ):
        raise AIError("AI response criteria must be an array of objects")
    expected = {item.id: item for item in criteria.definitions}
    normalized_details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for detail in details:
        returned_id = detail.get("id")
        criterion_id = _canonical_criterion_id(returned_id, expected)
        if criterion_id is None:
            raise AIError(
                f"AI response contains unknown criterion id: {returned_id}"
            )
        if criterion_id in seen:
            raise AIError(
                f"AI response contains duplicate criterion id: {criterion_id}"
            )
        seen.add(criterion_id)
        definition = expected[criterion_id]
        returned_category = detail.get("category")
        allowed_categories = (
            {"decision", "lawn", "evidence", "identity"}
            if definition.category == "decision"
            else {definition.category}
        )
        if returned_category not in allowed_categories:
            raise AIError(
                f"AI response category mismatch for criterion {criterion_id}"
            )
        if not isinstance(detail.get("satisfied"), bool):
            raise AIError(
                f"AI response satisfied must be boolean for {criterion_id}"
            )
        evidence = detail.get("evidence")
        if not isinstance(evidence, str):
            raise AIError(
                f"AI response evidence must be text for {criterion_id}"
            )
        normalized_details.append(
            {
                "id": criterion_id,
                "category": definition.category,
                "probability": _probability(
                    detail.get("probability"),
                    f"criteria[{criterion_id}].probability",
                ),
                "satisfied": detail["satisfied"],
                "evidence": evidence,
            }
        )
    missing = set(expected) - seen
    if missing:
        raise AIError(
            "AI response is missing criteria: " + ", ".join(sorted(missing))
        )
    comment = value.get("comment", "")
    if not isinstance(comment, str):
        raise AIError("AI response comment must be text")
    return Assessment(
        send_probability=send_probability,
        criteria_details=normalized_details,
        comment=comment,
        raw_response=content,
        lawn_probability=lawn_probability,
        evidence_quality_probability=evidence_probability,
        target_identity_probability=identity_probability,
    )


def _canonical_criterion_id(
    value: Any, expected: dict[str, Any]
) -> str | None:
    """Accept an exact ID or the unambiguous ``category:ID`` model variant."""
    if not isinstance(value, str):
        return None
    if value in expected:
        return value
    category, separator, criterion_id = value.partition(":")
    if not separator or criterion_id not in expected:
        return None
    definition = expected[criterion_id]
    if category == definition.category or (
        definition.category == "decision"
        and category in {"lawn", "evidence", "identity"}
    ):
        return criterion_id
    return None


def _message_text(body: Any) -> str:
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise AIError("AI response does not contain choices[0].message") from exc

    if isinstance(content, str):
        return content

    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    elif isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            parts.append(text)
    joined = "\n".join(part for part in parts if part.strip())
    if joined:
        return joined

    tool_calls = message.get("tool_calls")
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    finish_reason = choice.get("finish_reason")
    raise _UnusableMessageError(
        "AI response message content is not usable "
        f"(content_type={type(content).__name__}, "
        f"finish_reason={finish_reason!r}, "
        f"refusal={bool(message.get('refusal'))}, "
        f"tool_calls={tool_call_count}, "
        f"reasoning={isinstance(message.get('reasoning'), str)})",
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


class AIClient:
    def __init__(
        self, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.settings = settings
        self._request_gate_lock = threading.Lock()
        self._next_request_at = 0.0
        self._cooldown_until = 0.0
        self.client = httpx.Client(
            timeout=settings.ai_timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {settings.ai_api_key}",
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 ParkingScore/0.2"
                ),
            },
        )

    def close(self) -> None:
        self.client.close()

    def assess(
        self,
        observation: Observation,
        criteria: CriteriaSet,
        image: PreparedImage,
    ) -> Assessment:
        payload = self._payload(observation, criteria, image)
        last_error: Exception | None = None
        for attempt in range(1, self.settings.ai_request_retries + 1):
            try:
                self._wait_for_request_slot()
                response = self.client.post(self.settings.ai_api_url, json=payload)
                if response.status_code not in {200, 201}:
                    message = response.text[:500]
                    message_text = (
                        f"AI API returned HTTP {response.status_code}: {message}"
                    )
                    if (
                        response.status_code not in {403, 408, 409, 429}
                        and response.status_code < 500
                    ):
                        raise _FatalAIError(message_text)
                    raise _RetryableAIError(
                        message_text,
                        retry_after_seconds=_retry_after_seconds(response),
                        global_cooldown=response.status_code == 429,
                    )
                body = response.json()
                content = _message_text(body)
                return parse_assessment(content, criteria)
            except _FatalAIError as exc:
                raise AIError(str(exc)) from exc
            except (
                httpx.HTTPError,
                KeyError,
                ValueError,
                AIError,
                _RetryableAIError,
            ) as exc:
                last_error = exc
                if attempt >= self.settings.ai_request_retries:
                    break
                next_max_tokens = self._increase_token_limit_after_length(
                    payload, exc
                )
                delay = self._retry_delay(attempt, exc)
                if isinstance(exc, _RetryableAIError) and exc.global_cooldown:
                    self._extend_global_cooldown(delay)
                logger.warning(
                    "AI request attempt %d/%d failed; retrying in %.1fs%s: %s",
                    attempt,
                    self.settings.ai_request_retries,
                    delay,
                    (
                        f" with max_tokens={next_max_tokens}"
                        if next_max_tokens is not None
                        else ""
                    ),
                    exc,
                )
                time.sleep(delay)
        raise AITransientError(f"AI request failed after retries: {last_error}")

    def _increase_token_limit_after_length(
        self, payload: dict[str, Any], error: Exception
    ) -> int | None:
        if not (
            isinstance(error, _UnusableMessageError)
            and error.finish_reason == "length"
        ):
            return None
        current = int(payload["max_tokens"])
        ceiling = max(
            self.settings.ai_max_tokens,
            self.settings.ai_length_retry_max_tokens,
        )
        if current >= ceiling:
            return None
        increased = min(ceiling, max(current + 1, current * 2))
        payload["max_tokens"] = increased
        return increased

    def _wait_for_request_slot(self) -> None:
        requests_per_minute = self.settings.ai_requests_per_minute
        interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        while True:
            with self._request_gate_lock:
                now = time.monotonic()
                ready_at = max(self._next_request_at, self._cooldown_until)
                if now >= ready_at:
                    self._next_request_at = now + interval
                    return
                delay = ready_at - now
            time.sleep(delay)

    def _extend_global_cooldown(self, delay: float) -> None:
        with self._request_gate_lock:
            self._cooldown_until = max(
                self._cooldown_until, time.monotonic() + delay
            )

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        delay = min(
            self.settings.ai_retry_base_seconds * (2 ** (attempt - 1)),
            self.settings.ai_retry_max_seconds,
        )
        if isinstance(error, _RetryableAIError):
            retry_after = error.retry_after_seconds
            if retry_after is not None:
                delay = max(delay, retry_after)
        delay += random.uniform(0.0, self.settings.ai_retry_jitter_seconds)
        return min(delay, self.settings.ai_retry_max_seconds)

    def _payload(
        self,
        observation: Observation,
        criteria: CriteriaSet,
        image: PreparedImage,
    ) -> dict[str, Any]:
        structured = "\n".join(
            json.dumps(
                {
                    "id": criterion.id,
                    "category": criterion.category,
                    "criterion": criterion.text,
                },
                ensure_ascii=False,
            )
            for criterion in criteria.definitions
        )
        target_hint = (
            "Целевой автомобиль отмечен на изображении пурпурной рамкой вокруг "
            "его государственного номера и подписью TARGET."
            if observation.plate_box is not None
            else "На изображении нет графической рамки; ориентируйся на указанный ГРЗ."
        )
        prompt = f"""
Оцени только целевой автомобиль с ГРЗ {observation.plate}.
{target_hint}
Место фиксации: {observation.place}. Камера: {observation.camera}.

Критерии с обязательными стабильными ID:
{structured}

Группы критериев имеют разный смысл и влияют на разные измерения:
- lawn — только на lawn_probability: действительно ли целевой автомобиль
  находится на озеленённой территории;
- evidence — только на evidence_quality_probability: достаточно ли кадра для
  подтверждения, независимо от того, есть ли нарушение. Учитывай техническое
  качество фотографии: резкость, освещённость, пересветы и тёмные области,
  блики и отражения, погодные помехи и перекрытия важных деталей;
- identity — только на target_identity_probability: правильно ли выбран
  автомобиль и относится ли к нему рамка TARGET/ГРЗ.
- decision — совместимый формат старого criteria.txt без групп. Это общие
  правила итогового решения; для них сохрани category="decision".

Не переноси признаки с соседнего автомобиля на целевой. Отдельно оцени:
- нахождение целевого автомобиля на озеленённой территории;
- качество кадра как доказательства;
- уверенность, что рамка TARGET и ГРЗ относятся именно к целевому автомобилю;
- send_probability — вероятность, что этот фотофакт можно подтвердить и
  отправить как нарушение lawnParking для указанного целевого автомобиля.
  Это самостоятельная итоговая оценка с учётом трёх измерений и всех критериев,
  а не среднее арифметическое остальных вероятностей.

При оценке evidence хороший отдельный признак не должен компенсировать
критический дефект: если из-за блика, темноты, пересвета, размытия или
перекрытия нельзя уверенно увидеть автомобиль, колёса либо границу покрытия,
существенно снижай evidence_quality_probability и send_probability.

Считай фото «плохим фактом», если выполняется хотя бы одно условие:
- техническое качество фотографии не позволяет надёжно оценить нарушение;
- по визуальной оценке видно менее одной четверти целевого автомобиля.
В этом случае существенно снижай evidence_quality_probability и
send_probability. Не снижай lawn_probability только из-за плохого качества:
положение на газоне в таком кадре считается не отрицательным, а неустановимым.

Верни результат для каждого указанного ID. В поле id копируй только значение
поля id из списка критериев, без префикса категории (например, LEGACY001, а не
decision:LEGACY001). В поле category точно копируй соответствующее значение
category. Все вероятности — числа 0..100. Evidence и comment должны быть
краткими. Не выводи ход рассуждений.
Ответь только JSON без Markdown по схеме:
{{
  "schema_version": 2,
  "send_probability": 0,
  "lawn_probability": 0,
  "evidence_quality_probability": 0,
  "target_identity_probability": 0,
  "criteria": [
    {{
      "id": "ID критерия",
      "category": "категория критерия",
      "probability": 0,
      "satisfied": false,
      "evidence": "краткое наблюдение"
    }}
  ],
  "comment": "краткое обоснование"
}}
""".strip()
        return {
            "model": self.settings.ai_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты эксперт по визуальной проверке парковки. "
                        "Строго соблюдай формат ответа и оценивай только "
                        "указанную машину."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image.data_url},
                        },
                    ],
                },
            ],
            "stream": False,
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
        }

    @property
    def request_parameters(self) -> dict[str, Any]:
        """Reproducible request settings with no credentials or image data."""
        return {
            "model": self.settings.ai_model,
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "length_retry_max_tokens": (
                self.settings.ai_length_retry_max_tokens
            ),
            "stream": False,
        }


class _RetryableAIError(RuntimeError):
    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        global_cooldown: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.global_cooldown = global_cooldown


class _FatalAIError(RuntimeError):
    pass
