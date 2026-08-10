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


def parse_assessment(content: str) -> Assessment:
    value = _extract_json_object(content)
    probability = value.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise AIError("AI response probability must be a number")
    if not 0 <= float(probability) <= 100:
        raise AIError("AI response probability must be between 0 and 100")

    details = value.get("criteria", [])
    if not isinstance(details, list) or not all(
        isinstance(item, dict) for item in details
    ):
        raise AIError("AI response criteria must be an array of objects")
    comment = value.get("comment", "")
    if not isinstance(comment, str):
        comment = str(comment)
    return Assessment(
        probability=round(float(probability)),
        criteria_details=details,
        comment=comment,
        raw_response=content,
    )


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
    raise AIError(
        "AI response message content is not usable "
        f"(content_type={type(content).__name__}, "
        f"finish_reason={choice.get('finish_reason')!r}, "
        f"refusal={bool(message.get('refusal'))}, "
        f"tool_calls={tool_call_count}, "
        f"reasoning={isinstance(message.get('reasoning'), str)})"
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
                    "AppleWebKit/537.36 ParkingScore/0.1"
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
                return parse_assessment(content)
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
                delay = self._retry_delay(attempt, exc)
                if isinstance(exc, _RetryableAIError) and exc.global_cooldown:
                    self._extend_global_cooldown(delay)
                logger.warning(
                    "AI request attempt %d/%d failed; retrying in %.1fs: %s",
                    attempt,
                    self.settings.ai_request_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise AITransientError(f"AI request failed after retries: {last_error}")

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
        numbered = "\n".join(
            f"{index}. {criterion}"
            for index, criterion in enumerate(criteria.items, start=1)
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

Критерии:
{numbered}

Верни вероятность от 0 до 100 того, что целевой автомобиль одновременно
соответствует всему списку критериев. Не подменяй целевой автомобиль соседним.
Если важные детали не видны, снижай вероятность.

Ответь только JSON без Markdown по схеме:
{{
  "probability": 0,
  "criteria": [
    {{"criterion": "текст критерия", "probability": 0, "satisfied": false}}
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
                        "Строго соблюдай формат ответа и оценивай только указанную машину."
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
