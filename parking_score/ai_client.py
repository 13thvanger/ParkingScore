from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .config import Settings
from .criteria import CriteriaSet
from .image_processor import PreparedImage
from .models import Assessment, Observation

logger = logging.getLogger(__name__)


class AIError(RuntimeError):
    """Raised when the AI service cannot produce a valid assessment."""


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


class AIClient:
    def __init__(
        self, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.settings = settings
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
                    raise _RetryableAIError(message_text)
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise AIError("AI response message content is not text")
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
                delay = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "AI request attempt %d/%d failed; retrying in %ds: %s",
                    attempt,
                    self.settings.ai_request_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise AIError(f"AI request failed after retries: {last_error}")

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
    pass


class _FatalAIError(RuntimeError):
    pass
