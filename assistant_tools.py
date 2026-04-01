from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests
from openai import OpenAI

logger = logging.getLogger("telegram_pinecone_bot")


def get_random_dog_fact() -> str:
    started_at = time.perf_counter()
    logger.info("dogFactTool: старт HTTP-запроса (поиск в интернете) к dogapi.dog")
    response = requests.get("https://dogapi.dog/api/v2/facts", timeout=20)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    logger.info(
        "dogFactTool: HTTP-ответ получен, status=%s, записей=%s, elapsed_ms=%.1f",
        response.status_code,
        len(data),
        (time.perf_counter() - started_at) * 1000,
    )
    if not data:
        return "Не удалось получить факт о собаках прямо сейчас."
    fact = data[0].get("attributes", {}).get("body", "")
    logger.info("dogFactTool: факт сформирован, chars=%s", len(fact or ""))
    return fact or "Не удалось получить факт о собаках прямо сейчас."


def get_random_dog_image_with_description(
    openai_client: OpenAI,
    vision_model: str = "gpt-4o-mini",
) -> Dict[str, str]:
    image_started_at = time.perf_counter()
    logger.info("dogImageTool: старт HTTP-запроса (поиск в интернете) к dog.ceo")
    image_resp = requests.get("https://dog.ceo/api/breeds/image/random", timeout=20)
    image_resp.raise_for_status()
    image_payload = image_resp.json()
    image_url = image_payload.get("message", "")
    logger.info(
        "dogImageTool: URL изображения получен, status=%s, elapsed_ms=%.1f",
        image_resp.status_code,
        (time.perf_counter() - image_started_at) * 1000,
    )
    if not image_url:
        raise RuntimeError("Dog API returned an empty image URL.")

    vision_prompt = (
        "Ты кинолог-ассистент. Посмотри на фото собаки и дай ответ строго в 3 коротких абзацах:\n"
        "1) Предполагаемая порода (или ближайшие варианты)\n"
        "2) Внешние признаки, по которым ты сделал вывод\n"
        "3) Краткая история происхождения породы\n"
        "Если уверенность низкая, честно скажи это."
    )
    vision_started_at = time.perf_counter()
    logger.info("docImageAnalyzerTool: старт Vision LLM, model=%s", vision_model)
    completion = openai_client.chat.completions.create(
        model=vision_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Ты полезный эксперт по породам собак."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    )
    description = completion.choices[0].message.content or "Не удалось получить описание породы."
    logger.info(
        "docImageAnalyzerTool: ответ Vision LLM получен, chars=%s, elapsed_ms=%.1f",
        len(description),
        (time.perf_counter() - vision_started_at) * 1000,
    )
    result = {"image_url": image_url, "description": description}
    logger.info(
        "dogImageTool+docImageAnalyzerTool: единый объект результата сформирован, has_image=%s, description_chars=%s",
        bool(result["image_url"]),
        len(result["description"]),
    )
    return result
