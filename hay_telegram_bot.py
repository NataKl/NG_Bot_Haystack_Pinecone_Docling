import os
import time
import uuid
import logging
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
import telebot
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from assistant_tools import get_random_dog_fact, get_random_dog_image_with_description
from pinecone_manager import COSINE_SIMILARITY_THRESHOLD, PineconeManager


logger = logging.getLogger("telegram_pinecone_bot")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    json_formatter = JsonFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("bot_logs.json", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)


def _build_memory_context(memories: List[Any]) -> str:
    context_lines: List[str] = []
    for item in memories:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
        score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
        text = metadata.get("text")
        if not text:
            continue
        prefix = f"[score={score:.4f}] " if isinstance(score, (float, int)) else ""
        context_lines.append(f"{prefix}{text}")
    return "\n".join(context_lines[:10])


def _extract_matches(query_result: Any) -> List[Any]:
    if isinstance(query_result, dict):
        return query_result.get("matches", []) or []
    return getattr(query_result, "matches", []) or []


def _safe_send(bot: telebot.TeleBot, chat_id: int, text: str) -> None:
    logger.info("SEND: chat_id=%s, chars=%s", chat_id, len(text))
    if len(text) <= 4000:
        bot.send_message(chat_id, text)
        return
    for i in range(0, len(text), 4000):
        bot.send_message(chat_id, text[i : i + 4000])


def _build_haystack_chat_pipeline(chat_model: str) -> Pipeline:
    template = [
        ChatMessage.from_system(
            "Ты умный персональный помощник в Telegram. "
            "Обязательно учитывай контекст памяти пользователя и поддерживай связный диалог. "
            "Если данных не хватает, задавай уточняющий вопрос."
        ),
        ChatMessage.from_user(
            """
Контекст долговременной памяти:
{% for document in documents %}
- {{ document.content }}
{% endfor %}

Контекст инструментов:
{{ tool_context }}

Новое сообщение пользователя:
{{ question }}

Ответь дружелюбно и по делу.
"""
        ),
    ]
    prompt_builder = ChatPromptBuilder(template=template)
    llm = OpenAIChatGenerator(model=chat_model)
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def main() -> None:
    load_dotenv()
    setup_logging()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    memory_top_k = int(os.getenv("MEMORY_TOP_K", "5"))
    raw_high_similarity_action = os.getenv("MEMORY_HIGH_SIMILARITY_ACTION", "update")
    similarity_threshold = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", str(COSINE_SIMILARITY_THRESHOLD)))
    high_similarity_action = (raw_high_similarity_action or "update").strip().lower()
    if high_similarity_action not in {"skip", "update"}:
        logger.warning(
            "Некорректное MEMORY_HIGH_SIMILARITY_ACTION=%r, использую 'update'.",
            raw_high_similarity_action,
        )
        high_similarity_action = "update"

    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required.")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME is required.")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    bot = telebot.TeleBot(telegram_token)
    llm_client = OpenAI(api_key=openai_api_key)
    rag_chat_pipeline = _build_haystack_chat_pipeline(chat_model=openai_chat_model)
    memory = PineconeManager(index_name=pinecone_index_name, openai_api_key=openai_api_key)
    logger.info(
        "Конфигурация: index=%s, model=%s, top_k=%s, threshold=%.3f, high_similarity_action=%s",
        pinecone_index_name,
        openai_chat_model,
        memory_top_k,
        similarity_threshold,
        high_similarity_action,
    )

    # Проверка подключения к Pinecone при старте бота.
    try:
        stats = memory.index.describe_index_stats()
        logger.info("Подключение к Pinecone успешно. Stats: %s", stats)
    except Exception:
        logger.exception("Ошибка подключения к Pinecone. Проверьте ключ, index name и сеть.")
        raise

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: telebot.types.Message) -> None:
        help_text = (
            "Привет! Я умный персональный помощник с долговременной памятью.\n\n"
            "Как это работает:\n"
            "- Я ищу в памяти похожие факты о вас перед ответом.\n"
            "- Сохраняю в Pinecone только ваши сообщения (cosine similarity).\n"
            "- Если информация дублируется (высокое сходство), обновляю или пропускаю.\n\n"
            "Команды:\n"
            "/memory — показать, что я помню\n"
            "/dogfact — случайный факт о собаках\n"
            "/dogphoto — случайное фото собаки + определение породы и краткая история\n\n"
            "Просто отправьте сообщение."
        )
        bot.send_message(message.chat.id, help_text)

    @bot.message_handler(commands=["memory"])
    def handle_memory(message: telebot.types.Message) -> None:
        try:
            user_id = message.from_user.id if message.from_user else message.chat.id
            logger.info("Команда /memory от user_id=%s", user_id)
            query_result = memory.query_by_text(
                text="Ключевые факты о пользователе",
                top_k=memory_top_k,
                filter={"user_id": {"$eq": str(user_id)}},
                include_metadata=True,
                include_values=False,
            )
            matches = _extract_matches(query_result)
            logger.info("/memory: найдено %s записей для user_id=%s", len(matches), user_id)
            if not matches:
                bot.send_message(message.chat.id, "Память пока пустая.")
                return
            memory_text = _build_memory_context(matches)
            _safe_send(bot, message.chat.id, f"Что я помню:\n\n{memory_text}")
        except Exception:
            logger.exception("Ошибка при обработке /memory")
            bot.send_message(message.chat.id, "Ошибка чтения памяти. Попробуйте еще раз.")

    @bot.message_handler(commands=["dogfact"])
    def handle_dog_fact(message: telebot.types.Message) -> None:
        try:
            fact = get_random_dog_fact()
            _safe_send(bot, message.chat.id, f"Случайный факт о собаках:\n\n{fact}")
        except Exception:
            logger.exception("Ошибка в /dogfact")
            bot.send_message(message.chat.id, "Не удалось получить факт о собаках. Попробуйте позже.")

    @bot.message_handler(commands=["dogphoto"])
    def handle_dog_photo(message: telebot.types.Message) -> None:
        try:
            result = get_random_dog_image_with_description(
                openai_client=llm_client,
                vision_model=openai_vision_model,
            )
            image_url = result["image_url"]
            description = result["description"]
            caption = description[:1024]
            logger.info("SEND: /dogphoto image+caption, caption_chars=%s", len(caption))
            bot.send_photo(message.chat.id, image_url, caption=caption)
        except Exception:
            logger.exception("Ошибка в /dogphoto")
            bot.send_message(message.chat.id, "Не удалось получить фото собаки. Попробуйте позже.")

    def _maybe_run_tools(user_text: str) -> Dict[str, Optional[str]]:
        lowered = user_text.lower()
        tool_parts: List[str] = []
        photo_url: Optional[str] = None
        photo_caption: Optional[str] = None
        fact_tokens = [
            "факт о собак",
            "факта о собак",
            "факт про собак",
            "факты о собак",
            "интересных факта о собак",
            "dog fact",
            "dogfact",
        ]
        fact_triggered = [token for token in fact_tokens if token in lowered]
        if fact_triggered:
            logger.info("ROUTER: включен tool=dog_fact, matched_tokens=%s", fact_triggered)
            try:
                tool_parts.append(f"Случайный факт о собаках: {get_random_dog_fact()}")
            except Exception:
                logger.exception("Ошибка вызова инструмента факта о собаках")
        photo_tokens = ["фото соб", "картинк соб", "dog photo", "dog image", "dogphoto"]
        photo_triggered = [token for token in photo_tokens if token in lowered]
        has_photo_intent = any(token in lowered for token in ["фото", "картин", "image", "photo"])
        has_dog_intent = any(token in lowered for token in ["собак", "собач", "dog"])
        should_run_photo_tool = bool(photo_triggered) or (has_photo_intent and has_dog_intent)
        if should_run_photo_tool:
            logger.info(
                "ROUTER: включен tool=dog_photo, matched_tokens=%s, inferred_by_intent=%s",
                photo_triggered,
                has_photo_intent and has_dog_intent,
            )
            try:
                image_result = get_random_dog_image_with_description(
                    openai_client=llm_client,
                    vision_model=openai_vision_model,
                )
                photo_url = image_result["image_url"]
                photo_caption = image_result["description"]
                tool_parts.append(
                    "Найдено изображение собаки и анализ породы:\n"
                    f"URL: {image_result['image_url']}\n"
                    f"Описание: {image_result['description']}"
                )
            except Exception:
                logger.exception("Ошибка вызова инструмента фото собаки")
        if not tool_parts:
            logger.info("ROUTER: tools_not_triggered, источник ответа=LLM")
        else:
            logger.info("ROUTER: tools_triggered_count=%s", len(tool_parts))
        return {
            "tool_context": "\n\n".join(tool_parts),
            "photo_url": photo_url,
            "photo_caption": photo_caption,
        }

    @bot.message_handler(content_types=["text"])
    def handle_text(message: telebot.types.Message) -> None:
        try:
            user_id = message.from_user.id if message.from_user else message.chat.id
            username = message.from_user.username if message.from_user else None
            user_text = (message.text or "").strip()
            if not user_text:
                return
            logger.info("Новое сообщение от user_id=%s, text=%r", user_id, user_text[:200])

            # 1) Сначала читаем релевантную память пользователя
            memory_search_started_at = time.perf_counter()
            query_result = memory.query_by_text(
                text=user_text,
                top_k=memory_top_k,
                filter={"user_id": {"$eq": str(user_id)}},
                include_metadata=True,
                include_values=False,
            )
            matches = _extract_matches(query_result)
            memory_context = _build_memory_context(matches)
            logger.info(
                "Поиск памяти user_id=%s: найдено %s фрагментов, elapsed_ms=%.1f",
                user_id,
                len(matches),
                (time.perf_counter() - memory_search_started_at) * 1000,
            )

            # 2) Запускаем инструменты при релевантном запросе пользователя.
            tool_result = _maybe_run_tools(user_text)
            tool_context = tool_result.get("tool_context") or "Инструменты не вызывались."
            photo_url = tool_result.get("photo_url")
            photo_caption = tool_result.get("photo_caption")
            if photo_url:
                caption = (photo_caption or "Фото собаки из API")[:1024]
                logger.info(
                    "SEND: отправка фото пользователю user_id=%s, url=%s, caption_chars=%s",
                    user_id,
                    photo_url,
                    len(caption),
                )
                bot.send_photo(message.chat.id, photo_url, caption=caption)
                logger.info(
                    "SEND: photo+caption отправлены, пропускаем отдельное текстовое сообщение пользователю user_id=%s",
                    user_id,
                )
                assistant_text = photo_caption or "Отправлено фото собаки."
            else:
                # 3) Генерация ответа через Haystack pipeline с учетом memory + tool context.
                llm_started_at = time.perf_counter()
                docs = [Document(content=line) for line in (memory_context.splitlines() or ["Память пока пустая."])]
                logger.info(
                    "LLM: старт генерации, docs=%s, tool_context_chars=%s",
                    len(docs),
                    len(tool_context),
                )
                response = rag_chat_pipeline.run(
                    {
                        "prompt_builder": {
                            "documents": docs,
                            "question": user_text,
                            "tool_context": tool_context,
                        }
                    }
                )
                replies = response.get("llm", {}).get("replies", [])
                assistant_text = replies[0].text if replies else "Не получилось сформировать ответ."
                logger.info(
                    "LLM: ответ получен, replies=%s, chars=%s, elapsed_ms=%.1f",
                    len(replies),
                    len(assistant_text),
                    (time.perf_counter() - llm_started_at) * 1000,
                )
                _safe_send(bot, message.chat.id, assistant_text)
                logger.info("Ответ пользователю user_id=%s отправлен, preview=%r", user_id, assistant_text[:200])

            # 4) Сохраняем пользовательское сообщение в память с проверкой сходства
            doc_id = f"user-{user_id}-{uuid.uuid4().hex}"
            user_metadata: Dict[str, Any] = {
                "user_id": str(user_id),
                "username": username or "",
                "role": "user",
                "source": "telegram",
                "timestamp": int(time.time()),
            }
            user_memory_result = memory.save_memory_with_similarity_check(
                doc_id=doc_id,
                text=user_text,
                metadata=user_metadata,
                similarity_threshold=similarity_threshold,
                on_high_similarity=high_similarity_action,
                similarity_filter={
                    "user_id": {"$eq": str(user_id)},
                    "role": {"$eq": "user"},
                },
            )
            logger.info("Сохранение user memory: %s", user_memory_result)

        except Exception:
            logger.exception("Ошибка в обработчике текстового сообщения")
            bot.send_message(message.chat.id, "Произошла ошибка при обработке сообщения. Попробуйте еще раз.")

    logger.info("Telegram Pinecone assistant bot is running...")
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)


if __name__ == "__main__":
    main()
