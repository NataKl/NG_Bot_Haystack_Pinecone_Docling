import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import telebot
from telebot.apihelper import ApiTelegramException
from haystack import Pipeline
from openai import OpenAI

try:
    from hay_v2_bot.components.docling_processor import DoclingProcessor
    from hay_v2_bot.components.memory_service import MemoryService, build_memory_context, extract_matches
    from hay_v2_bot.components.summarizer import FileSummarizer
    from hay_v2_bot.components.tools import ToolsService
    from hay_v2_bot.pipelines.generation_pipeline import run_generation_pipeline
except ModuleNotFoundError:
    from components.docling_processor import DoclingProcessor
    from components.memory_service import MemoryService, build_memory_context, extract_matches
    from components.summarizer import FileSummarizer
    from components.tools import ToolsService
    from pipelines.generation_pipeline import run_generation_pipeline


class TelegramBotV2:
    def __init__(
        self,
        bot: telebot.TeleBot,
        logger: logging.Logger,
        memory_service: MemoryService,
        generation_pipeline: Pipeline,
        ingestion_pipeline: Pipeline,
        tools_service: ToolsService,
        openai_client: OpenAI,
        openai_chat_model: str,
        docling_processor: DoclingProcessor,
        memory_top_k: int,
    ) -> None:
        self.bot = bot
        self.logger = logger
        self.memory_service = memory_service
        self.generation_pipeline = generation_pipeline
        self.ingestion_pipeline = ingestion_pipeline
        self.tools_service = tools_service
        self.openai_client = openai_client
        self.openai_chat_model = openai_chat_model
        self.docling_processor = docling_processor
        self.memory_top_k = memory_top_k
        self.summarizer = FileSummarizer(openai_client=openai_client, model=openai_chat_model)
        self._register_handlers()

    def _safe_send(self, chat_id: int, text: str) -> None:
        if len(text) <= 4000:
            self.bot.send_message(chat_id, text)
            return
        for i in range(0, len(text), 4000):
            self.bot.send_message(chat_id, text[i : i + 4000])

    def _save_assistant_reply(self, user_id: int, username: Optional[str], text: str) -> None:
        if not text.strip():
            return
        try:
            self.memory_service.save_assistant_message(user_id=user_id, username=username, text=text)
        except Exception:
            self.logger.exception("Не удалось сохранить ответ ассистента в Pinecone")

    def _register_handlers(self) -> None:
        @self.bot.message_handler(commands=["start", "help"])
        def handle_start(message: telebot.types.Message) -> None:
            user_id = message.from_user.id if message.from_user else message.chat.id
            username = message.from_user.username if message.from_user else None
            help_text = (
                "Привет! Я умный персональный помощник с долговременной памятью.\n\n"
                "Команды:\n"
                "/memory — показать, что я помню\n"
                "/dogfact — случайный факт о собаках\n"
                "/dogphoto — случайное фото собаки + определение породы\n\n"
                "Также присылай файлы (PDF, DOCX и другие), я их проанализирую и добавлю в знания."
            )
            self.bot.send_message(message.chat.id, help_text)
            self._save_assistant_reply(user_id=user_id, username=username, text=help_text)

        @self.bot.message_handler(commands=["memory"])
        def handle_memory(message: telebot.types.Message) -> None:
            try:
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                query_result = self.memory_service.memory.query_by_text(
                    text="Ключевые факты о пользователе",
                    top_k=self.memory_top_k,
                    filter={"user_id": {"$eq": str(user_id)}},
                    include_metadata=True,
                    include_values=False,
                )
                matches = extract_matches(query_result)
                if not matches:
                    text = "Память пока пустая."
                    self.bot.send_message(message.chat.id, text)
                    self._save_assistant_reply(user_id=user_id, username=username, text=text)
                    return
                memory_text = build_memory_context(matches)
                answer = f"Что я помню:\n\n{memory_text}"
                self._safe_send(message.chat.id, answer)
                self._save_assistant_reply(user_id=user_id, username=username, text=answer)
            except Exception:
                self.logger.exception("Ошибка при обработке /memory")
                error_text = "Ошибка чтения памяти. Попробуйте еще раз."
                self.bot.send_message(message.chat.id, error_text)
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                self._save_assistant_reply(user_id=user_id, username=username, text=error_text)

        @self.bot.message_handler(commands=["dogfact"])
        def handle_dog_fact(message: telebot.types.Message) -> None:
            try:
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                fact = self.tools_service.get_dog_fact()
                answer = f"Случайный факт о собаках:\n\n{fact}"
                self._safe_send(message.chat.id, answer)
                self._save_assistant_reply(user_id=user_id, username=username, text=answer)
            except Exception:
                self.logger.exception("Ошибка в /dogfact")
                error_text = "Не удалось получить факт о собаках. Попробуйте позже."
                self.bot.send_message(message.chat.id, error_text)
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                self._save_assistant_reply(user_id=user_id, username=username, text=error_text)

        @self.bot.message_handler(commands=["dogphoto"])
        def handle_dog_photo(message: telebot.types.Message) -> None:
            try:
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                result = self.tools_service.get_dog_photo()
                caption = result["description"][:1024]
                self.bot.send_photo(message.chat.id, result["image_url"], caption=caption)
                self._save_assistant_reply(user_id=user_id, username=username, text=result["description"])
            except Exception:
                self.logger.exception("Ошибка в /dogphoto")
                error_text = "Не удалось получить фото собаки. Попробуйте позже."
                self.bot.send_message(message.chat.id, error_text)
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                self._save_assistant_reply(user_id=user_id, username=username, text=error_text)

        @self.bot.message_handler(content_types=["document"])
        def handle_document(message: telebot.types.Message) -> None:
            document = message.document
            if not document:
                return
            user_id = message.from_user.id if message.from_user else message.chat.id
            username = message.from_user.username if message.from_user else None
            file_name = document.file_name or "document.bin"
            if not self.docling_processor.is_supported(file_name):
                answer = "Формат файла пока не поддерживается. Поддерживаются: PDF, DOCX, DOC, PPTX, XLSX, HTML, MD, TXT, CSV."
                self.bot.send_message(message.chat.id, answer)
                self._save_assistant_reply(user_id=user_id, username=username, text=answer)
                return

            received_text = "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
            self.bot.send_message(message.chat.id, received_text)
            self._save_assistant_reply(user_id=user_id, username=username, text=received_text)

            temp_path: Optional[str] = None
            try:
                file_info = self.bot.get_file(document.file_id)
                downloaded = self.bot.download_file(file_info.file_path)
                suffix = os.path.splitext(file_name)[1] or ".bin"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(downloaded)
                    temp_path = tmp_file.name

                ingestion_result = self.ingestion_pipeline.run(
                    {"docling_chunker": {"file_path": temp_path, "file_name": file_name}}
                )
                prepared_documents: List[Dict[str, Any]] = ingestion_result.get("pinecone_writer", {}).get(
                    "prepared_documents", []
                )
                if not prepared_documents:
                    empty_text = "Не удалось извлечь текст из файла."
                    self.bot.send_message(message.chat.id, empty_text)
                    self._save_assistant_reply(user_id=user_id, username=username, text=empty_text)
                    return

                done_text = "Готово. Я изучил этот файл, теперь можем его обсудить."
                self.bot.send_message(message.chat.id, done_text)
                self._save_assistant_reply(user_id=user_id, username=username, text=done_text)
                joined_text = "\n".join(item["text"] for item in prepared_documents[:8])
                summary = self.summarizer.summarize_one_sentence(joined_text)
                self._safe_send(message.chat.id, summary)
                self._save_assistant_reply(user_id=user_id, username=username, text=summary)
            except Exception:
                self.logger.exception("Ошибка при обработке файла")
                error_text = "Не удалось обработать файл. Попробуйте еще раз."
                self.bot.send_message(message.chat.id, error_text)
                self._save_assistant_reply(user_id=user_id, username=username, text=error_text)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        @self.bot.message_handler(content_types=["text"])
        def handle_text(message: telebot.types.Message) -> None:
            try:
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                user_text = (message.text or "").strip()
                if not user_text:
                    return

                memory_matches = self.memory_service.load_user_memories(user_id=user_id)
                memory_context = build_memory_context(memory_matches)

                doc_matches = self.memory_service.search_docs(user_text, top_k=self.memory_top_k)
                doc_lines: List[str] = []
                for item in doc_matches:
                    metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
                    text = metadata.get("text")
                    filename = metadata.get("filename", "unknown")
                    page_no = metadata.get("page_no")
                    if text:
                        prefix = f"[{filename}"
                        if page_no is not None:
                            prefix += f", page={page_no}"
                        prefix += "] "
                        doc_lines.append(prefix + text)

                tool_context = "Инструменты не вызывались."
                lowered = user_text.lower()
                if any(token in lowered for token in ["факт о собак", "dog fact", "dogfact"]):
                    tool_context = f"Случайный факт о собаках: {self.tools_service.get_dog_fact()}"
                elif any(token in lowered for token in ["фото соб", "dog photo", "dog image", "dogphoto"]):
                    photo = self.tools_service.get_dog_photo()
                    caption = photo["description"][:1024]
                    self.bot.send_photo(message.chat.id, photo["image_url"], caption=caption)
                    assistant_text = photo["description"]
                    self.memory_service.save_user_message(user_id=user_id, username=username, text=user_text)
                    self._save_assistant_reply(user_id=user_id, username=username, text=assistant_text)
                    return

                assistant_text = run_generation_pipeline(
                    pipeline=self.generation_pipeline,
                    question=user_text,
                    memory_lines=memory_context.splitlines() if memory_context else [],
                    doc_lines=doc_lines,
                    tool_context=tool_context,
                )
                self._safe_send(message.chat.id, assistant_text)
                self.memory_service.save_user_message(user_id=user_id, username=username, text=user_text)
                self._save_assistant_reply(user_id=user_id, username=username, text=assistant_text)
            except Exception:
                self.logger.exception("Ошибка в обработчике текстового сообщения")
                error_text = "Произошла ошибка при обработке сообщения. Попробуйте еще раз."
                self.bot.send_message(message.chat.id, error_text)
                user_id = message.from_user.id if message.from_user else message.chat.id
                username = message.from_user.username if message.from_user else None
                self._save_assistant_reply(user_id=user_id, username=username, text=error_text)

    def run(self) -> None:
        self.logger.info("Telegram Pinecone assistant bot v2 is running...")
        try:
            self.bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)
        except ApiTelegramException as exc:
            error_message = str(exc)
            if "Error code: 409" in error_message or "terminated by other getUpdates request" in error_message:
                self.logger.error(
                    "Telegram вернул 409 Conflict: уже запущен другой экземпляр бота. "
                    "Остановите второй процесс и перезапустите текущий."
                )
                return
            raise
