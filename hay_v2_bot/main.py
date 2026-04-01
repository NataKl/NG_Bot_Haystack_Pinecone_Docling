import os
import sys
from pathlib import Path

import telebot
from dotenv import load_dotenv
from openai import OpenAI

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from hay_v2_bot.bot.app import TelegramBotV2
    from hay_v2_bot.components.docling_processor import DoclingProcessor
    from hay_v2_bot.components.logging_config import setup_logging
    from hay_v2_bot.components.memory_service import MemoryService
    from hay_v2_bot.components.tools import ToolsService
    from hay_v2_bot.pipelines.generation_pipeline import build_generation_pipeline
    from hay_v2_bot.pipelines.ingestion_pipeline import build_ingestion_pipeline
except ModuleNotFoundError:
    from bot.app import TelegramBotV2
    from components.docling_processor import DoclingProcessor
    from components.logging_config import setup_logging
    from components.memory_service import MemoryService
    from components.tools import ToolsService
    from pipelines.generation_pipeline import build_generation_pipeline
    from pipelines.ingestion_pipeline import build_ingestion_pipeline
from pinecone_manager import COSINE_SIMILARITY_THRESHOLD


def _normalize_high_similarity_action(raw_value: str) -> str:
    normalized = (raw_value or "").strip().lower()
    if normalized in {"skip", "update"}:
        return normalized
    if "skip" in normalized:
        return "skip"
    if "update" in normalized:
        return "update"
    return "update"


def main() -> None:
    load_dotenv()
    logger = setup_logging()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")  # must route all OpenAI calls via this base URL
    openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    memory_top_k = int(os.getenv("MEMORY_TOP_K", "5"))
    raw_high_similarity_action = os.getenv("MEMORY_HIGH_SIMILARITY_ACTION", "update")
    similarity_threshold = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", str(COSINE_SIMILARITY_THRESHOLD)))
    high_similarity_action = _normalize_high_similarity_action(raw_high_similarity_action)

    if (raw_high_similarity_action or "").strip().lower() not in {"skip", "update"}:
        logger.warning(
            "Некорректное MEMORY_HIGH_SIMILARITY_ACTION=%r, использую %r.",
            raw_high_similarity_action,
            high_similarity_action,
        )

    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required.")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME is required.")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    bot = telebot.TeleBot(telegram_token)
    # Ensure all OpenAI SDK and Haystack OpenAI components use the same base URL.
    # Some libraries read OPENAI_BASE_URL from env, so export it too.
    if openai_base_url:
        os.environ["OPENAI_BASE_URL"] = openai_base_url
        # Some tools expect OPENAI_API_BASE; keep them in sync.
        os.environ["OPENAI_API_BASE"] = openai_base_url
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url or None)
    tools_service = ToolsService(openai_client=openai_client, vision_model=openai_vision_model)
    memory_service = MemoryService(
        index_name=pinecone_index_name,
        openai_api_key=openai_api_key,
        memory_top_k=memory_top_k,
        similarity_threshold=similarity_threshold,
        high_similarity_action=high_similarity_action,
        docs_namespace=os.getenv("PINECONE_DOCS_NAMESPACE", "docs"),
    )
    generation_pipeline = build_generation_pipeline(chat_model=openai_chat_model)
    docling_processor = DoclingProcessor()
    ingestion_pipeline = build_ingestion_pipeline(processor=docling_processor, memory_service=memory_service)

    # Проверка доступа к индексу на старте.
    stats = memory_service.memory.index.describe_index_stats()
    logger.info("Подключение к Pinecone успешно. Stats: %s", stats)

    app = TelegramBotV2(
        bot=bot,
        logger=logger,
        memory_service=memory_service,
        generation_pipeline=generation_pipeline,
        ingestion_pipeline=ingestion_pipeline,
        tools_service=tools_service,
        openai_client=openai_client,
        openai_chat_model=openai_chat_model,
        docling_processor=docling_processor,
        memory_top_k=memory_top_k,
    )
    app.run()


if __name__ == "__main__":
    main()
