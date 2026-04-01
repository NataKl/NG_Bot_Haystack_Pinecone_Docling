# Hay V2 Telegram Bot

Модульная версия Telegram-бота на Haystack + Pinecone с поддержкой загрузки документов и ingestion через Docling.

## Возможности

- Все функции базовой версии:
  - команды `/start`, `/help`, `/memory`, `/dogfact`, `/dogphoto`
  - генерация ответов через Haystack pipeline
  - долговременная память в Pinecone
- Обработка файлов из Telegram (`document`):
  - поддержка `PDF`, `DOCX`, `DOC`, `PPTX`, `XLSX`, `HTML`, `MD`, `TXT`, `CSV`
  - анализ через Docling
  - разбиение на чанки
  - сохранение чанков в Pinecone (namespace `docs`)
  - метаданные чанка: `filename`, `chunk_no`, `page_no` (если доступно)
- После успешной загрузки файла:
  - бот сообщает о завершении обработки
  - отправляет одно короткое резюме (одно предложение)
- Сохранение в Pinecone:
  - сообщений пользователя
  - ответов ассистента (для всех типов входящих сообщений)

## Архитектура

```text
hay_v2_bot/
  components/
    docling_processor.py
    logging_config.py
    memory_service.py
    summarizer.py
    tools.py
  pipelines/
    generation_pipeline.py
    ingestion_pipeline.py
  bot/
    app.py
  main.py
```

## Потоки данных

### 1) Текстовые сообщения

1. Поиск памяти пользователя в Pinecone.
2. Поиск релевантных чанков документов в namespace `docs`.
3. Запуск generation pipeline (Haystack).
4. Отправка ответа в Telegram.
5. Сохранение:
   - сообщения пользователя в память;
   - ответа ассистента в память.

### 2) Файлы

1. Бот принимает файл и отправляет статус "файл получен".
2. Скачивает файл во временный путь.
3. Запускает ingestion pipeline:
   - `DoclingChunkComponent` извлекает чанки через Docling;
   - `PineconeUpsertComponent` сохраняет чанки в Pinecone (`docs`).
4. Отправляет "Готово..." и одно предложение-резюме.
5. Все отправленные ботом тексты сохраняются как ответы ассистента.

## Установка

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

> Убедись, что в `requirement.txt` присутствует `docling`.

## Переменные окружения

```env
TELEGRAM_BOT_TOKEN=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
OPENAI_API_KEY=...

# optional
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
MEMORY_TOP_K=5
MEMORY_SIMILARITY_THRESHOLD=0.85
MEMORY_HIGH_SIMILARITY_ACTION=update
PINECONE_DOCS_NAMESPACE=docs
```

## Запуск

```powershell
python hay_v2_bot/main.py
```

## Важно

- Для корректной работы обработчика файлов нужны актуальные версии `docling` и зависимостей.
- Если Docling не установлен, ingestion файлов будет недоступен и бот вернет ошибку инициализации при запуске.
