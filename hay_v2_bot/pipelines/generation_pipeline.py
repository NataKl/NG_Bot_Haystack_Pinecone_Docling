from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage


def build_generation_pipeline(chat_model: str) -> Pipeline:
    template = [
        ChatMessage.from_system(
            "Ты умный персональный помощник в Telegram. "
            "Учитывай память пользователя, контекст документов и инструменты. "
            "Если данных недостаточно, честно скажи об этом и задай уточняющий вопрос."
        ),
        ChatMessage.from_user(
            """
Контекст долговременной памяти:
{% for document in memory_documents %}
- {{ document.content }}
{% endfor %}

Контекст документов:
{% for document in retrieved_documents %}
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
    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=template,
            required_variables={"memory_documents", "retrieved_documents", "tool_context", "question"},
        ),
    )
    pipeline.add_component("llm", OpenAIChatGenerator(model=chat_model))
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def run_generation_pipeline(
    pipeline: Pipeline,
    question: str,
    memory_lines: List[str],
    doc_lines: List[str],
    tool_context: str,
) -> str:
    memory_documents = [Document(content=line) for line in (memory_lines or ["Память пока пустая."])]
    retrieved_documents = [Document(content=line) for line in (doc_lines or ["Релевантные документы не найдены."])]
    result = pipeline.run(
        {
            "prompt_builder": {
                "memory_documents": memory_documents,
                "retrieved_documents": retrieved_documents,
                "question": question,
                "tool_context": tool_context or "Инструменты не вызывались.",
            }
        }
    )
    replies = result.get("llm", {}).get("replies", [])
    if not replies:
        return "Не получилось сформировать ответ."
    return replies[0].text
