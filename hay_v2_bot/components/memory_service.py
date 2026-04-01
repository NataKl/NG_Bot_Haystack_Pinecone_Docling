import time
import uuid
from typing import Any, Dict, List, Optional

from pinecone_manager import COSINE_SIMILARITY_THRESHOLD, PineconeManager


def extract_matches(query_result: Any) -> List[Any]:
    if isinstance(query_result, dict):
        return query_result.get("matches", []) or []
    return getattr(query_result, "matches", []) or []


def build_memory_context(memories: List[Any], max_items: int = 10) -> str:
    context_lines: List[str] = []
    for item in memories:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
        score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
        text = metadata.get("text")
        if not text:
            continue
        prefix = f"[score={score:.4f}] " if isinstance(score, (float, int)) else ""
        context_lines.append(f"{prefix}{text}")
    return "\n".join(context_lines[:max_items])


class MemoryService:
    def __init__(
        self,
        index_name: str,
        openai_api_key: str,
        memory_top_k: int = 5,
        similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD,
        high_similarity_action: str = "update",
        docs_namespace: str = "docs",
    ) -> None:
        self.memory_top_k = memory_top_k
        self.similarity_threshold = similarity_threshold
        self.high_similarity_action = high_similarity_action
        self.docs_namespace = docs_namespace
        self.memory = PineconeManager(index_name=index_name, openai_api_key=openai_api_key)

    def load_user_memories(self, user_id: int) -> List[Any]:
        query_result = self.memory.query_by_text(
            text="Ключевые факты о пользователе",
            top_k=self.memory_top_k,
            filter={"user_id": {"$eq": str(user_id)}, "role": {"$eq": "user"}},
            include_metadata=True,
            include_values=False,
        )
        return extract_matches(query_result)

    def search_docs(self, text: str, top_k: int = 5) -> List[Any]:
        query_result = self.memory.query_by_text(
            text=text,
            top_k=top_k,
            namespace=self.docs_namespace,
            include_metadata=True,
            include_values=False,
        )
        return extract_matches(query_result)

    def save_user_message(self, user_id: int, username: Optional[str], text: str) -> Dict[str, Any]:
        doc_id = f"user-{user_id}-{uuid.uuid4().hex}"
        metadata: Dict[str, Any] = {
            "user_id": str(user_id),
            "username": username or "",
            "role": "user",
            "source": "telegram",
            "timestamp": int(time.time()),
        }
        return self.memory.save_memory_with_similarity_check(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            similarity_threshold=self.similarity_threshold,
            on_high_similarity=self.high_similarity_action,
            similarity_filter={"user_id": {"$eq": str(user_id)}, "role": {"$eq": "user"}},
        )

    def save_assistant_message(self, user_id: int, username: Optional[str], text: str) -> Dict[str, Any]:
        doc_id = f"assistant-{user_id}-{uuid.uuid4().hex}"
        metadata: Dict[str, Any] = {
            "user_id": str(user_id),
            "username": username or "",
            "role": "assistant",
            "source": "telegram",
            "timestamp": int(time.time()),
        }
        return self.memory.save_memory_with_similarity_check(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            similarity_threshold=self.similarity_threshold,
            on_high_similarity=self.high_similarity_action,
            similarity_filter={"user_id": {"$eq": str(user_id)}, "role": {"$eq": "assistant"}},
        )

    def upsert_doc_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.memory.upsert_documents(chunks, namespace=self.docs_namespace)
