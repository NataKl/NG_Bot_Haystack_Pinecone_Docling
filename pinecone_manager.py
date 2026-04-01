import os
import time
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Глобальный порог косинусного сходства:
# score < threshold -> новая информация (сохраняем)
# score >= threshold -> дубликат/вариация (пропускаем или обновляем)
COSINE_SIMILARITY_THRESHOLD = 0.85
logger = logging.getLogger("telegram_pinecone_bot")


class PineconeManager:
    """
    Универсальный менеджер для операций чтения/записи в Pinecone.

    Поддерживает:
    - прямую запись векторов
    - запись текстовых документов с авто-эмбеддингом (OpenAI)
    - чтение по вектору
    - чтение по тексту (авто-эмбеддинг + векторный поиск)
    - получение по id и операции удаления
    """

    def __init__(
        self,
        index_name: str,
        namespace: str = "",
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        load_dotenv()

        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model

        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required.")

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(index_name)
        # Route embeddings strictly through OpenAI API endpoint defined by OPENAI_BASE_URL, if provided.
        self.openai_client = (
            OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url or None)
            if self.openai_api_key
            else None
        )

    # --------------------------
    # Вспомогательные методы эмбеддинга
    # --------------------------
    def _embed_text(self, text: str) -> List[float]:
        if not self.openai_client:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Provide openai_api_key in constructor or set OPENAI_API_KEY."
            )
        started_at = time.perf_counter()
        logger.info(
            "PINECONE embed_text: старт эмбеддинга, model=%s, chars=%s",
            self.embedding_model,
            len(text),
        )
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        logger.info(
            "PINECONE embed_text: эмбеддинг готов, dim=%s, elapsed_ms=%.1f",
            len(response.data[0].embedding),
            (time.perf_counter() - started_at) * 1000,
        )
        return response.data[0].embedding

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.openai_client:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Provide openai_api_key in constructor or set OPENAI_API_KEY."
            )
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    # --------------------------
    # Операции записи
    # --------------------------
    def upsert_vectors(
        self,
        vectors: Iterable[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Записывает/обновляет заранее вычисленные векторы в Pinecone.
        Каждый элемент vectors должен иметь вид:
        {"id": "...", "values": [...], "metadata": {...}}
        """
        target_namespace = self.namespace if namespace is None else namespace
        return self.index.upsert(vectors=list(vectors), namespace=target_namespace)

    def upsert_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Векторизует один текстовый документ и записывает его как вектор.
        """
        target_namespace = self.namespace if namespace is None else namespace
        metadata = metadata or {}
        metadata["text"] = text

        embedding = self._embed_text(text)
        payload = [{"id": doc_id, "values": embedding, "metadata": metadata}]
        return self.index.upsert(vectors=payload, namespace=target_namespace)

    def upsert_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Векторизует и записывает несколько текстовых документов.
        Каждый документ должен включать:
        {"id": "...", "text": "...", "metadata": {...optional...}}
        """
        if not documents:
            return {"upserted_count": 0}

        target_namespace = self.namespace if namespace is None else namespace

        texts = [doc["text"] for doc in documents]
        embeddings = self._embed_texts(texts)

        payload = []
        for doc, vector in zip(documents, embeddings):
            meta = dict(doc.get("metadata", {}))
            meta["text"] = doc["text"]
            payload.append(
                {
                    "id": doc["id"],
                    "values": vector,
                    "metadata": meta,
                }
            )

        return self.index.upsert(vectors=payload, namespace=target_namespace)

    # --------------------------
    # Операции чтения
    # --------------------------
    def query_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Поиск похожих записей по заранее вычисленному вектору.
        """
        target_namespace = self.namespace if namespace is None else namespace
        return self.index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=target_namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )

    def query_by_text(
        self,
        text: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Поиск по тексту: преобразует текст в эмбеддинг и выполняет запрос в Pinecone.
        """
        started_at = time.perf_counter()
        logger.info(
            "PINECONE query_by_text: старт, top_k=%s, namespace=%r, has_filter=%s",
            top_k,
            namespace if namespace is not None else self.namespace,
            filter is not None,
        )
        query_vector = self._embed_text(text)
        result = self.query_by_vector(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )
        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", []) or []
        logger.info(
            "PINECONE query_by_text: завершен, matches=%s, elapsed_ms=%.1f",
            len(matches),
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def save_memory_with_similarity_check(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD,
        on_high_similarity: str = "skip",
        similarity_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Сохраняет сообщение в долговременную память только после проверки сходства.

        Логика:
        - если max_score < similarity_threshold: это новая информация, сохраняем
        - если max_score >= similarity_threshold:
          - on_high_similarity="skip": пропускаем запись
          - on_high_similarity="update": обновляем наиболее похожий слот памяти
        """
        if on_high_similarity not in {"skip", "update"}:
            raise ValueError("on_high_similarity must be either 'skip' or 'update'.")

        target_namespace = self.namespace if namespace is None else namespace
        metadata = dict(metadata or {})
        metadata["text"] = text
        started_at = time.perf_counter()
        logger.info(
            "PINECONE save_memory: старт, doc_id=%s, threshold=%.3f, action=%s, has_filter=%s",
            doc_id,
            similarity_threshold,
            on_high_similarity,
            similarity_filter is not None,
        )

        vector = self._embed_text(text)
        # Ищем ближайший слот только в релевантной группе памяти (например, user/assistant).
        query_result = self.index.query(
            vector=vector,
            top_k=1,
            namespace=target_namespace,
            filter=similarity_filter,
            include_values=False,
            include_metadata=True,
        )

        matches = query_result.get("matches", []) if isinstance(query_result, dict) else getattr(query_result, "matches", [])
        best_match = matches[0] if matches else None

        if best_match is None:
            upsert_result = self.index.upsert(
                vectors=[{"id": doc_id, "values": vector, "metadata": metadata}],
                namespace=target_namespace,
            )
            result = {
                "action": "saved",
                "reason": "memory_empty",
                "threshold": similarity_threshold,
                "score": None,
                "target_id": doc_id,
                "upsert_result": upsert_result,
            }
            logger.info(
                "PINECONE save_memory: %s, elapsed_ms=%.1f",
                result,
                (time.perf_counter() - started_at) * 1000,
            )
            return result

        best_score = best_match.get("score") if isinstance(best_match, dict) else getattr(best_match, "score", None)
        best_id = best_match.get("id") if isinstance(best_match, dict) else getattr(best_match, "id", None)

        if best_score is None or best_score < similarity_threshold:
            upsert_result = self.index.upsert(
                vectors=[{"id": doc_id, "values": vector, "metadata": metadata}],
                namespace=target_namespace,
            )
            result = {
                "action": "saved",
                "reason": "low_similarity",
                "threshold": similarity_threshold,
                "score": best_score,
                "nearest_id": best_id,
                "target_id": doc_id,
                "upsert_result": upsert_result,
            }
            logger.info(
                "PINECONE save_memory: %s, elapsed_ms=%.1f",
                result,
                (time.perf_counter() - started_at) * 1000,
            )
            return result

        if on_high_similarity == "skip":
            result = {
                "action": "skipped",
                "reason": "high_similarity",
                "threshold": similarity_threshold,
                "score": best_score,
                "nearest_id": best_id,
            }
            logger.info(
                "PINECONE save_memory: %s, elapsed_ms=%.1f",
                result,
                (time.perf_counter() - started_at) * 1000,
            )
            return result

        update_id = best_id or doc_id
        upsert_result = self.index.upsert(
            vectors=[{"id": update_id, "values": vector, "metadata": metadata}],
            namespace=target_namespace,
        )
        result = {
            "action": "updated",
            "reason": "high_similarity",
            "threshold": similarity_threshold,
            "score": best_score,
            "target_id": update_id,
            "upsert_result": upsert_result,
        }
        logger.info(
            "PINECONE save_memory: %s, elapsed_ms=%.1f",
            result,
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def fetch_by_ids(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_namespace = self.namespace if namespace is None else namespace
        return self.index.fetch(ids=list(ids), namespace=target_namespace)

    # --------------------------
    # Операции удаления
    # --------------------------
    def delete_by_ids(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_namespace = self.namespace if namespace is None else namespace
        return self.index.delete(ids=list(ids), namespace=target_namespace)

    def delete_by_filter(
        self,
        filter: Dict[str, Any],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_namespace = self.namespace if namespace is None else namespace
        return self.index.delete(filter=filter, namespace=target_namespace)

    def delete_all(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        target_namespace = self.namespace if namespace is None else namespace
        return self.index.delete(delete_all=True, namespace=target_namespace)
