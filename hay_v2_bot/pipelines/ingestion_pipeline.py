import uuid
from typing import Any, Dict, List

from haystack import Pipeline, component

try:
    from hay_v2_bot.components.docling_processor import DoclingProcessor
    from hay_v2_bot.components.memory_service import MemoryService
except ModuleNotFoundError:
    from components.docling_processor import DoclingProcessor
    from components.memory_service import MemoryService


@component
class DoclingChunkComponent:
    def __init__(self, processor: DoclingProcessor):
        self.processor = processor

    @component.output_types(chunks=List[Dict[str, Any]])
    def run(self, file_path: str, file_name: str) -> Dict[str, Any]:
        chunks = self.processor.to_chunks(file_path=file_path, file_name=file_name)
        return {"chunks": chunks}


@component
class PineconeUpsertComponent:
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    @component.output_types(upsert_result=dict, prepared_documents=list)
    def run(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        prepared_documents: List[Dict[str, Any]] = []
        for chunk in chunks:
            prepared_documents.append(
                {
                    "id": f"doc-{uuid.uuid4().hex}",
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )
        upsert_result = self.memory_service.upsert_doc_chunks(prepared_documents)
        return {"upsert_result": upsert_result, "prepared_documents": prepared_documents}


def build_ingestion_pipeline(processor: DoclingProcessor, memory_service: MemoryService) -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("docling_chunker", DoclingChunkComponent(processor))
    pipe.add_component("pinecone_writer", PineconeUpsertComponent(memory_service))
    pipe.connect("docling_chunker.chunks", "pinecone_writer.chunks")
    return pipe
