from pathlib import Path
from typing import Any, Dict, List

import os

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".xlsx",
    ".html",
    ".md",
    ".txt",
    ".csv",
}


class DoclingProcessor:
    def __init__(self) -> None:
        # Reduce memory pressure/concurrency for docling/torch on limited Windows setups.
        os.environ.setdefault("DOCLING_PARALLELISM", "1")
        os.environ.setdefault("RAPIDOCR_DEVICE", "cpu")
        try:
            import torch  # type: ignore

            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Install package 'docling' to enable file ingestion."
            ) from exc
        # Create the converter after environment tuning.
        self._converter = DocumentConverter()

    def is_supported(self, file_name: str) -> bool:
        return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS

    def to_chunks(self, file_path: str, file_name: str) -> List[Dict[str, Any]]:
        # Guard against memory errors during conversion; attempt a safe fallback later.
        try:
            conversion_result = self._converter.convert(file_path)
        except MemoryError:
            # Return an informative minimal chunk to avoid crashing the pipeline.
            return [
                {
                    "text": "Обработка файла прервана: недостаточно памяти при предварительной обработке (std::bad_alloc). "
                    "Попробуйте уменьшить размер/количество страниц документа.",
                    "metadata": {"filename": file_name, "chunk_no": 1},
                }
            ]
        except Exception:
            # Propagate non-memory errors downstream to fallback markdown path.
            conversion_result = self._converter.convert(file_path)
        document = conversion_result.document

        try:
            from docling.chunking import HybridChunker  # type: ignore

            chunker = HybridChunker()
            chunk_objects = list(chunker.chunk(document))
            chunks: List[Dict[str, Any]] = []
            for idx, chunk in enumerate(chunk_objects, start=1):
                text = getattr(chunk, "text", "") or ""
                if not text.strip():
                    continue
                meta = {"filename": file_name, "chunk_no": idx}
                page_no = None
                try:
                    if chunk.meta and chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
                        page_no = chunk.meta.doc_items[0].prov[0].page_no
                except Exception:
                    page_no = None
                if page_no is not None:
                    meta["page_no"] = page_no
                chunks.append({"text": text, "metadata": meta})
            if chunks:
                return chunks
        except Exception:
            pass

        markdown = document.export_to_markdown()
        fallback_chunks: List[Dict[str, Any]] = []
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        chunk_size = 30
        for i in range(0, len(lines), chunk_size):
            text = "\n".join(lines[i : i + chunk_size]).strip()
            if not text:
                continue
            fallback_chunks.append(
                {
                    "text": text,
                    "metadata": {"filename": file_name, "chunk_no": len(fallback_chunks) + 1},
                }
            )
        return fallback_chunks
