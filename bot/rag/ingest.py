# Роль файла: Индексирует чанки в векторное хранилище Qdrant.
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

from bot.config import load_config
from bot.rag.pdf_audit import extract_pdf_chunks

VECTOR_SIZE = 384   # all-MiniLM-L6-v2 dimension
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _ensure_collection(client: QdrantClient, name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _point_id(pdf_path: str | Path, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{Path(pdf_path)}::{chunk_index}"))


def ingest(pdf_path: str | Path, discipline_slug: str) -> None:
    config = load_config()
    audit, chunks = extract_pdf_chunks(pdf_path, chunk_size=500, overlap=50)
    if audit.is_suspicious:
        raise ValueError(
            f"PDF {audit.pdf_path.name} is not searchable enough for RAG: "
            + ", ".join(audit.issues)
        )

    model = _get_model()
    qdrant = QdrantClient(url=config.qdrant_url)
    _ensure_collection(qdrant, discipline_slug)

    batch_size = 64
    total = len(chunks)
    points = []

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectors = model.encode(batch, show_progress_bar=False)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        for offset, (chunk, vector) in enumerate(zip(batch, vectors), start=i):
            points.append(
                PointStruct(
                    id=_point_id(pdf_path, offset),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": str(pdf_path),
                        "source_file": Path(pdf_path).name,
                        "discipline": discipline_slug,
                        "chunk_index": offset,
                        "pdf_page_count": audit.page_count,
                        "pdf_word_count": audit.extracted_word_count,
                        "pdf_text_page_ratio": audit.text_page_ratio,
                        "pdf_extraction_method": audit.extraction_method,
                    },
                )
            )
        print(f"  {min(i + batch_size, total)}/{total} чанков обработано")

    qdrant.upsert(collection_name=discipline_slug, points=points)
