# Роль файла: Выполняет поиск в Qdrant и резервный поиск по PDF.
import logging
import re
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from bot.config import load_config
from bot.rag.pdf_audit import extract_pdf_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
logger = logging.getLogger(__name__)
STOPWORDS = {
    "и", "или", "над", "при", "для", "это", "что", "как", "где", "если",
    "они", "она", "оно", "них", "ними", "его", "ее", "её", "the",
}

# Пара (source_file, text) — базовая единица RAG-записи во внутренних функциях.
ChunkRecord = tuple[str, str]

_model = None
_FileSignature = tuple[tuple[str, int, int], ...]
_fallback_cache: dict[str, tuple[_FileSignature, list[str]]] = {}
_fallback_record_cache: dict[str, tuple[_FileSignature, list[ChunkRecord]]] = {}

STRUCTURE_MARKER_RE = re.compile(
    r"(?:Глава|ГЛАВА|Раздел|РАЗДЕЛ)\s+[A-ZА-Яа-яIVXLC0-9]+[.:]?\s+[^.]{3,140}"
    r"|§\s*\d+(?:\.\d+)*\.?\s*[^§.]{3,140}"
    r"|\b\d+(?:\.\d+){1,3}\.?\s+[А-ЯЁ][^.]{5,140}"
)
NUMERIC_MARKER_RE = re.compile(r"^(?P<head>\d+)(?:\.\d+)+")


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Модель поднимаем лениво, чтобы не грузить её при каждом импорте модуля.
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _model


def get_structure_chunks(discipline: str, max_chunks: int = 40) -> list[str]:
    """Возвращает первые чанки из PDF — там обычно оглавление и структура учебника.
    Не применяет штрафы за 'содержание'/'глава' — они нужны для анализа структуры."""
    chunks = _load_fallback_chunk_records(discipline)
    if not chunks:
        return []
    # Берём первые max_chunks чанков (начало книги = оглавление + введение)
    return [text for _, text in chunks[:max_chunks]]


def get_structure_context(discipline: str, max_chars: int = 14000) -> str:
    """Собирает компактную карту структуры по всем читаемым PDF дисциплины."""
    records = _load_fallback_chunk_records(discipline)
    return build_structure_context_from_records(records, max_chars=max_chars)


def build_structure_context_from_records(records: list[ChunkRecord], max_chars: int = 14000) -> str:
    grouped: dict[str, list[str]] = {}
    for source, chunk in records:
        cleaned = _clean_snippet(chunk, max_chars=900)
        if not cleaned:
            continue
        grouped.setdefault(source, []).append(cleaned)

    if not grouped:
        return ""

    source_budget = max(1800, max_chars // len(grouped))
    parts: list[str] = []
    for source, chunks in grouped.items():
        markers = _extract_structure_markers(chunks, limit=90)
        samples = _sample_chunks_evenly(chunks, sample_count=14, max_chars=420)

        lines = [
            f"### Источник: {source}",
            f"Всего текстовых чанков: {len(chunks)}",
            "Структурные маркеры, найденные сканированием всего учебника:",
        ]
        if markers:
            lines.extend(f"- {marker}" for marker in markers)
        else:
            lines.append("- Явные заголовки не найдены; смотри срезы содержания ниже.")

        lines.append("Срезы содержания по всему учебнику:")
        lines.extend(f"- {sample}" for sample in samples)

        part = "\n".join(lines)
        if len(part) > source_budget:
            part = part[:source_budget].rsplit("\n", 1)[0].rstrip() + "\n- ..."
        parts.append(part)

    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n..."
    return context


def _clean_snippet(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").replace("\xa0", " ").split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def _extract_structure_markers(chunks: list[str], limit: int = 90) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for match in STRUCTURE_MARKER_RE.finditer(chunk):
            marker = _clean_snippet(match.group(0), max_chars=180).strip(" .;:-")
            if len(marker) < 8:
                continue
            if _is_noisy_structure_marker(marker):
                continue
            key = marker.casefold()
            if key in seen:
                continue
            seen.add(key)
            markers.append(marker)
            if len(markers) >= limit:
                return markers
    return markers


def _is_noisy_structure_marker(marker: str) -> bool:
    if "ББК" in marker or "УДК" in marker:
        return True
    numeric_match = NUMERIC_MARKER_RE.match(marker)
    if numeric_match and int(numeric_match.group("head")) > 15:
        return True
    return False


def _sample_chunks_evenly(chunks: list[str], sample_count: int = 14, max_chars: int = 420) -> list[str]:
    if not chunks:
        return []
    if len(chunks) <= sample_count:
        indices = list(range(len(chunks)))
    else:
        indices = sorted({
            round(index * (len(chunks) - 1) / (sample_count - 1))
            for index in range(sample_count)
        })
    return [_clean_snippet(chunks[index], max_chars=max_chars) for index in indices]


def search(query: str, discipline: str, top_k: int = 5) -> list[str]:
    config = load_config()
    candidates: list[tuple[str, str]] = []
    try:
        # Сначала пытаемся получить кандидатов из Qdrant.
        qdrant = QdrantClient(url=config.qdrant_url, timeout=config.qdrant_timeout_seconds)

        existing = [c.name for c in qdrant.get_collections().collections]
        if discipline not in existing:
            return _fallback_pdf_search(query, discipline, top_k=top_k)

        model = _get_model()
        query_vector = model.encode(query).tolist()

        hits = qdrant.query_points(
            collection_name=discipline,
            query=query_vector,
            limit=max(top_k * 4, 8),
        ).points
        candidates.extend(
            (
                str(hit.payload.get("source_file") or hit.payload.get("source") or "qdrant"),
                hit.payload["text"],
            )
            for hit in hits
            if hit.payload and "text" in hit.payload
        )
    except Exception as exc:
        logger.warning("RAG search unavailable for discipline=%s: %s", discipline, exc)

    # Даже при рабочем Qdrant добавляем резервных кандидатов из PDF, чтобы не терять ответы.
    candidates.extend(_fallback_pdf_search_records(query, discipline, top_k=max(top_k * 4, 8)))
    # Потом единообразно пересортировываем всё общим ранжированием.
    reranked = _rerank_chunks(query, candidates, top_k=top_k)
    if reranked:
        return reranked
    return _fallback_pdf_search(query, discipline, top_k=top_k)


def _fallback_pdf_search_records(query: str, discipline: str, top_k: int = 5) -> list[ChunkRecord]:
    chunks = _load_fallback_chunk_records(discipline)
    if not chunks:
        return []

    query_terms = _query_terms(query)
    if not query_terms:
        return chunks[:top_k]

    scored = []
    for source, chunk in chunks:
        score = _score_chunk(query, chunk)
        if score <= 0:
            continue
        scored.append((score, source, chunk))

    scored.sort(key=lambda item: (-item[0], len(item[2])))
    return [(source, chunk) for _, source, chunk in scored[:top_k]]


def _load_fallback_chunk_records(discipline: str) -> list[ChunkRecord]:
    config = load_config()
    discipline_dir = Path(config.disciplines_dir) / discipline
    if not discipline_dir.exists():
        _fallback_cache[discipline] = ((), [])
        _fallback_record_cache[discipline] = ((), [])
        return []

    signature = tuple(
        (pdf_path.name, pdf_path.stat().st_size, pdf_path.stat().st_mtime_ns)
        for pdf_path in sorted(discipline_dir.glob("*.pdf"))
    )
    cached = _fallback_record_cache.get(discipline)
    if cached and cached[0] == signature:
        return cached[1]

    chunks: list[tuple[str, str]] = []
    for pdf_path in sorted(discipline_dir.glob("*.pdf")):
        try:
            audit, pdf_chunks = extract_pdf_chunks(pdf_path, chunk_size=220, overlap=40)
        except Exception as exc:
            logger.warning("Failed to read PDF fallback %s: %s", pdf_path, exc)
            continue
        if audit.is_suspicious:
            # Подозрительные PDF не используем, чтобы не засорять поиск мусорным текстом.
            logger.warning(
                "Skipping low-quality PDF in fallback search %s: %s",
                pdf_path,
                ", ".join(audit.issues),
            )
            continue
        chunks.extend((pdf_path.name, chunk) for chunk in pdf_chunks)

    _fallback_record_cache[discipline] = (signature, chunks)
    _fallback_cache[discipline] = (signature, [chunk for _, chunk in chunks])
    return chunks


def _load_fallback_chunks(discipline: str) -> list[str]:
    return [chunk for _, chunk in _load_fallback_chunk_records(discipline)]


def _fallback_pdf_search(query: str, discipline: str, top_k: int = 5) -> list[str]:
    return [text for _, text in _fallback_pdf_search_records(query, discipline, top_k=top_k)]


def _tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", text or "")
    }


def _query_terms(text: str) -> set[str]:
    return {token for token in _tokenize(text) if token not in STOPWORDS}


def _is_noisy_chunk(chunk: str) -> bool:
    lowered = (chunk or "").casefold()
    if lowered.startswith("содержание"):
        return True
    if lowered.count("глава") >= 3:
        return True
    if lowered.count("§") >= 3:
        return True
    if ". . ." in lowered:
        return True
    return False


def _score_chunk(query: str, chunk: str) -> int:
    if not chunk:
        return 0

    score = 0
    lowered_chunk = chunk.casefold()
    lowered_query = query.casefold()
    query_terms = _query_terms(query)
    chunk_terms = _tokenize(chunk)
    overlap = len(query_terms & chunk_terms)
    if overlap == 0 and lowered_query not in lowered_chunk:
        return 0

    score += overlap * 4
    if lowered_query in lowered_chunk:
        score += 20

    ordered_terms = [
        token.casefold()
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", query or "")
        if token.casefold() not in STOPWORDS
    ]
    for left, right in zip(ordered_terms, ordered_terms[1:]):
        if f"{left} {right}" in lowered_chunk:
            score += 6

    if _is_noisy_chunk(chunk):
        score -= 12

    return score


def _rerank_chunks(query: str, chunks: list[ChunkRecord], top_k: int) -> list[str]:
    seen: set[str] = set()
    scored: list[tuple[int, str, str]] = []
    for source, chunk in chunks:
        cleaned = (chunk or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        score = _score_chunk(query, cleaned)
        if score <= 0:
            continue
        scored.append((score, source, cleaned))

    scored.sort(key=lambda item: (-item[0], len(item[2])))

    picked: list[str] = []
    used_sources: set[str] = set()

    for _, source, chunk in scored:
        if len(picked) >= top_k:
            break
        if source in used_sources:
            continue
        picked.append(chunk)
        used_sources.add(source)

    if len(picked) < top_k:
        for _, _, chunk in scored:
            if len(picked) >= top_k:
                break
            if chunk in picked:
                continue
            picked.append(chunk)

    return picked
