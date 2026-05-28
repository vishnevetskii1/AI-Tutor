# Роль файла: Внешний поиск по теме для режима lesson chat.
from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re

import httpx


USER_AGENT = "ai-tutor-bot/1.0 (curated topic search)"
_PAGE_CACHE: dict[str, list[str]] = {}


@dataclass(frozen=True)
class CuratedSource:
    label: str
    url: str


CURATED_SOURCES: dict[str, list[CuratedSource]] = {
    "probability": [
        CuratedSource(
            "MathProfi: Теория вероятностей",
            "https://mathprofi.ru/teorija_verojatnostei.html",
        ),
        CuratedSource(
            "ИНТУИТ: Теория вероятностей и матстатистика",
            "https://intuit.ru/studies/courses/637/493/lecture/11141",
        ),
        CuratedSource(
            "Stepik: Теория вероятностей",
            "https://stepik.org/course/3089/syllabus",
        ),
    ],
    "mathstat": [
        CuratedSource(
            "MathProfi: Математическая статистика",
            "https://mathprofi.ru/matematicheskaya_statistika.html",
        ),
        CuratedSource(
            "Stepik: Математическая статистика",
            "https://stepik.org/course/326/promo",
        ),
        CuratedSource(
            "ВШЭ: Математическая статистика и A/B тестирование",
            "https://elearning.hse.ru/moocs/mathematical-statistics-and-ab-testing",
        ),
    ],
}


def _build_query(question: str, topic: str, discipline: str, subtopics: list[str] | None = None) -> str:
    parts = [discipline.strip(), topic.strip(), question.strip()]
    if subtopics:
        parts.extend(item.strip() for item in subtopics[:2] if item.strip())
    return " ".join(part for part in parts if part)


def _tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", text or "")
    }


def _normalize_html(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html or "")
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)</?(p|div|section|article|li|ul|ol|h1|h2|h3|h4|br|tr|td|th)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_chunks(text: str, chunk_size: int = 700) -> list[str]:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    for block in blocks:
        clean = " ".join(block.split())
        if len(clean) < 80:
            continue
        if len(clean) <= chunk_size:
            chunks.append(clean)
            continue
        for idx in range(0, len(clean), chunk_size):
            piece = clean[idx:idx + chunk_size].strip()
            if len(piece) >= 80:
                chunks.append(piece)
    return chunks


def _source_chunks(client: httpx.Client, source: CuratedSource) -> list[str]:
    cached = _PAGE_CACHE.get(source.url)
    if cached is not None:
        return cached

    response = client.get(source.url)
    response.raise_for_status()
    text = _normalize_html(response.text)
    chunks = [f"[{source.label}] {chunk}" for chunk in _split_chunks(text)]
    _PAGE_CACHE[source.url] = chunks
    return chunks


def _score_chunk(query: str, chunk: str, topic: str) -> int:
    query_terms = _tokenize(query)
    chunk_terms = _tokenize(chunk)
    if not query_terms or not chunk_terms:
        return 0

    overlap = len(query_terms & chunk_terms)
    if overlap == 0:
        return 0

    score = overlap * 5
    lowered_chunk = chunk.casefold()
    lowered_topic = topic.casefold()
    if lowered_topic and lowered_topic in lowered_chunk:
        score += 8
    if query.casefold() in lowered_chunk:
        score += 12
    return score


def search_topic_web(
    question: str,
    topic: str,
    discipline: str = "",
    subtopics: list[str] | None = None,
    *,
    limit: int = 3,
    timeout_seconds: float = 3.0,
) -> list[str]:
    sources = CURATED_SOURCES.get(discipline, [])
    if not sources:
        return []

    query = _build_query(question, topic, discipline, subtopics=subtopics)
    if not query:
        return []

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=1.5),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            scored: list[tuple[int, str]] = []
            seen: set[str] = set()
            for source in sources:
                for chunk in _source_chunks(client, source):
                    if chunk in seen:
                        continue
                    seen.add(chunk)
                    score = _score_chunk(query, chunk, topic)
                    if score > 0:
                        scored.append((score, chunk))
    except Exception:
        return []

    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [chunk for _, chunk in scored[:limit]]
