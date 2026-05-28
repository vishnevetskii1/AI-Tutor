# Роль файла: Режет извлечённый PDF-текст на чанки с перехлёстом.
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Делит текст на чанки по числу слов с небольшим перехлёстом."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks
