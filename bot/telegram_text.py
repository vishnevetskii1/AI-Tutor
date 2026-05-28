# Роль файла: Режет длинные сообщения под лимиты Telegram.
TELEGRAM_TEXT_LIMIT = 3500


def split_text_for_telegram(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= limit:
        return [cleaned]

    parts: list[str] = []
    remaining = cleaned
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit
        part = remaining[:split_at].rstrip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        parts.append(remaining)
    return parts
