# Роль файла: Нормализует ответы перед отправкой пользователю.
import re


def sanitize_teacher_voice(text: str) -> str:
    cleaned = (text or "").strip()
    replacements = [
        (r"\bв этом учебнике\b", "обычно"),
        (r"\bв данном учебнике\b", "обычно"),
        (r"\bв учебнике\b", "обычно"),
        (r"\bпо учебнику\b", "по этой теме"),
        (r"\bв контексте\b", "здесь"),
        (r"\bв тексте выше\b", "здесь"),
        (r"\bсогласно учебнику\b", "обычно"),
        (r"\bавтор учебника\b", "обычно"),
    ]
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\$([A-Za-zА-Яа-я0-9_()=+\-*/.,\s]{1,40})\$", r"\1", cleaned)
    return cleaned
