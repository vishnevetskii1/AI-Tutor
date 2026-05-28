# Роль файла: Подписи уровней и связанные helper-функции.
LEVEL_LABELS = {
    "beginner": "начинающий",
    "intermediate": "средний",
    "advanced": "продвинутый",
}


def get_level_label(level: str | None) -> str:
    if not level:
        return "не определён"
    return LEVEL_LABELS.get(level, level)
