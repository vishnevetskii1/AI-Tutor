# Роль файла: Названия дисциплин, описания и служебные helper-функции.
DISPLAY_NAMES = {
    "math": "Математика",
    "physics": "Физика",
    "probability": "Теория вероятностей",
    "mathstat": "Математическая статистика",
}

DISCIPLINE_DESCRIPTIONS = {
    "probability": (
        "Изучает случайные события, вероятности, распределения и базовые модели случайных процессов. "
        "Подойдёт, если хочешь понять логику вероятностных задач с нуля."
    ),
    "mathstat": (
        "Изучает выборки, оценки параметров, гипотезы, корреляцию и обработку данных. "
        "Подойдёт, если хочешь понять, как делать выводы по данным и экспериментам."
    ),
}


def get_discipline_label(slug: str) -> str:
    value = (slug or "").strip()
    if not value:
        return "—"
    return DISPLAY_NAMES.get(value, value.replace("_", " ").replace("-", " ").strip().capitalize())


def get_discipline_description(slug: str) -> str:
    value = (slug or "").strip()
    if not value:
        return ""
    return DISCIPLINE_DESCRIPTIONS.get(value, "")


def list_textbook_sources() -> dict[str, list[str]]:
    return {
        "probability": [
            "Александр Емелин — «Практикум по теории вероятностей. Краткий курс для начинающих»",
            "Н. И. Чернова — «Теория вероятностей»",
        ],
        "mathstat": [
            "Н. И. Чернова — «Математическая статистика»",
        ],
    }
