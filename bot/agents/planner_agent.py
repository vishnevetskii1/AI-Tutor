# Роль файла: Генерирует учебный план по дисциплине.
import json
import logging
from bot.agents.state import State, Block
from bot.agents.llm import generate
from bot.agents.generation_utils import ensure_topic_list, normalize_topic_name, parse_json_array
from bot.disciplines import get_discipline_label
from bot.rag.search import get_structure_context

logger = logging.getLogger(__name__)

GENERIC_SUBTOPIC_PATTERNS = (
    "смысл и ключевые определения",
    "как распознать тему",
    "типичные ошибки и проверка результата",
    "ключевые определения темы",
)

DISCIPLINE_SUBTOPIC_FALLBACKS = {
    "probability": {
        "Случайные события и операции над ними": [
            "Достоверные, невозможные и случайные события",
            "Сумма, произведение и противоположное событие",
            "Несовместные события",
        ],
        "Классическое определение вероятности": [
            "Равновозможные исходы",
            "Формула P(A)=m/n",
            "Подсчёт числа благоприятных исходов",
        ],
        "Геометрическая вероятность": [
            "Вероятность через длину, площадь и объём",
            "Равномерный выбор точки",
            "Геометрическая модель задачи",
        ],
        "Условная вероятность": [
            "Событие при известном условии",
            "Формула условной вероятности",
            "Переход от исходной вероятности к уточнённой",
        ],
        "Независимость событий": [
            "Определение независимости",
            "Проверка через произведение вероятностей",
            "Отличие независимости от несовместности",
        ],
        "Формула полной вероятности": [
            "Разбиение на гипотезы",
            "Группа попарно несовместных событий",
            "Сборка вероятности по случаям",
        ],
        "Формула Байеса": [
            "Апостериорная вероятность гипотезы",
            "Обновление вероятностей после наблюдения",
            "Связь с полной вероятностью",
        ],
        "Схема Бернулли и биномиальное распределение": [
            "Независимые испытания Бернулли",
            "Вероятность k успехов из n",
            "Биномиальные коэффициенты",
        ],
        "Случайные величины и их распределения": [
            "Дискретные и непрерывные случайные величины",
            "Закон распределения",
            "Функция распределения",
        ],
        "Математическое ожидание и дисперсия": [
            "Среднее значение случайной величины",
            "Разброс и дисперсия",
            "Свойства ожидания и дисперсии",
        ],
    },
    "mathstat": {
        "Генеральная совокупность и выборка": [
            "Объект наблюдения и признак",
            "Объём выборки",
            "Репрезентативность выборки",
        ],
        "Вариационный ряд и эмпирическое распределение": [
            "Упорядоченная выборка",
            "Частоты и относительные частоты",
            "Эмпирическая функция распределения",
        ],
        "Точечные оценки параметров": [
            "Выборочное среднее",
            "Выборочная дисперсия",
            "Оценивание неизвестных параметров",
        ],
        "Свойства оценок: несмещенность и состоятельность": [
            "Несмещённость оценки",
            "Состоятельность",
            "Эффективность оценки",
        ],
        "Интервальные оценки": [
            "Доверительный интервал",
            "Уровень доверия",
            "Интервал для среднего и дисперсии",
        ],
        "Статистические гипотезы": [
            "Нулевая и альтернативная гипотезы",
            "Ошибка первого и второго рода",
            "Критическая область",
        ],
        "Проверка гипотез для среднего значения": [
            "Проверка гипотезы о среднем",
            "Z- и t-критерии",
            "Выбор статистики критерия",
        ],
        "Проверка гипотез для дисперсии": [
            "Гипотезы о дисперсии",
            "Критерий хи-квадрат",
            "Сравнение разброса данных",
        ],
        "Корреляция и регрессия": [
            "Коэффициент корреляции",
            "Линейная регрессия",
            "Интерпретация связи между признаками",
        ],
        "Критерии согласия и анализ данных": [
            "Согласование выборки с распределением",
            "Критерий Пирсона",
            "Интерпретация статистических выводов",
        ],
    },
}

SYSTEM_PROMPT = (
    "Ты методист, составляешь учебные планы. "
    "Генерируй ТОЛЬКО валидный JSON без пояснений и без markdown-блоков."
)

PLANNER_TIMEOUT_SECONDS = 35.0

PLAN_PROMPT_WITH_CONTEXT = """Ты методист. Перед тобой структурная карта всех доступных PDF-учебников по дисциплине «{discipline}».
Карта построена сканированием всех текстовых чанков каждого PDF: в ней есть найденные заголовки, разделы и срезы содержания по всему учебнику.

[СТРУКТУРНАЯ_КАРТА_УЧЕБНИКОВ]
{context}
[/СТРУКТУРНАЯ_КАРТА_УЧЕБНИКОВ]

Составь учебный план по всей дисциплине для студента уровня «{level}».

Уровни:
- beginner: начинать с самых основ
- intermediate: базу пропускаем, идём к практике
- advanced: сразу глубина, теория, сложные случаи

Количество тем определи сам по реальной структуре всех PDF: обычно от 5 до 20.
Не ограничивайся десятью темами, если в учебниках явно больше содержательных разделов.
Не добавляй темы, которых нет в источниках, и не сокращай программу без причины.

Верни JSON (только JSON, без пояснений, всё на русском языке):
[
  {{
    "topic": "Название темы",
    "subtopics": ["Подтема 1", "Подтема 2", "Подтема 3"]
  }}
]

Требования:
- темы строго из учебников, логически последовательные;
- план должен покрывать основные разделы всех источников, а не только первого PDF;
- у каждой темы `subtopics` — конкретные подпункты, обычно 2-5;
- `subtopics` короткие и предметные, без воды;
- не пиши английские слова, slug-и;
- не делай абстрактные пункты «Обзор», «Практика», «Итоги» без уточнения предмета."""

PLAN_PROMPT_FALLBACK = """Составь учебный план по дисциплине «{discipline}» для студента уровня «{level}».

Уровни:
- beginner: начинать с самых основ
- intermediate: базу пропускаем, идём к практике
- advanced: сразу глубина, теория, сложные случаи

Количество тем — от 8 до 15, исходя из сложности дисциплины.

Верни JSON (только JSON, без пояснений, всё на русском языке):
[
  {{
    "topic": "Название темы",
    "subtopics": ["Подтема 1", "Подтема 2", "Подтема 3"]
  }}
]

Требования:
- темы конкретные, логически связанные и последовательные;
- у каждой темы `subtopics` — 2-5 подпунктов;
- `subtopics` короткие и предметные, без воды;
- не пиши английские слова, slug-и;
- не делай абстрактные пункты «Обзор», «Практика», «Итоги» без уточнения предмета."""


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _fallback_goal(topic: str, discipline_label: str, level: str) -> str:
    if level == "advanced":
        return f"Разобрать тонкие места темы «{topic}» и уверенно применять её в сложных задачах по дисциплине «{discipline_label}»."
    if level == "intermediate":
        return f"Связать тему «{topic}» с типовыми задачами и научиться выбирать подходящий метод решения."
    return f"Понять смысл темы «{topic}», ключевые определения и базовую логику решения простых задач."


def _is_generic_subtopic(value: str) -> bool:
    normalized = value.casefold()
    return any(pattern in normalized for pattern in GENERIC_SUBTOPIC_PATTERNS)


def _fallback_subtopics(discipline: str, topic: str) -> list[str]:
    discipline_key = (discipline or "").strip().casefold()
    mapped = DISCIPLINE_SUBTOPIC_FALLBACKS.get(discipline_key, {}).get(topic, [])
    if mapped:
        return mapped[:]
    return [
        f"Основные понятия темы «{topic}»",
        "Базовые связи между величинами и событиями",
        "Типовые задачи по теме",
    ]


def _normalize_subtopics(raw_subtopics: object, discipline: str, topic: str) -> list[str]:
    fallback_items = _fallback_subtopics(discipline, topic)
    min_required = min(2, len(fallback_items)) if fallback_items else 2

    if not isinstance(raw_subtopics, list):
        return fallback_items

    clean_subtopics: list[str] = []
    seen: set[str] = set()
    for item in raw_subtopics[:5]:
        value = _clean_text(item)
        key = value.casefold()
        if not value or key in seen or _is_generic_subtopic(value):
            continue
        seen.add(key)
        clean_subtopics.append(value)

    if len(clean_subtopics) >= min_required:
        return clean_subtopics

    for item in fallback_items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        clean_subtopics.append(item)
        if len(clean_subtopics) == len(fallback_items):
            break

    return clean_subtopics


MIN_PLAN_BLOCKS = 5
MAX_PLAN_BLOCKS = 20


def _normalize_plan_blocks(raw_items: list, discipline: str, level: str, min_count: int = MIN_PLAN_BLOCKS) -> list[Block]:
    discipline_label = get_discipline_label(discipline)
    plan: list[Block] = []
    seen_topics: set[str] = set()

    for item in raw_items:
        if isinstance(item, dict):
            topic = normalize_topic_name(item.get("topic", ""))
            goal = _clean_text(item.get("goal", ""))
            subtopics = _normalize_subtopics(item.get("subtopics", []), discipline, topic) if topic else []
        else:
            topic = normalize_topic_name(item)
            goal = ""
            subtopics = _fallback_subtopics(discipline, topic) if topic else []

        topic_key = topic.casefold()
        if not topic or topic_key in seen_topics:
            continue

        seen_topics.add(topic_key)
        plan.append({
            "index": len(plan),
            "topic": topic,
            "goal": goal or _fallback_goal(topic, discipline_label, level),
            "subtopics": subtopics,
            "completed": False,
        })
        if len(plan) == MAX_PLAN_BLOCKS:
            return plan

    # Если тем меньше минимума — добиваем из fallback
    if len(plan) < min_count:
        for topic in ensure_topic_list([], discipline, target_count=min_count):
            if len(plan) >= min_count:
                break
            topic_key = topic.casefold()
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)
            plan.append({
                "index": len(plan),
                "topic": topic,
                "goal": _fallback_goal(topic, discipline_label, level),
                "subtopics": _fallback_subtopics(discipline, topic),
                "completed": False,
            })

    return plan


def _format_plan_message(discipline_label: str, plan: list[Block]) -> str:
    lines = [f"📋 *План обучения по «{discipline_label}»*", ""]
    for block in plan:
        lines.append(f"*{block['index'] + 1}. Тема блока:* {block['topic']}")
        lines.append("*Учебные вопросы:*")
        for subtopic in block.get("subtopics", []):
            lines.append(f"• {subtopic}")
        lines.append("")
    return "\n".join(lines)


MAX_CONTEXT_CHARS = 14000


def _build_context(discipline: str) -> str:
    return get_structure_context(discipline, max_chars=MAX_CONTEXT_CHARS)


def planner_agent(state: State) -> dict:
    discipline = state.get("discipline", "неизвестная дисциплина")
    discipline_label = get_discipline_label(discipline)
    level = state.get("level", "beginner")

    context = _build_context(discipline)
    if context:
        prompt = PLAN_PROMPT_WITH_CONTEXT.format(
            discipline=discipline_label,
            level=level,
            context=context,
        )
        logger.info("planner_agent: using RAG context (%d chars) for %s", len(context), discipline)
    else:
        prompt = PLAN_PROMPT_FALLBACK.format(discipline=discipline_label, level=level)
        logger.info("planner_agent: no RAG context, using fallback prompt for %s", discipline)

    try:
        raw = generate(
            prompt,
            SYSTEM_PROMPT,
            label="planner_agent",
            options={"num_predict": 1200, "temperature": 0.1},
            timeout_seconds=PLANNER_TIMEOUT_SECONDS,
        ).strip()
    except Exception as exc:
        logger.warning("planner_agent falling back to static plan for %s: %s", discipline, exc)
        raw = "[]"

    try:
        topics_raw = parse_json_array(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        topics_raw = []

    min_count = 10 if not topics_raw else MIN_PLAN_BLOCKS
    plan = _normalize_plan_blocks(topics_raw, discipline, level, min_count=min_count)
    msg = _format_plan_message(discipline_label, plan)

    return {
        "plan": plan,
        "current_block_idx": 0,
        "response": msg,
        "history": [{"role": "assistant", "content": msg}],
        "mode": "awaiting_start",
    }
