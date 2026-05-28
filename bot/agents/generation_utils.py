# Роль файла: Общие helper-функции для парсинга и очистки генерации.
import json
import re

GENERIC_QUIZ_QUESTION_PATTERNS = (
    "какое утверждение верно по материалу блока",
    "лучше всего описывает тему",
)

GENERIC_QUIZ_OPTION_PATTERNS = (
    "это базовое определение",
    "это случайный факт",
    "это только историческая справка",
    "это исключительно вычислительный трюк",
    "не связанный с темой",
)

QUIZ_GROUNDING_STOPWORDS = {
    "это", "как", "что", "какой", "какая", "какие", "какое", "когда", "где", "если",
    "или", "для", "при", "над", "под", "про", "ещё", "уже", "только", "нужно",
    "можно", "нельзя", "тема", "блок", "ответ", "вариант", "вопрос", "из", "по",
}


def strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def extract_json_array(text: str) -> str:
    raw = strip_code_fences(text)
    if raw.startswith("[") and raw.endswith("]"):
        return raw

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and start < end:
        return raw[start:end + 1]
    return raw


def extract_json_object(text: str) -> str:
    raw = strip_code_fences(text)
    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and start < end:
        return raw[start:end + 1]
    return raw


def normalize_topic_name(topic: str) -> str:
    value = re.sub(r"^\s*(\d+[\).\s-]+|[-*•]+\s*)", "", (topic or "").strip())
    value = value.strip(" \"'`")
    return re.sub(r"\s+", " ", value).strip()


def fallback_topics(discipline: str) -> list[str]:
    normalized = (discipline or "").strip().casefold()

    if normalized in {"probability", "теория вероятностей", "тервер", "теорвер"}:
        return [
            "Случайные события и операции над ними",
            "Классическое определение вероятности",
            "Геометрическая вероятность",
            "Условная вероятность",
            "Независимость событий",
            "Формула полной вероятности",
            "Формула Байеса",
            "Схема Бернулли и биномиальное распределение",
            "Случайные величины и их распределения",
            "Математическое ожидание и дисперсия",
        ]

    if normalized in {"mathstat", "матстат", "математическая статистика"}:
        return [
            "Генеральная совокупность и выборка",
            "Вариационный ряд и эмпирическое распределение",
            "Точечные оценки параметров",
            "Свойства оценок: несмещенность и состоятельность",
            "Интервальные оценки",
            "Статистические гипотезы",
            "Проверка гипотез для среднего значения",
            "Проверка гипотез для дисперсии",
            "Корреляция и регрессия",
            "Критерии согласия и анализ данных",
        ]

    return [
        f"Введение в {discipline}",
        f"Базовые понятия {discipline}",
        "Ключевые определения и обозначения",
        "Основные типы задач",
        "Базовые методы решения",
        "Типичные ошибки и ловушки",
        "Практические примеры",
        "Задачи повышенной сложности",
        "Систематизация и повторение",
        "Итоговый обзор дисциплины",
    ]


def ensure_topic_list(raw_topics: list, discipline: str, target_count: int = 10) -> list[str]:
    topics: list[str] = []
    seen: set[str] = set()

    for item in raw_topics:
        topic = normalize_topic_name(item.get("topic") if isinstance(item, dict) else str(item))
        key = topic.casefold()
        if not topic or key in seen:
            continue
        seen.add(key)
        topics.append(topic)
        if len(topics) == target_count:
            return topics

    for topic in fallback_topics(discipline):
        key = topic.casefold()
        if key in seen:
            continue
        seen.add(key)
        topics.append(topic)
        if len(topics) == target_count:
            break

    return topics


def _quiz_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", text or "")
        if token.casefold() not in QUIZ_GROUNDING_STOPWORDS
    }


def _extract_context_facts(source_text: str, limit: int = 6) -> list[str]:
    cleaned = re.sub(r"[*_`#]", " ", source_text or "")
    raw_parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", cleaned)
    facts: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        fact = " ".join(part.split()).strip(" -•")
        if len(fact) < 40 or len(fact) > 220:
            continue
        lowered = fact.casefold()
        if any(
            marker in lowered for marker in (
                "готов к квизу",
                "задай уточняющий вопрос",
                "выбери один вариант",
                "вопрос ",
                "квиз по теме",
            )
        ):
            continue
        key = lowered
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
        if len(facts) == limit:
            break

    return facts


def _fallback_fact_category(fact: str) -> str:
    lowered = fact.casefold()

    if any(marker in fact for marker in ("P(", "=", "m/n", "n/m")):
        return "formula"
    if any(marker in lowered for marker in ("наудачу", "случайно выбрано", "честный", "формулировк", "указывают")):
        return "recognition"
    if any(marker in lowered for marker in ("не путать", "важно не", "нельзя", "ошиб", "ненулев")):
        return "warning"
    if any(marker in lowered for marker in ("равновозмож", "одинаково вероят", "конечн", "число исход", "всех исход")):
        return "condition"
    if any(marker in lowered for marker in ("когда", "если", "при ", "применяют", "используют", "выполнено")):
        return "applicability"
    if any(marker in lowered for marker in ("это", "означает", "называют", "состоит")):
        return "definition"
    return "generic"


def _select_fallback_facts(facts: list[str], limit: int = 3) -> list[str]:
    if len(facts) <= limit:
        return facts

    priority = {
        "formula": 0,
        "recognition": 1,
        "warning": 2,
        "condition": 3,
        "applicability": 4,
        "definition": 5,
        "generic": 6,
    }

    selected: list[str] = []
    seen_categories: set[str] = set()

    for fact in sorted(facts, key=lambda item: (priority.get(_fallback_fact_category(item), 99), facts.index(item))):
        category = _fallback_fact_category(fact)
        if category in seen_categories:
            continue
        seen_categories.add(category)
        selected.append(fact)
        if len(selected) == limit:
            return selected

    for fact in facts:
        if fact in selected:
            continue
        selected.append(fact)
        if len(selected) == limit:
            break

    return selected


def _fallback_question_stem(topic: str, fact: str, index: int) -> str:
    lowered = fact.casefold()
    topic_label = f"«{topic}»" if topic else "этой темы"

    if any(marker in fact for marker in ("P(", "=", "m/n", "n/m")):
        stems = (
            f"Какая формула используется в теме {topic_label}?",
            f"Какое равенство выражает основную вычислительную схему темы {topic_label}?",
            f"Какая запись в разборе была ключевой формулой темы {topic_label}?",
        )
        return stems[index % len(stems)]

    if any(marker in lowered for marker in ("равновозмож", "одинаково вероят")):
        stems = (
            f"Какое условие обязательно для применения темы {topic_label}?",
            f"Без какого свойства исходов тема {topic_label} применять нельзя?",
            f"Какое требование к исходам отдельно подчёркивали в теме {topic_label}?",
        )
        return stems[index % len(stems)]

    if any(marker in lowered for marker in ("конечн", "число исход", "всех исход")):
        stems = (
            f"С каким множеством исходов работает тема {topic_label}?",
            f"Какое ограничение на число исходов важно для темы {topic_label}?",
            f"Какой тип пространства исходов предполагался в теме {topic_label}?",
        )
        return stems[index % len(stems)]

    if any(marker in lowered for marker in ("наудачу", "случайно выбрано", "честный", "формулировк", "указывают")):
        stems = (
            f"Какая формулировка в условии задачи подсказывает тему {topic_label}?",
            f"Какие слова в условии обычно сигнализируют, что подходит тема {topic_label}?",
            f"По какому словесному признаку в задаче узнают тему {topic_label}?",
        )
        return stems[index % len(stems)]

    if any(marker in lowered for marker in ("не путать", "важно не", "нельзя", "ошиб", "ненулев")):
        stems = (
            f"Какую важную оговорку отдельно подчеркнули в теме {topic_label}?",
            f"Какую типичную ошибку или ловушку разбирали в теме {topic_label}?",
            f"Какое ограничение важно не потерять при решении по теме {topic_label}?",
        )
        return stems[index % len(stems)]

    if any(marker in lowered for marker in ("когда", "если", "при ", "применяют", "используют", "выполнено")):
        stems = (
            f"Когда по смыслу темы {topic_label} можно применять этот подход?",
            f"При каком условии в задаче по теме {topic_label} переходят к этому рассуждению?",
            f"В какой ситуации разбор темы {topic_label} советует использовать этот приём?",
        )
        return stems[index % len(stems)]

    if any(marker in lowered for marker in ("это", "означает", "называют", "состоит")):
        stems = (
            f"Что в теме {topic_label} означает этот тезис?",
            f"Как в теме {topic_label} формулировали основной смысл этого пункта?",
            f"Какое утверждение точнее всего передаёт ключевую идею темы {topic_label}?",
        )
        return stems[index % len(stems)]

    stems = (
        f"Какой факт относится к теме {topic_label}?",
        f"Какое утверждение было верным именно для темы {topic_label}?",
        f"Какой тезис соответствует одному из опорных пунктов темы {topic_label}?",
    )
    return stems[index % len(stems)]


def fallback_quiz_questions(topic: str, source_text: str = "") -> list[dict]:
    facts = _extract_context_facts(source_text, limit=6)
    if len(facts) < 3:
        return [
            {
                "q": "Какая мысль действительно прозвучала в разборе блока?",
                "options": [
                    f"В блоке разбиралась ключевая идея темы «{topic}» на конкретных примерах",
                    f"В блоке обсуждалась только история появления темы «{topic}»",
                    f"В блоке полностью отказались от объяснений и примеров",
                    f"В блоке рассматривалась другая тема, не связанная с «{topic}»",
                ],
                "answer": 0,
                "source_fact": f"В блоке разбиралась ключевая идея темы «{topic}» на конкретных примерах",
            },
            {
                "q": "Что в содержательном разборе темы делают сначала?",
                "options": [
                    "Сначала проясняют смысл ключевой идеи и обозначений",
                    "Сразу выбирают случайную формулу без разбора",
                    "Полностью игнорируют условия и ограничения",
                    "Оценивают тему только по количеству вычислений",
                ],
                "answer": 0,
                "source_fact": "Сначала проясняют смысл ключевой идеи и обозначений",
            },
            {
                "q": "Какой вариант больше похож на нормальное объяснение темы, а не на шум?",
                "options": [
                    "Пояснение смысла темы, примера и типичной ловушки",
                    "Набор случайных фактов без связи с материалом",
                    "Только исторические сведения без разбора идеи",
                    "Только формулы без объяснения, что они означают",
                ],
                "answer": 0,
                "source_fact": "Пояснение смысла темы, примера и типичной ловушки",
            },
        ]

    questions: list[dict] = []
    selected_facts = _select_fallback_facts(facts, limit=3)
    for idx, fact in enumerate(selected_facts):
        distractors = [item for item in facts if item != fact][:3]
        while len(distractors) < 3:
            distractors.append(f"Это утверждение не было опорной мыслью блока по теме «{topic}».")
        questions.append({
            "q": _fallback_question_stem(topic, fact, idx),
            "options": [fact, distractors[0], distractors[1], distractors[2]],
            "answer": 0,
            "source_fact": fact,
        })

    return questions


def _looks_generic_quiz_question(question: str, topic: str) -> bool:
    normalized = question.casefold()
    topic_lower = (topic or "").strip().casefold()
    return any(pattern in normalized for pattern in GENERIC_QUIZ_QUESTION_PATTERNS) or (
        topic_lower and normalized == f"что лучше всего описывает тему «{topic_lower}»?"
    )


def _looks_generic_quiz_option(option: str) -> bool:
    normalized = option.casefold()
    return any(pattern in normalized for pattern in GENERIC_QUIZ_OPTION_PATTERNS)


def _is_grounded_in_source(item: dict, topic: str, source_text: str) -> bool:
    if not source_text.strip():
        return True

    topic_tokens = _quiz_tokens(topic)
    source_tokens = _quiz_tokens(source_text) - topic_tokens
    if not source_tokens:
        return True

    question = " ".join(str(item.get("q", "")).split()).strip()
    options = item.get("options") if isinstance(item.get("options"), list) else []
    answer = item.get("answer")
    source_fact = " ".join(str(item.get("source_fact", "")).split()).strip()

    correct_option = ""
    if isinstance(answer, int) and 0 <= answer < len(options):
        correct_option = str(options[answer])

    candidate_tokens = (_quiz_tokens(question) | _quiz_tokens(correct_option) | _quiz_tokens(source_fact)) - topic_tokens
    overlap = candidate_tokens & source_tokens
    return len(overlap) >= 2


def valid_quiz_questions(raw_questions: list, topic: str, source_text: str = "", target_count: int = 3) -> list[dict]:
    questions: list[dict] = []
    seen_questions: set[str] = set()

    for item in raw_questions:
        if not isinstance(item, dict):
            continue

        question = " ".join(str(item.get("q", "")).split()).strip()
        options = item.get("options")
        answer = item.get("answer")
        source_fact = " ".join(str(item.get("source_fact", "")).split()).strip()

        if not question or not isinstance(options, list) or _looks_generic_quiz_question(question, topic):
            continue

        clean_options = []
        seen_options: set[str] = set()
        for option in options[:4]:
            value = " ".join(str(option).split()).strip()
            key = value.casefold()
            if value and key not in seen_options:
                seen_options.add(key)
                clean_options.append(value)

        if len(clean_options) != 4:
            continue

        if any(_looks_generic_quiz_option(option) for option in clean_options):
            continue

        if not isinstance(answer, int) or answer < 0 or answer > 3:
            continue

        if not _is_grounded_in_source(item, topic, source_text):
            continue

        question_key = question.casefold()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)

        questions.append({
            "q": question,
            "options": clean_options,
            "answer": answer,
            "source_fact": source_fact,
        })
        if len(questions) == target_count:
            return questions

    return questions


def ensure_quiz_questions(raw_questions: list, topic: str, source_text: str = "", target_count: int = 3) -> list[dict]:
    questions = valid_quiz_questions(raw_questions, topic, source_text=source_text, target_count=target_count)

    fallback_candidates = valid_quiz_questions(
        fallback_quiz_questions(topic, source_text=source_text),
        topic,
        source_text=source_text,
        target_count=target_count,
    )

    for fallback in fallback_candidates:
        questions.append(fallback)
        if len(questions) == target_count:
            break

    return questions


def parse_json_array(text: str):
    return json.loads(extract_json_array(text))


def parse_json_object(text: str):
    return json.loads(extract_json_object(text))
