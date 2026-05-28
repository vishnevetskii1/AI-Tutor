# Роль файла: Правила валидации для онбординга и проверки релевантности.
import re
from collections import Counter


STOPWORDS = {
    "и", "или", "это", "что", "как", "где", "когда", "если", "ли", "для",
    "про", "под", "над", "при", "без", "the", "with", "from",
}

LEVEL_KEYWORDS = {
    "beginner": {"нович", "слаб", "начина", "нул", "basic"},
    "intermediate": {"средн", "уверен", "normal", "medium", "intermediate"},
    "advanced": {"продвин", "сильн", "опыт", "advanced", "expert"},
}

BACKGROUND_ACTIVITY_KEYWORDS = {
    "изуч", "учил", "учила", "учусь", "школ", "вуз", "универ",
    "курс", "сам", "самостоят", "работ", "урок", "лекц", "пара",
}

BACKGROUND_CONTEXT_KEYWORDS = {
    "институт", "универс", "колледж", "школ", "вуз", "универ",
    "курс", "самостоят", "работ", "дома",
}

BACKGROUND_DETAIL_KEYWORDS = {
    "месяц", "месяца", "месяцев", "недел", "неделя", "недели", "год", "года",
    "лет", "день", "дня", "дней", "давно", "раньше", "школе", "вузе",
    "универе", "институте", "курсах", "работе", "самостоятельно", "дома",
}

BACKGROUND_NEGATIVE_KEYWORDS = {
    "не изучал", "не учил", "не учила", "нет", "с нуля", "вообще не",
}

GOAL_KEYWORDS = {
    "экзам", "егэ", "огэ", "зачет", "зачёт", "контрольн", "олимпиад",
    "поступ", "работ", "собесед", "интерес", "для себя", "разобрат",
    "подготов", "учеб", "оценк",
}

VAGUE_OR_PROFANE_KEYWORDS = {
    "хз", "не знаю", "без разницы", "пофиг", "пох", "хуй", "нах", "бля",
    "еб", "fuck", "shit",
}

GENERIC_FOLLOWUP_PATTERNS = (
    "почему",
    "зачем",
    "откуда",
    "объясни",
    "почему так",
    "объясни еще",
    "объясни ещё",
    "еще раз",
    "ещё раз",
    "не понял",
    "не поняла",
    "не понялa",
    "можно пример",
    "приведи пример",
    "что это значит",
    "как это работает",
    "в чем смысл",
    "в чём смысл",
    "в чем разница",
    "в чём разница",
    "откуда это",
    "а если",
    "что если",
)


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _word_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{2,}", text or "")
    ]


def _content_tokens(text: str) -> set[str]:
    return {
        token for token in _word_tokens(text)
        if len(token) >= 3 and token not in STOPWORDS
    }


def is_meaningful_text(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return False

    tokens = _word_tokens(normalized)
    if not tokens:
        return False

    letters = sum(ch.isalpha() for ch in normalized)
    if letters < 2:
        return False

    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    if len(tokens) >= 3 and most_common == len(tokens):
        return False

    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False

    return True


def parse_level_score(text: str) -> int | None:
    normalized = normalize_text(text).casefold()
    match = re.search(r"\b([1-5])\b", normalized)
    if match:
        return int(match.group(1))

    for level, stems in LEVEL_KEYWORDS.items():
        if any(stem in normalized for stem in stems):
            return {"beginner": 1, "intermediate": 3, "advanced": 5}[level]
    return None


def _contains_any_keyword(text: str, keywords: set[str]) -> bool:
    normalized = normalize_text(text).casefold()
    return any(keyword in normalized for keyword in keywords)


def _is_vague_or_profane(text: str) -> bool:
    return _contains_any_keyword(text, VAGUE_OR_PROFANE_KEYWORDS)


def _has_study_background_details(text: str) -> bool:
    if _contains_any_keyword(text, BACKGROUND_NEGATIVE_KEYWORDS):
        return True
    if not is_meaningful_text(text):
        return False

    normalized = normalize_text(text)
    token_count = len(_word_tokens(normalized))
    has_activity = _contains_any_keyword(text, BACKGROUND_ACTIVITY_KEYWORDS)
    has_context = _contains_any_keyword(text, BACKGROUND_CONTEXT_KEYWORDS)
    has_detail = _contains_any_keyword(text, BACKGROUND_DETAIL_KEYWORDS)

    if has_activity and has_detail:
        return True

    return token_count >= 2 and has_context and has_detail


def onboarding_answer_is_valid(question_idx: int, text: str) -> bool:
    if _is_vague_or_profane(text):
        return False
    if question_idx == 0:
        return parse_level_score(text) is not None
    if question_idx == 1:
        return _has_study_background_details(text)
    if question_idx == 2:
        return _contains_any_keyword(text, GOAL_KEYWORDS)
    return is_meaningful_text(text)


def is_generic_followup(question: str) -> bool:
    normalized = normalize_text(question).casefold()
    return any(pattern in normalized for pattern in GENERIC_FOLLOWUP_PATTERNS)


def is_relevant_to_topic(
    question: str,
    topic: str,
    block_context: str,
    scope_texts: list[str] | None = None,
) -> bool:
    if is_generic_followup(question):
        return True

    question_terms = _content_tokens(question)
    if not question_terms:
        return False

    scope = [topic, *(scope_texts or [])]
    for scope_text in scope:
        scope_terms = _content_tokens(scope_text)
        if question_terms & scope_terms:
            return True

    context_terms = _content_tokens(block_context)
    if question_terms & context_terms:
        return True

    return False
