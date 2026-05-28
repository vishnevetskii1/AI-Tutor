# Роль файла: Собирает учебные блоки по текущей теме.
from bot.agents.state import State
from bot.agents.llm import generate
from bot.agents.response_cleanup import sanitize_teacher_voice
from bot.rag.search import search
from bot.disciplines import get_discipline_label
import re
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Ты — AI-репетитор по математике, настоящий преподаватель-мастер. "
    "Твоя цель — создать исчерпывающий, структурированный и очень понятный учебный блок по заданной теме. "
    "Ты не просто сообщаешь факты, а учишь студента думать, видеть смысл и логику в материале. "
    "Этот учебный блок должен быть самодостаточным и хорошо подходить для того, чтобы другой AI-агент смог на его основе составить проверочный тест.\n\n"
    "### КЛЮЧЕВЫЕ ПРАВИЛА ГЕНЕРАЦИИ ###\n"
    "1. Педагогический подход:\n"
    "- объясняй точно, предметно и без воды;\n"
    "- адаптируй глубину, примеры и язык под уровень целевой аудитории;\n"
    "- не используй шаблонные вступления; заход в тему должен быть естественным и вытекать из её сути;\n"
    "- не начинай учебный блок словами «Представь», «Представим» или «Давай представим».\n"
    "2. Работа с материалом:\n"
    "- если опорный материал предоставлен, он является главным источником истины;\n"
    "- категорически запрещено копировать контекст дословно; перерабатывай его в живое и ясное объяснение;\n"
    "- можно аккуратно дополнять материал базовой интуицией и общепринятыми пояснениями, если они не противоречат опорному тексту и помогают пониманию;\n"
    "- не выдумывай факты и не ссылайся на несуществующие источники.\n"
    "3. Стиль и тон:\n"
    "- говори прямо и уверенно, как преподаватель;\n"
    "- не упоминай «учебник», «опорный материал», «контекст» или «текст выше»;\n"
    "- используй русский язык.\n"
    "4. Формат вывода:\n"
    "- всегда отвечай в Markdown;\n"
    "- строго придерживайся заданной структуры разделов и подзаголовков."
)

LEVEL_HINT = {
    "beginner": (
        "Пользователь новичок. Объясняй медленно и подробно, как для человека без базы. "
        "Каждый новый термин сразу расшифровывай простыми словами. "
        "Не перескакивай через шаги и не сокращай объяснение ради краткости."
    ),
    "intermediate": (
        "Пользователь на среднем уровне. Объясняй структурированно и достаточно подробно, "
        "но без разжёвывания самых базовых вещей. Делай упор на понимание, связи между идеями и типичные переходы от смысла к формуле."
    ),
    "advanced": (
        "Пользователь продвинутый. Можно говорить плотнее, использовать строгие формулировки, "
        "нюансы, ограничения, тонкие различия между похожими понятиями и сложные случаи."
    ),
}

LEVEL_PERSONA = {
    "beginner": (
        "Роль преподавателя: суперучитель для новичка. "
        "Говори живо, по-человечески, тепло и разговорно, но без фамильярности. "
        "Твоя задача — брать сложные или сухие формулировки из материала и переводить их на нормальный понятный русский язык. "
        "Если фраза из книги звучит тяжело, сначала объясни её совсем просто, а уже потом аккуратно дай более точную формулировку. "
        "Примеры делай очень наглядными: можно использовать бытовые, игровые или совсем простые учебные ситуации."
    ),
    "intermediate": (
        "Роль преподавателя: сильный наставник для студента со средней базой. "
        "Говори уверенно, ясно и собранно. "
        "Не разжёвывай очевидное, но обязательно строй мост от интуиции к формуле и обратно к задаче. "
        "Показывай, как идея работает в типичном решении, и где студент обычно теряет смысл за символами."
    ),
    "advanced": (
        "Роль преподавателя: сильный строгий преподаватель для продвинутого студента. "
        "Говори плотно, точно и интеллектуально честно. "
        "Не упрощай там, где важны оговорки. "
        "Подсвечивай границы применимости, тонкие различия между близкими понятиями и нетривиальные следствия."
    ),
}

LEVEL_REQUIREMENTS = {
    "beginner": {
        "focus": (
            "Начни с простой человеческой идеи темы, потом аккуратно перейди к точной формулировке. "
            "Проведи ученика по подпунктам блока так, чтобы каждый следующий абзац опирался на предыдущий. "
            "Обязательно дай один конкретный пример по шагам и отдельно подсвети типичную ошибку новичка. "
            "Разрешено добавлять свои простые жизненные примеры, если они помогают понять материал точнее."
        ),
        "length": "650-950 слов",
        "num_predict": 2200,
        "temperature": 0.18,
    },
    "intermediate": {
        "focus": (
            "Собери связное объяснение: смысл темы, переход от интуиции к формуле, затем применение в задаче. "
            "Подпункты блока раскрой как единую линию рассуждения, а не как разрозненный список. "
            "Отдельно покажи ловушку или место, где студент обычно теряет смысл за символами. "
            "Если уместно, добавь свой свежий пример сверх материала, чтобы сделать идею прозрачнее."
        ),
        "length": "520-760 слов",
        "num_predict": 1700,
        "temperature": 0.18,
    },
    "advanced": {
        "focus": (
            "Начни сразу с содержательной формулировки и строгой идеи темы. "
            "Подчеркни нюансы, границы применимости и то, что часто упрощают слишком грубо. "
            "Добавь нетривиальный пример, тонкий случай или контрпример и отдельно выдели ключевую ловушку. "
            "Разрешено аккуратно добавить свой собственный контрпример или мысленный эксперимент, если он усиливает понимание."
        ),
        "length": "420-620 слов",
        "num_predict": 1450,
        "temperature": 0.18,
    },
}

CONTEXT_SETTINGS = {
    "beginner": {"top_k": 12, "max_chars": 9000},
    "intermediate": {"top_k": 10, "max_chars": 7600},
    "advanced": {"top_k": 9, "max_chars": 6600},
}
LESSON_EMOJI = {
    "beginner": "🌱",
    "intermediate": "📘",
    "advanced": "🧠",
}

TARGET_AUDIENCE = {
    "beginner": "студент-новичок по теме, который путается в базовых понятиях и нуждается в очень ясном, спокойном объяснении",
    "intermediate": "студент со средней базой, которому нужно связать смысл темы с формулами и типовыми задачами",
    "advanced": "студент с сильной базой, которому нужны строгость, тонкие различия и границы применимости идей",
}

SECTION_EMOJI = {
    "краткое введение": "🧭",
    "основная часть": "📚",
    "ключевые выводы": "✅",
    "пример": "🧩",
    "примеры": "🧩",
    "типичная ошибка": "⚠️",
    "типичные ошибки": "⚠️",
}

FINAL_HEADINGS = {"ключевые выводы", "типичная ошибка", "типичные ошибки"}
INTRO_HEADINGS = {"краткое введение"}


def _parse_raw_sections(raw: str) -> list[tuple[str, str]]:
    """Разбивает сырой markdown на секции вида [(heading_line, body_text), ...]."""
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_body: list[str] = []
    for line in raw.split("\n"):
        if re.match(r"^#{1,6}\s", line):
            if current_heading is not None or current_body:
                sections.append((current_heading or "", "\n".join(current_body).strip()))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)
    if current_heading is not None or current_body:
        sections.append((current_heading or "", "\n".join(current_body).strip()))
    return sections


def _heading_text(heading_line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", heading_line).strip()


def _render_section(heading_line: str, body: str) -> str:
    """Собирает одну секцию в формате Telegram Markdown."""
    heading_text = _heading_text(heading_line)
    formatted_heading = _format_section_heading(heading_text)
    body_formatted = _format_lesson_body("", body)
    parts = []
    if formatted_heading:
        parts.append(formatted_heading)
    if body_formatted:
        parts.append(body_formatted)
    return "\n\n".join(parts)


def _build_block_parts(topic: str, level: str, raw: str, footer: str) -> list[str]:
    """
    Разбивает raw блок на части для степпера.

    Часть 0 : заголовок темы + введение + первый основной раздел
    Части 1..N-2 : промежуточные основные разделы
    Часть N-1 : последний основной раздел + финальные секции + footer

    Если основных разделов 0 или 1 — возвращает один элемент (целый блок).
    """
    emoji = LESSON_EMOJI.get(level, "📘")
    sections = _parse_raw_sections(raw)

    intro_secs: list[tuple[str, str]] = []
    main_secs: list[tuple[str, str]] = []
    final_secs: list[tuple[str, str]] = []

    for heading_line, body in sections:
        if not heading_line:
            continue
        text = _heading_text(heading_line)
        norm = text.casefold()
        heading_level = len(re.match(r"^(#+)", heading_line).group(1))

        if heading_level <= 2 or norm == topic.casefold():
            continue  # заголовок темы — добавим сами в шапку
        elif norm in INTRO_HEADINGS:
            intro_secs.append((heading_line, body))
        elif norm in FINAL_HEADINGS:
            final_secs.append((heading_line, body))
        else:
            main_secs.append((heading_line, body))

    if len(main_secs) <= 1:
        # Нет разделов или один — один монолитный блок
        return [_render_lesson(topic, level, raw) + "\n\n" + footer]

    result: list[str] = []

    # Часть 0: заголовок темы, введение и первый основной раздел.
    part0: list[str] = [f"{emoji} *{topic}*"]
    for sec in intro_secs:
        part0.extend(["", _render_section(*sec)])
    part0.extend(["", _render_section(*main_secs[0])])
    result.append("\n".join(part0).strip())

    # Промежуточные части нужны только когда основных разделов больше двух.
    for sec in main_secs[1:-1]:
        result.append(_render_section(*sec))

    # В последнюю часть складываем последний раздел, выводы и footer.
    last: list[str] = [_render_section(*main_secs[-1])]
    for sec in final_secs:
        last.extend(["", _render_section(*sec)])
    last.extend(["", footer])
    result.append("\n".join(last).strip())

    return result


def _trim_context(chunks: list[str], max_chars: int = 3200) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        cleaned = (chunk or "").strip()
        if not cleaned:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(cleaned) > remaining:
            cleaned = cleaned[:remaining].rstrip() + "..."
        parts.append(cleaned)
        total += len(cleaned) + 2
    return "\n\n".join(parts)


def _safe_text(value: object, fallback: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text or fallback


def _format_section_heading(heading: str) -> str:
    clean_heading = heading.strip(" *_#").strip()
    if not clean_heading:
        return ""
    emoji = SECTION_EMOJI.get(clean_heading.casefold(), "🔹")
    return f"{emoji} *{clean_heading}*"


def _format_lesson_body(topic: str, raw_text: str) -> str:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    lines: list[str] = []
    skipped_title = False

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            heading = heading_match.group(2).strip()
            normalized_heading = heading.strip(" *_#").strip()
            if not skipped_title and normalized_heading.casefold() == topic.casefold():
                skipped_title = True
                continue
            rendered_heading = _format_section_heading(normalized_heading)
            if rendered_heading:
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(rendered_heading)
                lines.append("")
            continue

        compact = re.sub(r"\s+", " ", line)
        compact = re.sub(r"^[-*]\s+", "• ", compact)
        lines.append(compact)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def _render_lesson(topic: str, level: str, raw_text: str) -> str:
    emoji = LESSON_EMOJI.get(level, "📘")
    body = _format_lesson_body(topic, raw_text)
    if not body:
        body = (
            "Разберём тему как преподаватель: что здесь означает каждый ключевой шаг, "
            "как это увидеть в задаче и где чаще всего ошибаются."
        )
    return f"{emoji} *{topic}*\n\n{body}"


def _target_audience_text(level: str, discipline_label: str) -> str:
    audience = TARGET_AUDIENCE.get(level, TARGET_AUDIENCE["beginner"])
    return f"{audience}. Дисциплина: {discipline_label}."


def content_agent(state: State) -> dict:
    plan = state.get("plan", [])
    idx = state.get("current_block_idx", 0)
    discipline = state.get("discipline", "")
    level = state.get("level", "beginner")
    lesson_profile = LEVEL_REQUIREMENTS.get(level, LEVEL_REQUIREMENTS["beginner"])
    context_profile = CONTEXT_SETTINGS.get(level, CONTEXT_SETTINGS["intermediate"])
    retry_count = state.get("retry_count", 0)
    discipline_label = get_discipline_label(discipline)

    if idx >= len(plan):
        msg = (
            f"🎓 *Курс «{discipline_label}» уже завершён.*\n\n"
            "Основной маршрут по этой дисциплине закрыт. Можно открыть другой курс "
            "или вернуться к плану и повторить нужные темы."
        )
        return {"response": msg, "mode": "idle"}

    block = plan[idx]
    topic = block["topic"]
    block_goal = _safe_text(
        block.get("goal", ""),
        f"Понять тему «{topic}» и научиться уверенно использовать её в задачах.",
    )
    block_subtopics = [
        _safe_text(item, "")
        for item in block.get("subtopics", [])
        if _safe_text(item, "")
    ][:3]
    if not block_subtopics:
        block_subtopics = [
            f"Смысл темы «{topic}»",
            f"Ключевые формулировки по теме «{topic}»",
            f"Применение темы «{topic}» в задачах",
        ]

    previous_topic = ""
    if idx > 0 and idx - 1 < len(plan):
        previous_topic = _safe_text(plan[idx - 1].get("topic", ""), "")

    search_query = " ".join([discipline_label, topic, block_goal, *block_subtopics])
    chunks = search(search_query, discipline, top_k=context_profile["top_k"])
    context = _trim_context(chunks, max_chars=context_profile["max_chars"]) if chunks else ""
    subtopics_text = "\n".join(f"- {item}" for item in block_subtopics)
    target_audience = _target_audience_text(level, discipline_label)

    context_section = (
        context if context else "Опорный материал не предоставлен."
    )
    retry_section = ""
    if retry_count > 0:
        retry_section = (
            "Это повторное объяснение после неудачного квиза.\n"
            "Сделай блок заметно проще, подробнее и спокойнее, чем в прошлый раз.\n"
            "Обязательно начни с самой базы, не пропускай шаги, используй другой пример и яснее подсвети место, где чаще всего ошибаются.\n\n"
        )
    prompt = (
        f"{LEVEL_HINT.get(level, '')}\n"
        f"{LEVEL_PERSONA.get(level, '')}\n\n"
        "### КОНТЕКСТ ДЛЯ РАБОТЫ ###\n"
        f"- Тема: {topic}\n"
        f"- Целевая аудитория: {target_audience}\n"
        f"- Дисциплина: {discipline_label}\n"
        f"- Цель блока: {block_goal}\n"
        f"- Подпункты, которые обязательно нужно раскрыть:\n{subtopics_text}\n"
        f"- Номер блока в курсе: {idx + 1}\n"
        f"- Предыдущая тема: {previous_topic or 'это первый блок'}\n"
        f"- Опорный материал:\n{context_section}\n\n"
        f"{retry_section}"
        "### ДОПОЛНИТЕЛЬНАЯ АДАПТАЦИЯ ПОД УРОВЕНЬ ###\n"
        f"{lesson_profile['focus']}\n\n"
        "### СТРУКТУРА И ФОРМАТ ВЫВОДА (ОБЯЗАТЕЛЬНО) ###\n"
        "Всегда придерживайся этой Markdown-структуры:\n\n"
        f"## {topic}\n\n"
        "### Краткое введение\n"
        "1-2 абзаца: о чём пойдёт речь, почему это важно и как это связано с текущей темой без шаблонных заходов.\n\n"
        "### [Название подраздела 1]\n"
        "Первый логический раздел основной части. Используй ### (три решётки) — не ####.\n\n"
        "### [Название подраздела 2]\n"
        "Второй логический раздел. Можно добавить ### [подраздел 3] и ### [подраздел 4] по необходимости.\n\n"
        "Для каждого подраздела:\n"
        "- используй маркированные списки для перечислений;\n"
        "- выделяй ключевые термины, формулы и определения жирным;\n"
        "- приводи простые и понятные примеры;\n"
        "- раскрывай цель блока и все подпункты блока содержательно.\n\n"
        "### Ключевые выводы\n"
        "В конце дай 2-3 пункта, которые студент должен запомнить.\n\n"
        "### ЖЁСТКИЕ ОГРАНИЧЕНИЯ ###\n"
        "- не начинай текст словами «Представь», «Представим» или «Давай представим»;\n"
        "- не упоминай учебник, контекст, опорный материал или текст выше;\n"
        "- не уходи в соседние темы;\n"
        "- не копируй опорный материал дословно;\n"
        "- не пиши пустую мотивационную воду;\n"
        "- не используй LaTeX и долларовые знаки;\n"
        f"- ориентир по объёму: {lesson_profile['length']}."
    )

    raw = generate(
        prompt,
        SYSTEM_PROMPT,
        label="content_agent",
        options={
            "num_predict": lesson_profile["num_predict"],
            "temperature": lesson_profile["temperature"],
        },
    )
    footer = (
        "Если хочешь разобрать этот блок в формате диалога, нажми кнопку «💬 Спросить по блоку».\n\n"
        "Готов к квизу? Нажми кнопку ниже."
    )
    block_parts = _build_block_parts(topic, level, raw, footer)
    block_parts = [sanitize_teacher_voice(p) for p in block_parts]

    # Полный текст блока нужен для квиза и кнопки "Вернуться к блоку"
    full_lesson = _render_lesson(topic, level, raw)
    full_msg = sanitize_teacher_voice(f"{full_lesson}\n\n{footer}")
    study_context = f"{context}\n\n{full_msg}" if context else full_msg

    return {
        "response": block_parts[0],
        "history": [{"role": "assistant", "content": full_msg}],
        "mode": "quiz",
        "last_block_content": study_context,
        "last_block_quiz_context": full_msg,
        "retry_count": retry_count,
        "block_parts": block_parts,
        "current_subtopic_idx": 0,
    }
