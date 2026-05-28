# Роль файла: Отвечает на уточняющие вопросы по текущему уроку.
from bot.agents.state import State
from bot.agents.input_validation import is_meaningful_text, is_relevant_to_topic
from bot.agents.llm import generate
from bot.agents.response_cleanup import sanitize_teacher_voice
from bot.rag.search import search
from bot.topic_web_search import search_topic_web


SYSTEM_PROMPT = (
    "Ты персональный AI-тьютор, закреплённый за одним конкретным учебным блоком. "
    "Твоя цель — вести живой учебный диалог и помогать студенту глубоко разобраться именно в текущей теме. "
    "Ты не просто отвечаешь на вопросы, а ведёшь студента к пониманию как сильный репетитор один на один.\n\n"
    "Главные правила:\n"
    "1. Отвечай только в рамках текущей темы и подпунктов. Если вопрос не по теме, мягко верни студента к текущей теме.\n"
    "2. Если вопрос показывает пробел в базе или сформулирован неясно, не спеши отвечать в лоб: сначала коротко уточни, что именно студент уже понимает.\n"
    "3. Если студент ошибается, корректируй мягко: без формулировок вроде 'ты не прав', лучше 'почти, но есть нюанс'.\n"
    "4. Если в доступных данных нет точного ответа, не выдумывай. Честно обозначь границу и верни разговор к главной теме.\n"
    "5. Учитывай предыдущие сообщения диалога, чтобы не повторяться и двигаться последовательно.\n"
    "6. Говори как живой преподаватель: прямо, спокойно, уверенно, с поддержкой, но без шаблонных вступлений вроде 'Конечно!' или 'Отличный вопрос!'.\n"
    "7. Не употребляй слова 'учебник', 'материал', 'контекст', 'блок', 'текст выше', 'источник' и не ссылайся на поиск.\n"
    "8. Отвечай на русском языке.\n\n"
    "Формат ответа:\n"
    "- обычно 2-5 предложений;\n"
    "- если тема требует развёрнутого объяснения, используй короткий список, но не превращай ответ в полотно;\n"
    "- выделяй ключевые термины жирным;\n"
    "- по возможности завершай ответ коротким вопросом или предложением следующего шага, чтобы поддержать диалог."
)


LEVEL_GUIDANCE = {
    "beginner": (
        "Студенту тема даётся тяжело. Объясняй максимально просто, без перегруза терминами, "
        "через бытовые аналогии и очень короткие шаги."
    ),
    "intermediate": (
        "Студент понимает базу. Объясняй уверенно, связывай смысл с формулой и показывай, "
        "как идея работает на типичном примере."
    ),
    "advanced": (
        "Студент сильный. Отвечай более строго, подчёркивай нюансы, границы применимости и типичные ловушки."
    ),
}


def _recent_dialogue(history: list[dict], question: str) -> str:
    if not history:
        return "Предыдущих сообщений почти нет."

    recent_messages = history[-6:]
    lines: list[str] = []
    for item in recent_messages:
        role = "Студент" if item.get("role") == "user" else "Тьютор"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    if not lines:
        return "Предыдущих сообщений почти нет."

    if not any(line == f"Студент: {question.strip()}" for line in lines) and question.strip():
        lines.append(f"Студент: {question.strip()}")

    return "\n".join(lines)


def qa_agent(state: State) -> dict:
    question = state.get("last_message", "")
    discipline = state.get("discipline", "")
    plan = state.get("plan", [])
    idx = state.get("current_block_idx", 0)
    block = plan[idx] if idx < len(plan) else {}
    topic = block.get("topic", "")
    block_subtopics = [
        str(item).strip()
        for item in block.get("subtopics", [])
        if str(item).strip()
    ]
    level = state.get("level", "beginner")
    mode = state.get("mode", "")
    history = state.get("history", [])

    block_context = state.get("last_block_content", "")

    if not question.strip():
        intro = (
            f"Здравствуйте! Я тьютор по теме «{topic}». 👋\n\n"
            "Задайте мне любые вопросы по теме учебного блока и я помогу Вам разобраться с этим! 🎓"
        )
        return {
            "response": intro,
            "history": [{"role": "assistant", "content": intro}],
            "mode": "lesson_chat",
        }

    if not is_meaningful_text(question):
        msg = (
            "Не смог понять этот текст. "
            f"Пиши, пожалуйста, осмысленный вопрос по текущей теме «{topic}»."
        )
        return {
            "response": msg,
            "history": [{"role": "assistant", "content": msg}],
            "mode": "lesson_chat" if mode == "lesson_chat" else "learning",
        }

    if not is_relevant_to_topic(question, topic, block_context, scope_texts=block_subtopics):
        subtopics_hint = f" и его подпунктам: {', '.join(block_subtopics)}" if block_subtopics else ""
        msg = (
            f"Сейчас разбираем тему «{topic}». "
            f"Задай, пожалуйста, вопрос только по текущему блоку{subtopics_hint}."
        )
        return {
            "response": msg,
            "history": [{"role": "assistant", "content": msg}],
            "mode": "lesson_chat" if mode == "lesson_chat" else "learning",
        }

    # Сначала опираемся на уже показанный блок и локальный RAG по текущей теме.
    chunks = search(f"{question} {topic}", discipline, top_k=3)
    context_parts: list[str] = []
    if block_context:
        context_parts.append(block_context)
    if chunks:
        context_parts.append("\n\n".join(chunks))

    web_chunks = search_topic_web(
        question,
        topic,
        discipline=discipline,
        subtopics=block_subtopics,
    ) if mode == "lesson_chat" and not context_parts else []

    if web_chunks:
        context_parts.append("Дополнительные пояснения по теме:\n" + "\n".join(f"- {item}" for item in web_chunks))
    context = "\n\n".join(part for part in context_parts if part)
    level_guidance = LEVEL_GUIDANCE.get(level, LEVEL_GUIDANCE["beginner"])
    dialogue_context = _recent_dialogue(history, question)

    context_section = (
        f"Опора по теме (используй молча, без ссылок на происхождение):\n{context}\n\n"
        if context
        else "Опоры по теме мало. Ответь аккуратно и не делай вид, что знаешь больше, чем действительно знаешь.\n\n"
    )
    prompt = (
        f"Тема: {topic}\n"
        f"Уровень студента: {level}\n"
        f"Подсказка по уровню: {level_guidance}\n"
        f"Текущие подтемы: {', '.join(block_subtopics) if block_subtopics else 'без уточняющих подпунктов'}\n\n"
        f"Последние реплики диалога:\n{dialogue_context}\n\n"
        f"{context_section}"
        f"Последний вопрос студента: {question}\n\n"
        "Как отвечать:\n"
        "- сначала определи, можно ли ответить сразу или нужно уточнить пробел в понимании;\n"
        "- если вопрос неясный, задай один короткий уточняющий вопрос вместо длинного ответа;\n"
        "- если вопрос основан на ошибке, мягко поправь и затем объясни верный ход мысли;\n"
        "- если вопрос по смежной теме, дай 1-2 предложения только в объёме, нужном для текущей темы, и сразу верни фокус назад;\n"
        "- если точного ответа нет, честно скажи об этом без догадок и верни к главному;\n"
        "- не расширяй ответ на соседние темы, следующие блоки и другие разделы дисциплины.\n\n"
        "Требования к оформлению:\n"
        "- выдели ключевые понятия *жирным*;\n"
        "- сделай ответ визуально лёгким для Telegram: короткие абзацы и короткие списки, только если они реально помогают;\n"
        "- не начинай каждый ответ с дежурной похвалы или пустого вступления;\n"
        "- не упоминай слова из запретного списка из системной инструкции;\n"
        "- по возможности закончи ответ коротким вопросом или предложением разобрать пример.\n"
    )

    msg = generate(prompt, SYSTEM_PROMPT, label="qa_agent")
    msg = sanitize_teacher_voice(msg)

    return {
        "response": msg,
        "history": [{"role": "assistant", "content": msg}],
        "mode": "lesson_chat" if mode == "lesson_chat" else "learning",
    }
