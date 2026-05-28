# Роль файла: Генерирует вопросы квиза по текущему блоку.
import json

from bot.agents.state import State
from bot.agents.llm import generate
from bot.agents.generation_utils import ensure_quiz_questions, parse_json_array


SYSTEM_PROMPT = """### РОЛЬ ###
Ты — AI-методист, эксперт по созданию проверочных материалов. Твоя задача — превращать учебный текст в эффективный тест для проверки знаний. Ты работаешь как саб-агент в системе AI-репетитора.

### КОНТЕКСТ ###
Ты получаешь готовый учебный блок от другого агента. Твоя единственная цель — на основе этого текста создать квиз с вопросами и вариантами ответов. Студент только что прочитал этот текст и теперь должен проверить, как он его понял.

### ЗАДАЧА ###
Проанализируй текст, который будет предоставлен ниже в блоке [УЧЕБНЫЙ_БЛОК], и сгенерируй на его основе квиз из 3 вопросов.

### КЛЮЧЕВЫЕ ПРАВИЛА И ТРЕБОВАНИЯ ###
1. СТРОГАЯ ПРИВЯЗКА К ИСТОЧНИКУ:
- Все вопросы, правильные ответы и неверные варианты должны быть сформулированы исключительно на основе информации из предоставленного текста [УЧЕБНЫЙ_БЛОК].
- Не используй никакие внешние знания.
- Ответ на каждый вопрос должен прямо или косвенно содержаться в тексте.

2. КАЧЕСТВО ВОПРОСОВ:
- Вопросы должны проверять понимание концепций, причинно-следственных связей, определений и ключевых идей, а не простое запоминание отдельных фраз.
- Формулируй вопросы так, как это сделал бы опытный преподаватель: чётко, однозначно и по существу.

3. КАЧЕСТВО ВАРИАНТОВ ОТВЕТОВ:
- Среди вариантов должен быть только один однозначно правильный ответ согласно тексту.
- Неправильные варианты должны быть правдоподобными и тематически близкими, но неверными в контексте предоставленного материала.
- Не создавай очевидно глупых, абсурдных или заведомо мусорных вариантов ответа.

### ФОРМАТ ВЫВОДА ###
Верни только валидный JSON без пояснений и без markdown-блоков."""

QUIZ_PROMPT = """[УЧЕБНЫЙ_БЛОК]
{context}
[/УЧЕБНЫЙ_БЛОК]

Сгенерируй квиз из 3 вопросов по этому учебному блоку.

Верни JSON строго в таком формате:
[
  {{
    "q": "текст вопроса",
    "options": ["вариант A", "вариант B", "вариант C", "вариант D"],
    "answer": 0,
    "explanation": "1-2 предложения: почему этот ответ правильный и в чём суть"
  }}
]

Требования к формату:
- ровно 3 вопроса;
- у каждого вопроса ровно 4 варианта ответа;
- поле "answer" — индекс правильного ответа от 0 до 3;
- поле "explanation" — краткое объяснение правильного ответа (не пересказывай весь блок, только суть);
- без пояснений до или после JSON."""

MAX_CONTEXT_CHARS = 3200


def _trim_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _normalize_quiz_questions(raw_questions: list, target_count: int = 3) -> list[dict]:
    questions: list[dict] = []
    seen_questions: set[str] = set()

    for item in raw_questions:
        if not isinstance(item, dict):
            continue

        question = " ".join(str(item.get("q", "")).split()).strip()
        options = item.get("options")
        answer = item.get("answer")

        if not question or not isinstance(options, list) or not isinstance(answer, int):
            continue
        if answer < 0 or answer > 3:
            continue

        clean_options: list[str] = []
        seen_options: set[str] = set()
        for option in options[:4]:
            value = " ".join(str(option).split()).strip()
            key = value.casefold()
            if not value or key in seen_options:
                continue
            seen_options.add(key)
            clean_options.append(value)

        if len(clean_options) != 4:
            continue

        question_key = question.casefold()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)

        normalized = {
            "q": question,
            "options": clean_options,
            "answer": answer,
        }
        explanation = " ".join(str(item.get("explanation", "")).split()).strip()
        if explanation:
            normalized["explanation"] = explanation
        source_fact = " ".join(str(item.get("source_fact", "")).split()).strip()
        if source_fact:
            normalized["source_fact"] = source_fact

        questions.append(normalized)
        if len(questions) == target_count:
            return questions

    return questions


def _format_quiz_question(topic: str, questions: list[dict], index: int) -> str:
    question = questions[index]
    options_text = "\n".join(
        f"{['A', 'B', 'C', 'D'][option_idx] if option_idx < 4 else option_idx + 1}. {option}"
        for option_idx, option in enumerate(question["options"])
    )
    return (
        f"📝 *Квиз по теме «{topic}»*\n\n"
        f"*Вопрос {index + 1} из {len(questions)}*\n"
        f"{question['q']}\n\n"
        f"_Выбери один вариант ответа из {len(question['options'])}._\n\n"
        f"{options_text}"
    )


def quiz_agent(state: State) -> dict:
    plan = state.get("plan", [])
    idx = state.get("current_block_idx", 0)
    topic = plan[idx]["topic"] if idx < len(plan) else "тема"
    quiz_source = state.get("last_block_quiz_context") or state.get("response") or state.get("last_block_content", "")
    context = _trim_context(quiz_source)

    raw = generate(
        QUIZ_PROMPT.format(context=context),
        SYSTEM_PROMPT,
        label="quiz_agent",
        options={"num_predict": 520, "temperature": 0.2},
    ).strip()

    try:
        questions_raw = parse_json_array(raw)
    except json.JSONDecodeError:
        questions_raw = []

    questions = _normalize_quiz_questions(questions_raw)
    if len(questions) < 3:
        questions = ensure_quiz_questions(questions_raw, topic, source_text=context)

    msg = _format_quiz_question(topic, questions, 0)

    return {
        "response": msg,
        "history": [{"role": "assistant", "content": msg}],
        "mode": "quiz_active",
        "quiz_questions": questions,
        "quiz_current_idx": 0,
        "quiz_answers": [],
    }
