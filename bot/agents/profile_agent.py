# Роль файла: Обрабатывает онбординг и определяет уровень пользователя.
from bot.agents.state import State
from bot.agents.input_validation import onboarding_answer_is_valid, parse_level_score
from bot.levels import get_level_label

ONBOARDING_QUESTIONS = [
    "Как бы ты оценил свои знания по этой дисциплине? (1 — новичок, 5 — продвинутый)",
    "Изучал ли ты эту тему раньше? Если да — где и как долго?",
    "Какая у тебя цель: сдать экзамен, разобраться для работы или просто интерес?",
]


def profile_agent(state: State) -> dict:
    """Анализирует ответы пользователя и определяет уровень."""
    step = state.get("onboarding_step", 0)
    answers = list(state.get("onboarding_answers", []))
    last_answer = state.get("last_message", "").strip()

    if step < len(ONBOARDING_QUESTIONS) and last_answer:
        if not onboarding_answer_is_valid(step, last_answer):
            retry_q = ONBOARDING_QUESTIONS[step]
            if step == 0:
                msg = (
                    "Не смог понять уровень из этого ответа. "
                    "Напиши число от 1 до 5 или словами: новичок / средний / продвинутый.\n\n"
                    f"{retry_q}"
                )
            elif step == 1:
                msg = (
                    "Нужен более конкретный ответ про прошлый опыт. "
                    "Напиши, изучал ли ты тему раньше и где или как долго: например "
                    "`не изучал`, `2 года в институте`, `в школе 2 месяца`, `самостоятельно неделю`.\n\n"
                    f"{retry_q}"
                )
            elif step == 2:
                msg = (
                    "Нужна понятная цель без сленга и мата. "
                    "Напиши коротко: `экзамен`, `для работы`, `для себя`, `собеседование`.\n\n"
                    f"{retry_q}"
                )
            else:
                msg = (
                    "Ответ получился слишком непонятным. "
                    "Напиши коротко и по делу, чтобы я мог продолжить интервью.\n\n"
                    f"{retry_q}"
                )
            return {
                "response": msg,
                "history": [{"role": "assistant", "content": msg}],
                "mode": "onboarding",
                "onboarding_step": step,
                "onboarding_answers": answers,
            }
        answers.append(last_answer)
        step += 1

    if step < len(ONBOARDING_QUESTIONS):
        next_q = ONBOARDING_QUESTIONS[step]
        return {
            "response": next_q,
            "history": [{"role": "assistant", "content": next_q}],
            "mode": "onboarding",
            "onboarding_step": step,
            "onboarding_answers": answers,
        }

    score = parse_level_score(answers[0]) or 2

    if score <= 2:
        level = "beginner"
    elif score <= 3:
        level = "intermediate"
    else:
        level = "advanced"

    msg = f"Отлично! Определил твой уровень: *{get_level_label(level)}*. Строю план обучения..."
    return {
        "level": level,
        "response": msg,
        "history": [{"role": "assistant", "content": msg}],
        "mode": "planning",
        "onboarding_step": step,
        "onboarding_answers": answers,
    }
