# Роль файла: Проверяет ответы квиза и обновляет прогресс.
from bot.agents.state import State
from bot.disciplines import get_discipline_label
from bot.levels import get_level_label
from bot.progress import (
    PASS_THRESHOLD,
    calculate_xp,
    check_achievements,
    get_course_completion_achievement,
)

LETTER_TO_IDX = {"a": 0, "b": 1, "c": 2, "d": 3,
                 "а": 0, "б": 1, "в": 2, "г": 3}  # поддержка рус. букв


def _build_course_completion_message(
    state: State,
    *,
    xp_total: int,
    xp_gained: int,
    correct_count: int,
    total_questions: int,
    new_achievements: list[str],
) -> str:
    discipline = state.get("discipline", "")
    discipline_label = get_discipline_label(discipline)
    level_label = get_level_label(state.get("level"))
    plan = state.get("plan", [])
    completed_blocks = sum(1 for block in plan if block.get("completed"))
    course_achievement = get_course_completion_achievement(discipline)
    extra_achievements = [item for item in new_achievements if item != course_achievement]

    quiz_line = (
        f"Финальный квиз закрыт идеально: {correct_count}/{total_questions}."
        if total_questions > 0 and correct_count == total_questions
        else f"Финальный квиз закрыт уверенно: {correct_count}/{total_questions}."
    )

    lines = [
        f"🎓 *Курс «{discipline_label}» завершён!*",
        "",
        "Ты дошёл до конца всей дисциплины и закрыл полный учебный маршрут.",
        "Теперь это не просто набор пройденных тем, а уже собранная база, к которой можно возвращаться для повторения и уверенной практики.",
        "",
        f"🏆 Именная ачивка: *{course_achievement}*",
    ]

    if extra_achievements:
        lines.append(f"✨ Дополнительно открыто: {', '.join(extra_achievements)}")

    lines.extend([
        f"⚡ За финальный квиз: +{xp_gained} XP",
        f"⚡ Общий результат по курсу: {xp_total} XP",
        f"📚 Пройдено блоков: {completed_blocks}/{len(plan)}",
        f"🎯 Уровень курса: {level_label}",
        "",
        quiz_line,
        "",
        "Можно открыть другой курс или вернуться к этому плану и повторить ключевые темы.",
    ])
    return "\n".join(lines)


def evaluator(state: State) -> dict:
    """Оценивает результат квиза, начисляет XP, обновляет прогресс."""
    last_message = state.get("last_message", "").lower()
    idx = state.get("current_block_idx", 0)
    plan = state.get("plan", [])
    xp = state.get("xp", 0)
    questions = state.get("quiz_questions", [])
    retry_count = state.get("retry_count", 0)

    # Парсим ответы: "A, B, C" или "a b c" или "1, 2, 3"
    raw_parts = [p.strip().rstrip(".,;") for p in last_message.replace(",", " ").split()]
    answers = []
    for part in raw_parts:
        if part in LETTER_TO_IDX:
            answers.append(LETTER_TO_IDX[part])
        elif part.isdigit():
            answers.append(int(part) - 1)

    correct_answers = [q["answer"] for q in questions] if questions else [0]
    total = len(correct_answers)
    correct_count = sum(1 for a, c in zip(answers, correct_answers) if a == c)
    score = correct_count / total if total > 0 else 0

    xp_gained = calculate_xp(correct_count, total)
    xp += xp_gained

    updated_plan = list(plan)
    if idx < len(updated_plan):
        updated_plan[idx] = {**updated_plan[idx], "completed": score >= PASS_THRESHOLD}

    perfect_quiz = total > 0 and correct_count == total
    streak = correct_count if perfect_quiz else 0
    interim_state = {**state, "plan": updated_plan, "current_block_idx": idx + 1 if score >= PASS_THRESHOLD else idx}
    new_achievements = check_achievements(
        interim_state,
        correct_streak=streak,
        perfect_quiz=perfect_quiz,
    )

    # Разбор ошибок: какие вопросы были неправильными
    labels = ["A", "B", "C", "D"]
    error_lines = []
    for i, (q, correct) in enumerate(zip(questions, correct_answers)):
        user_ans = answers[i] if i < len(answers) else None
        if user_ans != correct:
            q_text = q.get("q", f"Вопрос {i + 1}")
            options = q.get("options", [])
            correct_text = options[correct] if correct < len(options) else "—"
            if user_ans is not None and user_ans < len(options):
                user_text = f"{labels[user_ans]}. {options[user_ans]}"
            else:
                user_text = "нет ответа"
            explanation = q.get("explanation", "")
            line = (
                f"*Вопрос {i + 1}:* {q_text}\n"
                f"✗ Твой ответ: {user_text}\n"
                f"✓ Правильно: {labels[correct]}. {correct_text}"
            )
            if explanation:
                line += f"\n📖 {explanation}"
            error_lines.append(line)

    if score >= PASS_THRESHOLD:
        next_idx = idx + 1
        course_completed = bool(updated_plan) and next_idx >= len(updated_plan)
        if course_completed:
            msg = _build_course_completion_message(
                interim_state,
                xp_total=xp,
                xp_gained=xp_gained,
                correct_count=correct_count,
                total_questions=total,
                new_achievements=new_achievements,
            )
            mode = "idle"
        else:
            if correct_count == total:
                msg = f"✅ Идеально! {correct_count}/{total} — все правильно. +{xp_gained} XP\nВсего XP: {xp}"
            else:
                msg = f"✅ {correct_count}/{total} правильных. +{xp_gained} XP\nВсего XP: {xp}"
                if error_lines:
                    msg += "\n\n*Разбор ошибок:*\n\n" + "\n\n".join(error_lines)
            if new_achievements:
                msg += f"\n\n🏆 Новая ачивка: {', '.join(new_achievements)}!"
            mode = "awaiting_next"
        next_retry_count = 0
    else:
        next_idx = idx
        msg = f"❌ {correct_count}/{total} — нужно повторить. +{xp_gained} XP"
        if error_lines:
            msg += "\n\n*Разбор ошибок:*\n\n" + "\n\n".join(error_lines)
        msg += "\n\nПовторим этот блок с другим примером, а потом попробуем ещё раз."
        mode = "awaiting_retry"
        next_retry_count = retry_count + 1

    return {
        "xp": xp,
        "achievements": new_achievements,
        "plan": updated_plan,
        "current_block_idx": next_idx,
        "retry_count": next_retry_count,
        "response": msg,
        "history": [{"role": "assistant", "content": msg}],
        "mode": mode,
    }
