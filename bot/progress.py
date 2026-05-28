# Роль файла: Логика XP, ачивок и сводки прогресса.
import re

from bot.agents.state import State
from bot.disciplines import get_discipline_label
from bot.levels import get_level_label

XP_PER_CORRECT = 10
PASS_THRESHOLD = 0.6
COURSE_ACHIEVEMENT_RE = re.compile(r"^Выпускник курса «(.+)»$")

ALL_ACHIEVEMENTS = {
    "Первый блок": "Завершил первый блок",
    "Пятёрка подряд": "5 правильных ответов подряд без ошибок",
    "Идеальный результат": "100% в квизе",
}


def get_course_completion_achievement(discipline: str) -> str:
    return f"Выпускник курса «{get_discipline_label(discipline)}»"


def describe_achievement(name: str) -> str:
    static_description = ALL_ACHIEVEMENTS.get(name)
    if static_description:
        return static_description

    match = COURSE_ACHIEVEMENT_RE.match(name)
    if match:
        return f"Полностью завершил дисциплину «{match.group(1)}»."

    if name == "Дисциплина завершена":
        return "Прошёл все блоки дисциплины."

    return ""


def build_progress_snapshot(state: State) -> dict:
    plan = state.get("plan", [])
    completed = sum(1 for b in plan if b.get("completed"))
    total = len(plan)
    current_idx = state.get("current_block_idx", 0)
    percent = round((completed / total) * 100) if total else 0

    if total == 0:
        status = "Онбординг и план ещё впереди"
        next_topic = ""
    elif current_idx < total:
        next_topic = plan[current_idx]["topic"]
        status = f"Остановился на: {next_topic}"
    else:
        next_topic = ""
        status = "✅ Курс завершён"

    return {
        "discipline": state.get("discipline", ""),
        "discipline_label": get_discipline_label(state.get("discipline", "—")),
        "level": get_level_label(state.get("level")),
        "xp": state.get("xp", 0),
        "completed": completed,
        "total": total,
        "percent": percent,
        "current_idx": current_idx,
        "next_topic": next_topic,
        "status": status,
    }


def calculate_xp(correct: int, total: int) -> int:
    """XP за квиз: 10 за каждый правильный ответ."""
    return correct * XP_PER_CORRECT


def check_achievements(state: State, correct_streak: int = 0, perfect_quiz: bool = False) -> list[str]:
    """Возвращает список НОВЫХ ачивок, которые нужно выдать."""
    existing = set(state.get("achievements", []))
    plan = state.get("plan", [])
    new = []

    completed_count = sum(1 for b in plan if b.get("completed"))

    # Первый блок завершён
    if completed_count >= 1 and "Первый блок" not in existing:
        new.append("Первый блок")

    # Пятёрка подряд
    if correct_streak >= 5 and "Пятёрка подряд" not in existing:
        new.append("Пятёрка подряд")

    # Идеальный результат в квизе
    if perfect_quiz and "Идеальный результат" not in existing:
        new.append("Идеальный результат")

    course_achievement = get_course_completion_achievement(state.get("discipline", ""))
    if plan and completed_count == len(plan) and course_achievement not in existing:
        new.append(course_achievement)

    return new


def format_progress_card(state: State) -> str:
    """Карточка прогресса для /прогресс."""
    snapshot = build_progress_snapshot(state)
    xp = snapshot["xp"]
    achievements = state.get("achievements", [])
    level = snapshot["level"]
    discipline = snapshot["discipline_label"]
    user_id = state.get("user_id", "")
    completed = snapshot["completed"]
    total = snapshot["total"]
    current_idx = snapshot["current_idx"]
    plan = state.get("plan", [])

    bar_filled = int((completed / total * 10)) if total > 0 else 0
    bar = "█" * bar_filled + "░" * (10 - bar_filled)

    lines = [
        f"📊 *Прогресс*",
        f"",
        f"👤 ID: {user_id}",
        f"📚 Дисциплина: {discipline}",
        f"🎯 Уровень: {level}",
        f"⚡ XP: {xp}",
        f"",
        f"📈 Блоки: {completed}/{total}",
        f"`[{bar}]`",
    ]

    if total == 0:
        lines.append("📍 Следующий шаг: пройти онбординг и получить план.")
    elif current_idx < total:
        lines.append(f"📍 Следующий блок: {snapshot['next_topic']}")
    else:
        lines.append(f"🎓 Курс «{discipline}» завершён.")
        lines.append("🏁 Маршрут по дисциплине закрыт полностью.")

    if achievements:
        lines.append(f"")
        lines.append(f"🏆 *Ачивки ({len(achievements)}):*")
        for a in achievements:
            desc = describe_achievement(a)
            lines.append(f"  • {a}" + (f" — {desc}" if desc else ""))
    else:
        lines.append(f"")
        lines.append(f"Ачивок пока нет — проходи блоки!")

    return "\n".join(lines)
