# Роль файла: Форматирование планов, прогресса и служебного текста.
from bot.agents.state import State
from bot.progress import format_progress_card
from bot.disciplines import get_discipline_label
from bot.levels import get_level_label


def format_progress(state: State) -> str:
    return format_progress_card(state)


def _format_progress_legacy(state: State) -> str:
    xp = state.get("xp", 0)
    achievements = state.get("achievements", [])
    plan = state.get("plan", [])
    level = state.get("level", "—")
    discipline = get_discipline_label(state.get("discipline", "—"))

    completed = sum(1 for b in plan if b.get("completed"))
    total = len(plan)

    lines = [
        f"📊 *Прогресс*",
        f"",
        f"Дисциплина: {discipline}",
        f"Уровень: {level}",
        f"XP: {xp}",
        f"Блоки: {completed}/{total}",
    ]

    if achievements:
        lines.append(f"")
        lines.append(f"🏆 *Ачивки:*")
        for a in achievements:
            lines.append(f"  • {a}")
    else:
        lines.append(f"Ачивок пока нет.")

    return "\n".join(lines)


def format_plan(state: State) -> str:
    plan = state.get("plan", [])
    current_idx = state.get("current_block_idx", 0)
    discipline = get_discipline_label(state.get("discipline", ""))

    if not plan:
        return "План ещё не построен."

    level = state.get("level", "")
    level_label = get_level_label(level) if level else ""

    lines = [f"📋 *Учебный план по «{discipline}»*", ""]
    if level_label:
        lines.append(f"_Адаптировано под уровень: {level_label}_")
        lines.append("")
    for block in plan:
        idx = block["index"]
        topic = block["topic"]
        done = block.get("completed", False)

        if done:
            marker = "✅"
        elif idx == current_idx:
            marker = "▶️"
        else:
            marker = "⬜"

        lines.append(f"{marker} {idx + 1}. Тема блока: {topic}")
        subtopics = [str(item).strip() for item in block.get("subtopics", []) if str(item).strip()]
        if subtopics:
            lines.append("   Учебные вопросы:")
            for subtopic in subtopics:
                lines.append(f"   • {subtopic}")
        lines.append("")

    return "\n".join(lines)
