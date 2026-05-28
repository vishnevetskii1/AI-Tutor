# Роль файла: SQLite-запросы для админ-отчётов и списков пользователей.
import sqlite3

from bot.admin_controls import get_maintenance_status, is_user_blocked
from bot.agents.graph import build_graph
from bot.disciplines import get_discipline_label
from bot.levels import get_level_label


def list_recent_thread_ids(db_path: str, limit: int = 20) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT thread_id, MAX(rowid) AS last_rowid
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_rowid DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _base_user_id(thread_id: str) -> str:
    return str(thread_id).split(":", 1)[0]


def list_recent_user_ids(db_path: str, limit: int = 20) -> list[str]:
    user_ids: list[str] = []
    seen: set[str] = set()
    for thread_id in list_recent_thread_ids(db_path, limit=limit * 5):
        user_id = _base_user_id(thread_id)
        if user_id in seen:
            continue
        seen.add(user_id)
        user_ids.append(user_id)
        if len(user_ids) >= limit:
            break
    return user_ids


def list_user_thread_ids(db_path: str, user_id: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT thread_id, MAX(rowid) AS last_rowid
            FROM checkpoints
            WHERE thread_id = ? OR thread_id LIKE ?
            GROUP BY thread_id
            ORDER BY last_rowid DESC
            """,
            (user_id, f"{user_id}:%"),
        )
        return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _get_state(graph, thread_id: str) -> dict:
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values if snapshot and snapshot.values else {}


def summarize_state_fields(state: dict) -> dict:
    plan = state.get("plan", [])
    completed = sum(1 for block in plan if block.get("completed"))
    total = len(plan)
    current_idx = state.get("current_block_idx", 0)
    current_topic = "—"
    if current_idx < len(plan):
        current_topic = plan[current_idx].get("topic", "—")
    return {
        "discipline": get_discipline_label(state.get("discipline", "—")),
        "level": get_level_label(state.get("level")),
        "xp": state.get("xp", 0),
        "mode": state.get("mode", "—"),
        "completed": completed,
        "total": total,
        "current_topic": current_topic,
        "achievements": state.get("achievements", []),
        "onboarding_step": state.get("onboarding_step", 0),
        "quiz_questions": len(state.get("quiz_questions", [])),
        "quiz_answers": len(state.get("quiz_answers", [])),
    }


def _summarize_state(thread_id: str, state: dict) -> str:
    summary = summarize_state_fields(state)
    return (
        f"👤 `{thread_id}`\n"
        f"📚 {summary['discipline']} | 🎯 {summary['level']} | ⚡ XP {summary['xp']} | "
        f"📈 {summary['completed']}/{summary['total']} | 🧭 {summary['mode']}"
    )


def list_recent_users(db_path: str, limit: int = 20) -> list[dict]:
    user_ids = list_recent_user_ids(db_path, limit=limit)
    if not user_ids:
        return []

    graph = build_graph(db_path)
    users = []
    for user_id in user_ids:
        thread_ids = list_user_thread_ids(db_path, user_id)
        states = []
        for thread_id in thread_ids:
            state = _get_state(graph, thread_id)
            if state:
                states.append({
                    "thread_id": thread_id,
                    "state": state,
                    "summary": summarize_state_fields(state),
                })
        total_xp = sum(item["summary"]["xp"] for item in states)
        blocked = is_user_blocked(db_path, int(user_id)) if str(user_id).isdigit() else False
        users.append({
            "user_id": user_id,
            "thread_ids": thread_ids,
            "courses": states,
            "summary": {
                "courses_count": len(states),
                "total_xp": total_xp,
                "blocked": blocked,
            },
            "line": (
                f"👤 `{user_id}`\n"
                f"📚 Курсов: {len(states)} | ⚡ XP: {total_xp} | {'⛔ Выключен' if blocked else '✅ Активен'}"
            ),
        })
    return users


def build_db_users_report(db_path: str, limit: int = 20) -> str:
    users = list_recent_users(db_path, limit=limit)
    if not users:
        return "Пользователей в базе пока нет."

    lines = [f"👥 *Пользователи в базе* ({len(users)})", ""]
    for user in users:
        lines.append(user["line"])
        lines.append("")
    lines.append("Открыть карточку: `/dbuser <telegram_user_id>`")
    return "\n".join(lines).strip()


def build_db_user_report(db_path: str, user_id: str) -> str:
    graph = build_graph(db_path)
    thread_ids = list_user_thread_ids(db_path, user_id)
    if not thread_ids:
        return f"Пользователь `{user_id}` не найден в базе."

    states = []
    for thread_id in thread_ids:
        state = _get_state(graph, thread_id)
        if not state:
            continue
        states.append((thread_id, summarize_state_fields(state)))

    if not states:
        return f"Пользователь `{user_id}` не найден в базе."

    total_xp = sum(summary["xp"] for _, summary in states)
    blocked = is_user_blocked(db_path, int(user_id)) if str(user_id).isdigit() else False
    lines = [
        f"👤 *Пользователь* `{user_id}`",
        "",
        f"Статус доступа: {'⛔ отключён' if blocked else '✅ активен'}",
        f"📚 Активных курсов в базе: {len(states)}",
        f"⚡ Суммарный XP: {total_xp}",
        "",
    ]
    for index, (thread_id, summary) in enumerate(states, start=1):
        lines.extend([
            f"*Курс {index}: {summary['discipline']}*",
            f"🆔 Thread: `{thread_id}`",
            f"🎯 Уровень: {summary['level']}",
            f"⚡ XP: {summary['xp']}",
            f"🧭 Режим: {summary['mode']}",
            f"📈 Блоки: {summary['completed']}/{summary['total']}",
            f"📍 Текущий блок: {summary['current_topic']}",
            f"📝 Онбординг шаг: {summary['onboarding_step']}",
            f"❓ Вопросов в квизе: {summary['quiz_questions']}",
            f"✅ Ответов в квизе: {summary['quiz_answers']}",
            f"🏆 Ачивки: {', '.join(summary['achievements']) if summary['achievements'] else 'нет'}",
            "",
        ])
    return "\n".join(lines)


def build_admin_status_text(db_path: str) -> str:
    maintenance = get_maintenance_status(db_path)
    if maintenance["enabled"]:
        return "🛠 Техработы: включены"
    return "✅ Техработы: выключены"
