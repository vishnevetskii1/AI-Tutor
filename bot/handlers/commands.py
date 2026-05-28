# Роль файла: Обработчики slash-команд: start, reset, progress и других.
import asyncio
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.access_control import get_access_denial_message, is_admin_user
from bot.admin_controls import get_maintenance_status
from bot.admin_db import build_admin_status_text, build_db_user_report, build_db_users_report, list_recent_users
from bot.graph_runner import (
    get_active_discipline,
    get_user_state,
    is_user_busy,
    invoke_graph_async,
    list_user_course_states,
    reset_user,
)
from bot.formatting import format_progress, format_plan
from bot.config import load_config
from bot.disciplines import get_discipline_description, get_discipline_label
from bot.progress import build_progress_snapshot


def _is_admin_user(user_id: int) -> bool:
    return is_admin_user(user_id)


async def _deny_if_restricted(update: Update) -> bool:
    denial = get_access_denial_message(update.effective_user.id)
    if not denial:
        return False
    await update.message.reply_text(denial)
    return True


def _get_disciplines() -> list[str]:
    config = load_config()
    disciplines_dir = Path(config.disciplines_dir)
    if not disciplines_dir.exists():
        return []
    return [d.name for d in disciplines_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]


def _build_discipline_markup(disciplines: list[str]) -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(get_discipline_label(d), callback_data=d)] for d in disciplines]
    return InlineKeyboardMarkup(keyboard)


def _build_discipline_prompt(first_name: str, disciplines: list[str], intro: str | None = None) -> tuple[str, InlineKeyboardMarkup | None]:
    if not disciplines:
        text = (
            f"Привет, {first_name}! 👋\n\n"
            "Пока нет загруженных дисциплин. Администратор должен добавить PDF-учебники."
        )
        return text, None

    lead = intro or f"Привет, {first_name}! 👋\n\nЯ твой AI-репетитор."
    descriptions = [
        f"• *{get_discipline_label(d)}* — {get_discipline_description(d)}"
        for d in disciplines
        if get_discipline_description(d)
    ]
    description_block = ""
    if descriptions:
        description_block = "\n\n*Что изучаем:*\n" + "\n".join(descriptions)
    text = f"{lead}\nВыбери дисциплину для обучения:{description_block}"
    return text, _build_discipline_markup(disciplines)


def _build_course_browser_markup(
    disciplines: list[str],
    started: dict[str, dict],
    active_discipline: str | None,
) -> InlineKeyboardMarkup:
    keyboard = []
    for discipline in disciplines:
        label = get_discipline_label(discipline)
        summary = started.get(discipline)
        if summary:
            prefix = "▶️" if discipline == active_discipline else "📘"
            button_text = f"{prefix} {label} · {summary['percent']}% · {summary['xp']} XP"
        else:
            button_text = f"✨ {label} · начать"
        keyboard.append([InlineKeyboardButton(button_text[:64], callback_data=f"course:{discipline}")])
    return InlineKeyboardMarkup(keyboard)


def _build_course_browser_prompt(
    first_name: str,
    disciplines: list[str],
    started_states: list[dict],
    active_discipline: str | None,
    intro: str | None = None,
) -> tuple[str, InlineKeyboardMarkup | None]:
    if not disciplines:
        return _build_discipline_prompt(first_name, disciplines, intro=intro)

    started = {
        state["discipline"]: build_progress_snapshot(state)
        for state in started_states
        if state.get("discipline")
    }
    lead = intro or (
        f"Привет, {first_name}! 👋\n\n"
        "Вот твои курсы. Можно продолжать начатые и параллельно заходить в новые."
    )
    lines = [lead]

    if started:
        lines.append("\n*Текущий прогресс:*")
        for discipline in disciplines:
            summary = started.get(discipline)
            if not summary:
                continue
            active_mark = " • активный" if discipline == active_discipline else ""
            level = summary["level"] if summary["level"] else "не определён"
            lines.append(
                f"• *{summary['discipline_label']}*{active_mark} — {level}, "
                f"{summary['xp']} XP, {summary['percent']}%, {summary['status']}"
            )
    else:
        lines.append("\nПока ни один курс не начат.")

    fresh_disciplines = [discipline for discipline in disciplines if discipline not in started]
    if fresh_disciplines:
        lines.append("\n*Новые курсы:*")
        for discipline in fresh_disciplines:
            description = get_discipline_description(discipline)
            suffix = f" — {description}" if description else ""
            lines.append(f"• *{get_discipline_label(discipline)}*{suffix}")

    return "\n".join(lines), _build_course_browser_markup(disciplines, started, active_discipline)


def build_admin_panel_text(status_line: str) -> str:
    return (
        "👥 *Пользователи в базе*\n"
        f"{status_line}\n\n"
        "Выбери пользователя ниже, чтобы открыть его карточку."
    )


def build_admin_panel_markup(users: list[dict], maintenance_enabled: bool) -> InlineKeyboardMarkup:
    rows = []
    toggle_label = "🛠 Выключить техработы" if maintenance_enabled else "🛠 Включить техработы"
    rows.append([InlineKeyboardButton(toggle_label, callback_data="admin_maintenance_toggle")])
    for user in users[:10]:
        summary = user["summary"]
        access_icon = "⛔" if summary.get("blocked") else "✅"
        label = f"{access_icon} {user['user_id']} · {summary['courses_count']} курсов · {summary['total_xp']} XP"
        rows.append([InlineKeyboardButton(label[:64], callback_data=f"admin_user:{user['user_id']}")])
    rows.append([InlineKeyboardButton("📄 Документация", callback_data="admin_docs")])
    rows.append([InlineKeyboardButton("🔄 Обновить", callback_data="admin_users")])
    rows.append([InlineKeyboardButton("❌ Закрыть", callback_data="admin_close")])
    return InlineKeyboardMarkup(rows)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    first_name = update.effective_user.first_name or "Студент"
    disciplines = _get_disciplines()
    user_id = update.effective_user.id
    started_states = list_user_course_states(user_id)
    active_discipline = get_active_discipline(user_id)
    text, markup = _build_course_browser_prompt(first_name, disciplines, started_states, active_discipline)
    await update.message.reply_text(text, reply_markup=markup, parse_mode="Markdown")


async def cmd_course(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    first_name = update.effective_user.first_name or "Студент"
    disciplines = _get_disciplines()
    user_id = update.effective_user.id
    started_states = list_user_course_states(user_id)
    active_discipline = get_active_discipline(user_id)
    text, markup = _build_course_browser_prompt(first_name, disciplines, started_states, active_discipline)
    await update.message.reply_text(text, reply_markup=markup, parse_mode="Markdown")


async def cmd_progress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    user_id = update.effective_user.id
    state = get_user_state(user_id)

    if not state or not state.get("discipline"):
        await update.message.reply_text("Сначала выбери дисциплину через /start")
        return

    await update.message.reply_text(format_progress(state), parse_mode="Markdown")


def _plan_nav_markup(state: dict) -> InlineKeyboardMarkup:
    """Контекстные кнопки под планом — зависят от текущего режима."""
    mode = state.get("mode", "")
    plan = state.get("plan", [])
    current_idx = state.get("current_block_idx", 0)

    if mode in ("quiz", "learning", "qa", "lesson_chat"):
        btn = InlineKeyboardButton("▶️ Вернуться к уроку", callback_data="back_to_block")
    elif mode == "awaiting_next":
        btn = InlineKeyboardButton("➡️ Следующий блок", callback_data="next_block")
    elif mode == "awaiting_retry":
        btn = InlineKeyboardButton("🔁 Повторить блок", callback_data="next_block")
    elif mode == "awaiting_start" or (plan and current_idx == 0 and not any(b.get("completed") for b in plan)):
        btn = InlineKeyboardButton("🚀 Начать обучение", callback_data="start_learning")
    elif plan and current_idx < len(plan) and state.get("last_block_quiz_context"):
        btn = InlineKeyboardButton("📝 Пройти квиз", callback_data="start_quiz")
    elif plan and current_idx < len(plan):
        btn = InlineKeyboardButton("➡️ Продолжить", callback_data="next_block")
    else:
        return InlineKeyboardMarkup([[InlineKeyboardButton("📚 Мои курсы", callback_data="courses")]])

    return InlineKeyboardMarkup([
        [btn],
        [InlineKeyboardButton("📚 Мои курсы", callback_data="courses")],
    ])


async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    user_id = update.effective_user.id
    state = get_user_state(user_id)

    if not state or not state.get("plan"):
        await update.message.reply_text("План ещё не построен. Пройди онбординг через /start")
        return

    await update.message.reply_text(
        format_plan(state),
        parse_mode="Markdown",
        reply_markup=_plan_nav_markup(state),
    )


async def cmd_skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    user_id = update.effective_user.id
    state = get_user_state(user_id)

    if not state or not state.get("plan"):
        await update.message.reply_text("Нет активного плана.")
        return

    idx = state.get("current_block_idx", 0)
    plan = state.get("plan", [])

    if idx >= len(plan):
        await update.message.reply_text("Ты уже прошёл все блоки!")
        return

    result = await invoke_graph_async(user_id, {
        "current_block_idx": idx + 1,
        "mode": "learning",
        "last_message": "",
    })
    if not result:
        await update.message.reply_text("⏳ Ещё обрабатывается предыдущий запрос, подожди немного.")
        return
    response = result.get("response", f"Блок пропущен. Переходим к блоку {idx + 2}.")
    await update.message.reply_text(response, parse_mode="Markdown")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    user_id = update.effective_user.id
    first_name = update.effective_user.first_name or "Студент"
    state = get_user_state(user_id)
    discipline = state.get("discipline", "") if state else ""
    disciplines = _get_disciplines()

    if is_user_busy(user_id):
        await update.message.reply_text(
            "⏳ Сейчас ещё обрабатывается предыдущий запрос. "
            "Дождись ответа по блоку и затем повтори /reset."
        )
        return

    if state:
        reset_user(user_id, discipline=discipline or None)

    if discipline:
        intro = f"Прогресс по дисциплине «{get_discipline_label(discipline)}» сброшен.\n\nНачнём заново."
    else:
        intro = f"Активный прогресс очищен, {first_name}."

    started_states = list_user_course_states(user_id)
    active_discipline = get_active_discipline(user_id)
    text, markup = _build_course_browser_prompt(first_name, disciplines, started_states, active_discipline, intro=intro)
    await update.message.reply_text(text, reply_markup=markup, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_restricted(update):
        return
    text = (
        "📚 *AI-репетитор — команды:*\n\n"
        "/start — выбрать дисциплину и начать\n"
        "/plan — мой план обучения\n"
        "/course — список всех курсов и прогресс\n"
        "/progress — XP, ачивки, статистика\n"
        "/skip — пропустить текущий блок\n"
        "/reset — начать дисциплину заново\n"
        "/help — эта справка"
    )
    if _is_admin_user(update.effective_user.id):
        text += "\n/admin — пользователи и их уровень"
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin(update):
        return
    config = load_config()
    users = await asyncio.to_thread(list_recent_users, config.sqlite_path, 10)
    status = await asyncio.to_thread(get_maintenance_status, config.sqlite_path)
    await update.message.reply_text(
        build_admin_panel_text(await asyncio.to_thread(build_admin_status_text, config.sqlite_path)),
        parse_mode="Markdown",
        reply_markup=build_admin_panel_markup(users, status["enabled"]),
    )


async def cmd_dbusers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin(update):
        return
    config = load_config()
    report = await asyncio.to_thread(build_db_users_report, config.sqlite_path)
    await update.message.reply_text(report, parse_mode="Markdown")


async def cmd_dbuser(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin(update):
        return

    raw_user_id = (context.args[0] if context.args else "").strip()
    if not raw_user_id:
        await update.message.reply_text("Укажи `telegram_user_id`, например `/dbuser 146025524`.", parse_mode="Markdown")
        return

    config = load_config()
    report = await asyncio.to_thread(build_db_user_report, config.sqlite_path, raw_user_id)
    await update.message.reply_text(report, parse_mode="Markdown")


async def _ensure_admin(update: Update) -> bool:
    config = load_config()
    user_id = update.effective_user.id
    if not config.admin_user_ids:
        await update.message.reply_text(
            "Admin-команды выключены: добавь `TELEGRAM_ADMIN_IDS` в `.env`.",
            parse_mode="Markdown",
        )
        return False
    if user_id not in config.admin_user_ids:
        await update.message.reply_text("Эта команда доступна только администратору.")
        return False
    return True
