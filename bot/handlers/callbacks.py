# Роль файла: Обработчики кнопок уроков, курсов, квиза и админ-действий.
import logging
import asyncio
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.access_control import get_access_denial_message
from bot.admin_controls import (
    DEFAULT_MAINTENANCE_MESSAGE,
    get_maintenance_status,
    set_maintenance_mode,
    set_user_blocked,
)
from bot.admin_db import build_admin_status_text, build_db_user_report, list_recent_users
from bot.graph_runner import (
    get_active_discipline,
    get_user_state,
    invoke_graph_async,
    list_user_course_states,
    make_initial_state,
    reset_user,
    set_active_discipline,
    update_user_state,
)
from bot.config import load_config
from bot.disciplines import get_discipline_description, get_discipline_label
from bot.formatting import format_progress
from bot.handlers.commands import (
    _build_course_browser_prompt,
    _get_disciplines,
    build_admin_panel_markup,
    build_admin_panel_text,
)
from bot.agents.quiz_agent import _format_quiz_question
from bot.handlers.ui import _lesson_chat_markup, _lesson_markup, _safe_reply, _subtopic_step_markup, _with_courses_button

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
ADMIN_DOCS = {
    "architecture": ("architecture.html", "Архитектура бота"),
    "history": ("dev-history.html", "История разработки"),
    "guide": ("user-guide.html", "Руководство пользователя"),
}


def _quiz_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([[InlineKeyboardButton("📝 Пройти квиз", callback_data="start_quiz")]])


def _next_block_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([[InlineKeyboardButton("➡️ Следующий блок", callback_data="next_block")]])


def _start_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([[InlineKeyboardButton("🚀 Начать обучение", callback_data="start_learning")]])


def _retry_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([[InlineKeyboardButton("🔁 Повторить блок", callback_data="next_block")]])


def _resume_plan_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([[InlineKeyboardButton("📋 Достроить план", callback_data="resume_planning")]])


def _quiz_question_markup(options: list[str], question_idx: int) -> InlineKeyboardMarkup:
    # В кнопках показываем только короткие метки A/B/C/D, а сами варианты остаются в тексте вопроса.
    labels = ["A", "B", "C", "D"]
    answer_buttons = [
        InlineKeyboardButton(
            labels[option_idx] if option_idx < len(labels) else str(option_idx + 1),
            callback_data=f"quiz_answer:{question_idx}:{option_idx}",
        )
        for option_idx, option in enumerate(options)
    ]
    keyboard = [answer_buttons[:2], answer_buttons[2:4]]
    keyboard = [row for row in keyboard if row]
    return _with_courses_button(keyboard)


def _admin_user_markup(thread_id: str) -> InlineKeyboardMarkup:
    config = load_config()
    blocked = set()
    if thread_id.isdigit():
        from bot.admin_controls import is_user_blocked
        if is_user_blocked(config.sqlite_path, int(thread_id)):
            blocked.add(thread_id)
    toggle_label = "✅ Включить пользователя" if thread_id in blocked else "⛔ Отключить пользователя"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(toggle_label, callback_data=f"admin_block:{thread_id}")],
        [InlineKeyboardButton("🔄 Обновить", callback_data=f"admin_user:{thread_id}")],
        [InlineKeyboardButton("⬅️ К пользователям", callback_data="admin_users")],
        [InlineKeyboardButton("❌ Закрыть", callback_data="admin_close")],
    ])


def _course_action_markup(state: dict, user_id: int | None = None) -> InlineKeyboardMarkup:
    mode = state.get("mode", "")
    plan = state.get("plan", [])
    current_idx = state.get("current_block_idx", 0)

    if mode == "awaiting_start":
        return _start_markup()
    if mode in ("quiz", "learning"):
        return _lesson_markup()
    if mode in ("qa", "lesson_chat"):
        return _lesson_chat_markup()
    if mode == "awaiting_next":
        return _next_block_markup()
    if mode == "awaiting_retry":
        return _retry_markup()
    if mode == "planning":
        return _resume_plan_markup()

    # Если состояние оказалось нештатным, но план уже есть, даём пользователю
    # рабочую кнопку продолжения вместо пустого экрана.
    if plan and current_idx < len(plan):
        if state.get("last_block_quiz_context"):
            return _lesson_markup()
        if current_idx == 0 and not any(block.get("completed") for block in plan):
            return _start_markup()
        return _next_block_markup()

    return _with_courses_button([])


async def handle_show_plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    state = get_user_state(user_id)
    plan = state.get("plan", []) if state else []
    if not plan:
        await query.message.reply_text("📋 План ещё не сформирован. Начни курс командой /start.")
        return

    from bot.formatting import format_plan
    from bot.handlers.commands import _plan_nav_markup
    text = format_plan(state)
    await _safe_reply(query.message, text, _plan_nav_markup(state))


async def handle_course_browser(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    denial = get_access_denial_message(user_id)
    if denial:
        await query.message.reply_text(denial)
        return
    first_name = getattr(query.from_user, "first_name", None) or "Студент"
    disciplines = _get_disciplines()
    started_states = list_user_course_states(user_id)
    active_discipline = get_active_discipline(user_id)
    text, markup = _build_course_browser_prompt(first_name, disciplines, started_states, active_discipline)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)


async def handle_course_select(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    denial = get_access_denial_message(user_id)
    if denial:
        await query.message.reply_text(denial)
        return
    discipline = query.data.split(":", 1)[1]
    existing = get_user_state(user_id, discipline=discipline)
    set_active_discipline(user_id, discipline)

    if existing and existing.get("discipline") == discipline:
        summary = format_progress(existing)
        markup = _course_action_markup(existing, user_id)
        header = f"Открыл курс *{get_discipline_label(discipline)}*.\n\n"
        await query.edit_message_text(header + summary, parse_mode="Markdown", reply_markup=markup)
        return

    description = get_discipline_description(discipline)
    description_block = f"\n\n_{description}_" if description else ""
    await query.edit_message_text(
        f"Выбрали новый курс: *{get_discipline_label(discipline)}*{description_block}\n\nОпределяю твой уровень...",
        parse_mode="Markdown",
    )

    reset_user(user_id, discipline=discipline)
    try:
        result = await invoke_graph_async(user_id, make_initial_state(user_id, discipline), discipline=discipline)
    except Exception as e:
        logger.exception("course select invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка: {e}")
        return

    response = result.get("response", "Начинаем!")
    await _safe_reply(query.message, response)


async def handle_discipline_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    denial = get_access_denial_message(user_id)
    if denial:
        await query.message.reply_text(denial)
        return
    discipline = query.data
    description = get_discipline_description(discipline)
    description_block = f"\n\n_{description}_" if description else ""
    await query.edit_message_text(
        f"Отлично! Выбрана дисциплина: *{get_discipline_label(discipline)}*{description_block}\n\nОпределяю твой уровень...",
        parse_mode="Markdown",
    )

    # Явный прямой выбор дисциплины тоже стартует с чистого checkpoint именно этой дисциплины.
    set_active_discipline(user_id, discipline)
    reset_user(user_id, discipline=discipline)
    try:
        result = await invoke_graph_async(user_id, make_initial_state(user_id, discipline), discipline=discipline)
    except Exception as e:
        logger.exception("discipline choice invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка: {e}")
        return

    response = result.get("response", "Начинаем!")
    await _safe_reply(query.message, response)


async def handle_start_learning(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка 'Начать обучение' — запускаем первый блок контента."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("⏳ Готовлю первый блок (~30 сек)...")

    user_id = query.from_user.id
    try:
        result = await invoke_graph_async(user_id, {"mode": "learning", "last_message": ""})
    except Exception as e:
        logger.exception("start_learning invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка генерации блока: {e}")
        return

    if not result:
        await query.message.reply_text("⏳ Ещё обрабатывается предыдущий запрос, подожди немного.")
        return

    response = result.get("response", "")
    if response:
        block_parts = result.get("block_parts", [])
        markup = _subtopic_step_markup(0, len(block_parts)) if len(block_parts) > 1 else _lesson_markup()
        await _safe_reply(query.message, response, markup)
    else:
        await query.message.reply_text("⚠️ Не удалось получить контент. Попробуй ещё раз.")


async def handle_resume_planning(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка восстановления курса, застрявшего на этапе построения плана."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("⏳ Достраиваю план курса...")

    user_id = query.from_user.id
    try:
        result = await invoke_graph_async(user_id, {"mode": "planning", "last_message": ""})
    except Exception as e:
        logger.exception("resume_planning invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка построения плана: {e}")
        return

    if not result:
        await query.message.reply_text("⏳ Ещё обрабатывается предыдущий запрос, подожди немного.")
        return

    response = result.get("response", "")
    markup = _course_action_markup(result)
    if response:
        await _safe_reply(query.message, response, markup)
    else:
        await query.message.reply_text("⚠️ Не удалось достроить план. Попробуй ещё раз.")


async def handle_start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка 'Пройти квиз' — генерируем 3 вопроса."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("⏳ Составляю вопросы (~30 сек)...")

    user_id = query.from_user.id
    try:
        result = await invoke_graph_async(user_id, {"mode": "quiz", "last_message": ""})
    except Exception as e:
        logger.exception("start_quiz invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка квиза: {e}")
        return

    response = result.get("response", "")
    questions = result.get("quiz_questions", [])
    if response and questions:
        await _safe_reply(query.message, response, _quiz_question_markup(questions[0]["options"], 0))
    else:
        await query.message.reply_text("⚠️ Не удалось составить вопросы. Попробуй ещё раз.")


async def handle_start_lesson_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка входа в чат-режим по текущему блоку."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)

    user_id = query.from_user.id
    try:
        result = await invoke_graph_async(user_id, {"mode": "lesson_chat_intro", "last_message": ""})
    except Exception as e:
        logger.exception("start_lesson_chat invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка чат-режима: {e}")
        return

    response = result.get("response", "")
    if response:
        await _safe_reply(query.message, response)
    else:
        await query.message.reply_text("⚠️ Не удалось открыть чат по блоку. Попробуй ещё раз.")


async def handle_back_to_block(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает последний учебный блок с обычными кнопками урока."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)

    existing = get_user_state(query.from_user.id)
    lesson_text = existing.get("last_block_quiz_context") or existing.get("response") or ""
    if not lesson_text:
        await query.message.reply_text("⚠️ Не удалось найти текущий учебный блок. Открой следующий блок или курс заново.")
        return

    await _safe_reply(query.message, lesson_text, _lesson_markup())


async def handle_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)

    user_id = query.from_user.id
    existing = get_user_state(user_id)
    questions = existing.get("quiz_questions", [])
    current_idx = existing.get("quiz_current_idx", 0)
    answers = list(existing.get("quiz_answers", []))
    plan = existing.get("plan", [])
    block_idx = existing.get("current_block_idx", 0)
    topic = plan[block_idx]["topic"] if block_idx < len(plan) else "тема"

    if existing.get("mode") != "quiz_active" or current_idx >= len(questions):
        await query.message.reply_text("Квиз уже завершён или сброшен. Запусти новый через кнопку «Пройти квиз».")
        return

    try:
        _, raw_question_idx, raw_option_idx = query.data.split(":")
        question_idx = int(raw_question_idx)
        option_idx = int(raw_option_idx)
    except ValueError:
        await query.message.reply_text("⚠️ Не удалось распознать ответ. Попробуй пройти квиз заново.")
        return

    if question_idx != current_idx:
        await query.message.reply_text("Этот вопрос уже неактуален. Ответь на текущий вопрос ниже.")
        return

    answers.append(option_idx)
    next_idx = current_idx + 1

    # Мгновенный фидбек — ✅ или ❌
    is_correct = (option_idx == questions[current_idx]["answer"])
    await query.message.reply_text("✅ Верно!" if is_correct else "❌ Неверно!")

    if next_idx < len(questions):
        await invoke_graph_async(user_id, {
            "mode": "quiz_active",
            "quiz_current_idx": next_idx,
            "quiz_answers": answers,
        })
        await _safe_reply(
            query.message,
            _format_quiz_question(topic, questions, next_idx),
            _quiz_question_markup(questions[next_idx]["options"], next_idx),
        )
        return

    encoded_answers = ", ".join(str(answer + 1) for answer in answers)
    try:
        result = await invoke_graph_async(user_id, {
            "mode": "evaluate",
            "last_message": encoded_answers,
            "quiz_answers": answers,
            "quiz_current_idx": next_idx,
        })
    except Exception as e:
        logger.exception("quiz evaluate invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка проверки квиза: {e}")
        return

    response = result.get("response", "")
    result_mode = result.get("mode", "")
    if result_mode == "awaiting_next":
        markup = _next_block_markup()
    elif result_mode == "awaiting_retry":
        markup = _retry_markup()
    else:
        markup = _with_courses_button([])
    if response:
        await _safe_reply(query.message, response, markup)
    else:
        await query.message.reply_text("⚠️ Квиз завершён, но не удалось получить результат. Попробуй /reset.")


async def handle_next_block(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка перехода к следующему или повторному блоку."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)

    user_id = query.from_user.id
    existing = get_user_state(user_id)
    is_retry = existing.get("mode") == "awaiting_retry"
    status_text = "⏳ Готовлю повтор блока (~30 сек)..." if is_retry else "⏳ Готовлю следующий блок (~30 сек)..."
    await query.message.reply_text(status_text)

    try:
        result = await invoke_graph_async(user_id, {"mode": "learning", "last_message": ""})
    except Exception as e:
        logger.exception("next_block invoke failed")
        await query.message.reply_text(f"⚠️ Ошибка: {e}")
        return

    if not result:
        await query.message.reply_text("⏳ Ещё обрабатывается предыдущий запрос.")
        return

    response = result.get("response", "")
    if response:
        block_parts = result.get("block_parts", [])
        markup = _subtopic_step_markup(0, len(block_parts)) if len(block_parts) > 1 else _lesson_markup()
        await _safe_reply(query.message, response, markup)
    else:
        await query.message.reply_text("⚠️ Не удалось получить блок. Попробуй ещё раз.")

async def handle_next_subtopic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка степпера — показывает следующий раздел текущего блока."""
    query = update.callback_query
    await query.answer()
    denial = get_access_denial_message(query.from_user.id)
    if denial:
        await query.message.reply_text(denial)
        return
    await query.edit_message_reply_markup(reply_markup=None)

    user_id = query.from_user.id
    state = get_user_state(user_id)
    block_parts = state.get("block_parts") or []
    current_idx = state.get("current_subtopic_idx") or 0
    next_idx = current_idx + 1

    if not block_parts or next_idx >= len(block_parts):
        await query.message.reply_text("⚠️ Нет следующего раздела.", reply_markup=_lesson_markup())
        return

    update_user_state(user_id, {"current_subtopic_idx": next_idx})

    is_last = (next_idx == len(block_parts) - 1)
    markup = _lesson_markup() if is_last else _subtopic_step_markup(next_idx, len(block_parts))
    await _safe_reply(query.message, block_parts[next_idx], markup)


async def handle_admin_docs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    config = load_config()
    if not config.admin_user_ids or query.from_user.id not in config.admin_user_ids:
        await query.message.reply_text("Эта панель доступна только администратору.")
        return

    markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("🗺 Архитектура бота", callback_data="admin_doc:architecture")],
        [InlineKeyboardButton("📖 История разработки", callback_data="admin_doc:history")],
        [InlineKeyboardButton("📘 Руководство пользователя", callback_data="admin_doc:guide")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="admin_users")],
    ])
    await query.edit_message_text(
        "📄 *Документация*\n\nВыбери документ — бот отправит актуальный HTML-файл прямо в чат.",
        parse_mode="Markdown",
        reply_markup=markup,
    )


async def handle_admin_doc_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    config = load_config()
    if not config.admin_user_ids or query.from_user.id not in config.admin_user_ids:
        await query.message.reply_text("Эта панель доступна только администратору.")
        return

    doc_key = query.data.split(":", 1)[1]
    doc_meta = ADMIN_DOCS.get(doc_key)
    if not doc_meta:
        await query.message.reply_text("⚠️ Неизвестный документ.")
        return

    filename, title = doc_meta
    doc_path = DOCS_DIR / filename
    if not doc_path.exists():
        await query.message.reply_text(f"⚠️ Файл документации не найден: {filename}")
        return

    with doc_path.open("rb") as document:
        await query.message.reply_document(
            document=document,
            filename=filename,
            caption=f"📄 {title}",
        )


async def handle_admin_panel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    config = load_config()
    user_id = query.from_user.id
    if not config.admin_user_ids or user_id not in config.admin_user_ids:
        await query.message.reply_text("Эта панель доступна только администратору.")
        return

    data = query.data

    if data == "admin_close":
        await query.edit_message_reply_markup(reply_markup=None)
        return

    if data == "admin_users":
        users = await asyncio.to_thread(list_recent_users, config.sqlite_path, 10)
        maintenance = await asyncio.to_thread(get_maintenance_status, config.sqlite_path)
        status_line = await asyncio.to_thread(build_admin_status_text, config.sqlite_path)
        text = build_admin_panel_text(status_line)
        if not users:
            text = f"{text}\n\nПользователей в базе пока нет."
            markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("🛠 Выключить техработы" if maintenance["enabled"] else "🛠 Включить техработы", callback_data="admin_maintenance_toggle")],
                [InlineKeyboardButton("❌ Закрыть", callback_data="admin_close")],
            ])
        else:
            markup = build_admin_panel_markup(users, maintenance["enabled"])
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)
        return

    if data.startswith("admin_user:"):
        thread_id = data.split(":", 1)[1]
        report = await asyncio.to_thread(build_db_user_report, config.sqlite_path, thread_id)
        await query.edit_message_text(
            report,
            parse_mode="Markdown",
            reply_markup=_admin_user_markup(thread_id),
        )
        return

    if data == "admin_maintenance_toggle":
        status = await asyncio.to_thread(get_maintenance_status, config.sqlite_path)
        new_enabled = not status["enabled"]
        new_status = await asyncio.to_thread(
            set_maintenance_mode,
            config.sqlite_path,
            new_enabled,
            status["message"] or DEFAULT_MAINTENANCE_MESSAGE,
        )
        users = await asyncio.to_thread(list_recent_users, config.sqlite_path, 10)
        await query.edit_message_text(
            build_admin_panel_text(await asyncio.to_thread(build_admin_status_text, config.sqlite_path)),
            parse_mode="Markdown",
            reply_markup=build_admin_panel_markup(users, new_status["enabled"]),
        )
        return

    if data.startswith("admin_block:"):
        raw_user_id = data.split(":", 1)[1]
        if not raw_user_id.isdigit():
            await query.message.reply_text("Не удалось определить пользователя.")
            return
        target_user_id = int(raw_user_id)
        from bot.admin_controls import is_user_blocked

        blocked = await asyncio.to_thread(is_user_blocked, config.sqlite_path, target_user_id)
        await asyncio.to_thread(set_user_blocked, config.sqlite_path, target_user_id, not blocked)
        report = await asyncio.to_thread(build_db_user_report, config.sqlite_path, raw_user_id)
        await query.edit_message_text(
            report,
            parse_mode="Markdown",
            reply_markup=_admin_user_markup(raw_user_id),
        )
        if not blocked:
            try:
                await context.bot.send_message(
                    chat_id=target_user_id,
                    text="⛔ Доступ к боту временно отключён администратором.",
                )
            except Exception:
                logger.exception("blocked user notification failed for user_id=%s", target_user_id)
        return
