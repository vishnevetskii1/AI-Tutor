# Роль файла: Обработка обычных текстовых сообщений пользователя.
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.access_control import get_access_denial_message
from bot.graph_runner import get_user_state, invoke_graph_async
from bot.agents.state import Message
from bot.handlers.ui import _lesson_chat_markup, _lesson_markup, _safe_reply, _with_courses_button

logger = logging.getLogger(__name__)


def _make_markup(mode: str, state: dict | None = None, user_id: int | None = None) -> InlineKeyboardMarkup | None:
    if mode == "awaiting_start":
        return _with_courses_button([[InlineKeyboardButton("🚀 Начать обучение", callback_data="start_learning")]])
    if mode in ("quiz", "learning"):
        return _lesson_markup()
    if mode in ("qa", "lesson_chat"):
        return _lesson_chat_markup()
    if mode == "awaiting_next":
        return _with_courses_button([[InlineKeyboardButton("➡️ Следующий блок", callback_data="next_block")]])
    if mode == "awaiting_retry":
        return _with_courses_button([[InlineKeyboardButton("🔁 Повторить блок", callback_data="next_block")]])
    return None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = update.message.text
    denial = get_access_denial_message(user_id)
    if denial:
        await update.message.reply_text(denial)
        return

    try:
        existing = get_user_state(user_id)
    except Exception as e:
        logger.exception("get_user_state failed")
        await update.message.reply_text("⚠️ Ошибка состояния. Попробуй /reset")
        return

    if not existing:
        await update.message.reply_text("Напиши /start чтобы начать!")
        return

    mode = existing.get("mode", "idle")

    if mode == "quiz_active":
        await update.message.reply_text("Выбери вариант ответа кнопкой под текущим вопросом.")
        return

    CHAT_MODES = {"qa", "lesson_chat", "lesson_chat_intro"}
    if mode not in CHAT_MODES and mode != "onboarding":
        try:
            await update.message.delete()
        except Exception:
            pass
        markup = _lesson_markup()
        text = (
            "✍️ Писать боту можно только в режиме чата по учебному блоку.\n\n"
            "Нажми `💬 Спросить по блоку`, если хочешь задать вопрос по текущему материалу,\n"
            "или `📝 Пройти квиз`, чтобы продолжить урок."
        )
        if context and getattr(context, "bot", None):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=text,
                parse_mode="Markdown",
                reply_markup=markup,
            )
        else:
            await _safe_reply(update.message, text, markup)
        return

    next_mode = mode

    user_msg: Message = {"role": "user", "content": text}

    try:
        result = await invoke_graph_async(user_id, {
            "last_message": text,
            "history": [user_msg],
            "mode": next_mode,
        })
    except Exception as e:
        logger.exception("invoke_graph_async failed (message handler)")
        await update.message.reply_text(f"⚠️ Ошибка обработки: {e}\n\nПопробуй ещё раз или /reset")
        return

    if not result:
        # Уже идёт предыдущая обработка этого пользователя.
        await update.message.reply_text("⏳ Обрабатываю предыдущий запрос, подожди...")
        return

    response = result.get("response", "")
    result_mode = result.get("mode", "")

    # Онбординг завершён — автоматически запускаем планировщик
    if result_mode == "planning":
        if response:
            await _safe_reply(update.message, response)
        await update.message.reply_text("⏳ Составляю план курса...")
        try:
            plan_result = await invoke_graph_async(user_id, {"mode": "planning", "last_message": ""})
        except Exception as e:
            logger.exception("planner invoke failed")
            await update.message.reply_text(f"⚠️ Ошибка планировщика: {e}\n\nПопробуй /reset")
            return
        response = plan_result.get("response", "")
        result_mode = plan_result.get("mode", "awaiting_start")
        markup = _make_markup(result_mode, plan_result, user_id)
        if response:
            await _safe_reply(update.message, response, markup)
        return

    if not response:
        return

    markup = _make_markup(result_mode, result, user_id)
    await _safe_reply(update.message, response, markup)
