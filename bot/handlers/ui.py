# Роль файла: Общие клавиатуры и безопасная отправка ответов.
"""Общие UI-хелперы для Telegram."""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from bot.telegram_text import split_text_for_telegram


def _subtopic_step_markup(current_idx: int, total_parts: int) -> InlineKeyboardMarkup:
    """Кнопка степпера для навигации по разделам блока.

    current_idx — индекс текущей (уже показанной) части (0-based).
    total_parts — всего частей.

    Последняя часть (индекс total_parts-1) содержит выводы и footer — для неё
    эта функция не вызывается, там идут стандартные кнопки квиза.
    """
    next_idx = current_idx + 1
    is_penultimate = (next_idx == total_parts - 1)  # следующая = последняя (выводы)
    if is_penultimate:
        label = "📋 К ключевым выводам →"
    else:
        total_q = total_parts - 1  # количество учебных вопросов (без финальной части)
        next_q = current_idx + 2   # 1-based номер следующего вопроса
        label = f"➡️ Учебный вопрос {next_q} из {total_q}"
    return _with_courses_button([[InlineKeyboardButton(label, callback_data="next_subtopic")]])


def _with_courses_button(rows: list[list[InlineKeyboardButton]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(rows + [[
        InlineKeyboardButton("📋 Мой план", callback_data="show_plan"),
        InlineKeyboardButton("📚 Все курсы", callback_data="courses"),
    ]])


def _lesson_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([
        [InlineKeyboardButton("💬 Спросить по блоку", callback_data="start_lesson_chat")],
        [InlineKeyboardButton("📝 Пройти квиз", callback_data="start_quiz")],
    ])


def _lesson_chat_markup() -> InlineKeyboardMarkup:
    return _with_courses_button([
        [InlineKeyboardButton("📝 Пройти квиз", callback_data="start_quiz")],
        [InlineKeyboardButton("🔙 Вернуться к блоку", callback_data="back_to_block")],
    ])


async def _safe_reply(message, text: str, markup=None) -> None:
    """Отправляет ответ и при необходимости режет его на части."""
    chunks = split_text_for_telegram(text)
    if not chunks:
        return
    for index, chunk in enumerate(chunks):
        chunk_markup = markup if index == len(chunks) - 1 else None
        try:
            await message.reply_text(chunk, parse_mode="Markdown", reply_markup=chunk_markup)
        except Exception:
            await message.reply_text(chunk, reply_markup=chunk_markup)
