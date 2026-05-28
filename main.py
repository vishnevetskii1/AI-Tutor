# Роль файла: Точка входа бота и регистрация Telegram-обработчиков.
import logging
import os
from telegram.error import Conflict
from telegram import (
    BotCommand,
    BotCommandScopeAllChatAdministrators,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    BotCommandScopeDefault,
)
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from bot.config import load_config
from bot.handlers.commands import (
    cmd_start,
    cmd_course,
    cmd_plan,
    cmd_progress,
    cmd_skip,
    cmd_reset,
    cmd_help,
    cmd_admin,
)
from bot.handlers.messages import handle_message
from bot.handlers.callbacks import (
    handle_discipline_choice,
    handle_course_browser,
    handle_course_select,
    handle_resume_planning,
    handle_start_learning,
    handle_start_lesson_chat,
    handle_back_to_block,
    handle_start_quiz,
    handle_quiz_answer,
    handle_next_block,
    handle_next_subtopic,
    handle_admin_panel_callback,
    handle_admin_docs,
    handle_admin_doc_file,
    handle_show_plan,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBLIC_COMMANDS = [
    BotCommand("start", "Выбрать дисциплину и начать"),
    BotCommand("plan", "Мой план обучения"),
    BotCommand("course", "Список курсов и прогресс"),
    BotCommand("progress", "Текущий прогресс"),
    BotCommand("skip", "Пропустить текущий блок"),
    BotCommand("reset", "Начать заново"),
    BotCommand("help", "Справка"),
]

ADMIN_COMMANDS = PUBLIC_COMMANDS + [
    BotCommand("admin", "Пользователи и их уровень"),
]


async def _configure_bot_commands(app, admin_user_ids: list[int]) -> None:
    base_scopes = [
        BotCommandScopeDefault(),
        BotCommandScopeAllPrivateChats(),
        BotCommandScopeAllGroupChats(),
        BotCommandScopeAllChatAdministrators(),
    ]
    for scope in base_scopes:
        await app.bot.delete_my_commands(scope=scope)
    for admin_user_id in admin_user_ids:
        await app.bot.delete_my_commands(scope=BotCommandScopeChat(chat_id=admin_user_id))

    await app.bot.set_my_commands(PUBLIC_COMMANDS, scope=BotCommandScopeDefault())
    await app.bot.set_my_commands(PUBLIC_COMMANDS, scope=BotCommandScopeAllPrivateChats())
    for admin_user_id in admin_user_ids:
        await app.bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=admin_user_id))


def main():
    config = load_config()
    logger.info(
        "Bot config loaded provider=%s ollama_url=%s ollama_model=%s cwd=%s pid=%s",
        config.llm_provider,
        config.ollama_url,
        config.ollama_model,
        os.getcwd(),
        os.getpid(),
    )

    app = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .post_init(lambda app: _configure_bot_commands(app, config.admin_user_ids))
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("plan", cmd_plan))
    app.add_handler(CommandHandler("course", cmd_course))
    app.add_handler(CommandHandler("progress", cmd_progress))
    app.add_handler(CommandHandler("skip", cmd_skip))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("admin", cmd_admin))

    # Сначала регистрируем точечные обработчики кнопок.
    app.add_handler(CallbackQueryHandler(handle_admin_panel_callback, pattern="^admin_(users|user:|maintenance_toggle|block:|close)"))
    app.add_handler(CallbackQueryHandler(handle_admin_docs, pattern="^admin_docs$"))
    app.add_handler(CallbackQueryHandler(handle_admin_doc_file, pattern="^admin_doc:"))
    app.add_handler(CallbackQueryHandler(handle_show_plan, pattern="^show_plan$"))
    app.add_handler(CallbackQueryHandler(handle_course_browser, pattern="^courses$"))
    app.add_handler(CallbackQueryHandler(handle_course_select, pattern="^course:"))
    app.add_handler(CallbackQueryHandler(handle_resume_planning, pattern="^resume_planning$"))
    app.add_handler(CallbackQueryHandler(handle_start_learning, pattern="^start_learning$"))
    app.add_handler(CallbackQueryHandler(handle_start_lesson_chat, pattern="^start_lesson_chat$"))
    app.add_handler(CallbackQueryHandler(handle_back_to_block, pattern="^back_to_block$"))
    app.add_handler(CallbackQueryHandler(handle_start_quiz, pattern="^start_quiz$"))
    app.add_handler(CallbackQueryHandler(handle_quiz_answer, pattern="^quiz_answer:"))
    app.add_handler(CallbackQueryHandler(handle_next_block, pattern="^next_block$"))
    app.add_handler(CallbackQueryHandler(handle_next_subtopic, pattern="^next_subtopic$"))
    # Этот обработчик забирает все остальные нажатия выбора дисциплины.
    app.add_handler(CallbackQueryHandler(handle_discipline_choice))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting bot polling")
    try:
        app.run_polling()
    except Conflict:
        logger.exception(
            "Telegram polling conflict: another bot instance is already calling getUpdates. "
            "For this project keep only one active runtime, preferably Docker Compose, "
            "and stop/remove any PM2 or manual launch."
        )
        raise
    except Exception:
        logger.exception("Bot failed during startup or polling")
        raise


if __name__ == "__main__":
    main()
