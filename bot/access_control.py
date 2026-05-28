# Роль файла: Проверки доступа для закрытых и admin-only сценариев.
from bot.admin_controls import DEFAULT_MAINTENANCE_MESSAGE, get_maintenance_status, is_user_blocked
from bot.config import load_config


def is_admin_user(user_id: int) -> bool:
    return user_id in load_config().admin_user_ids


def get_access_denial_message(user_id: int) -> str | None:
    config = load_config()
    if user_id in config.admin_user_ids:
        return None
    if is_user_blocked(config.sqlite_path, user_id):
        return "⛔ Доступ к боту для тебя отключён администратором."
    maintenance = get_maintenance_status(config.sqlite_path)
    if maintenance["enabled"]:
        return maintenance["message"] or DEFAULT_MAINTENANCE_MESSAGE
    return None
