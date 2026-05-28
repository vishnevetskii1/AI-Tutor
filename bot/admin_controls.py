# Роль файла: Переключатели техобслуживания и связанные helper-функции.
import sqlite3


DEFAULT_MAINTENANCE_MESSAGE = (
    "⚙️ Сейчас идут технические работы.\n\n"
    "Бот временно недоступен. Попробуй чуть позже."
)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blocked_users (
                user_id INTEGER PRIMARY KEY,
                is_blocked INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_flags (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def is_user_blocked(db_path: str, user_id: int) -> bool:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT is_blocked FROM blocked_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return bool(row and row["is_blocked"])


def set_user_blocked(db_path: str, user_id: int, blocked: bool) -> None:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        if blocked:
            conn.execute(
                """
                INSERT INTO blocked_users (user_id, is_blocked, updated_at)
                VALUES (?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    is_blocked = 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id,),
            )
        else:
            conn.execute("DELETE FROM blocked_users WHERE user_id = ?", (user_id,))
        conn.commit()


def get_maintenance_status(db_path: str) -> dict:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        enabled_row = conn.execute(
            "SELECT value, updated_at FROM admin_flags WHERE key = 'maintenance_mode'"
        ).fetchone()
        message_row = conn.execute(
            "SELECT value FROM admin_flags WHERE key = 'maintenance_message'"
        ).fetchone()
    enabled = enabled_row["value"] == "1" if enabled_row else False
    return {
        "enabled": enabled,
        "message": message_row["value"] if message_row else DEFAULT_MAINTENANCE_MESSAGE,
        "updated_at": enabled_row["updated_at"] if enabled_row else None,
    }


def set_maintenance_mode(db_path: str, enabled: bool, message: str | None = None) -> dict:
    ensure_tables(db_path)
    text = (message or DEFAULT_MAINTENANCE_MESSAGE).strip() or DEFAULT_MAINTENANCE_MESSAGE
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO admin_flags (key, value, updated_at)
            VALUES ('maintenance_mode', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            ("1" if enabled else "0",),
        )
        conn.execute(
            """
            INSERT INTO admin_flags (key, value, updated_at)
            VALUES ('maintenance_message', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (text,),
        )
        conn.commit()
    return get_maintenance_status(db_path)


def list_all_known_user_ids(db_path: str) -> list[int]:
    ensure_tables(db_path)
    raw_user_ids: set[int] = set()

    with _connect(db_path) as conn:
        try:
            checkpoint_rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        except sqlite3.OperationalError:
            checkpoint_rows = []
        active_rows = conn.execute("SELECT DISTINCT user_id FROM active_courses").fetchall()
        blocked_rows = conn.execute("SELECT DISTINCT user_id FROM blocked_users").fetchall()

    for row in checkpoint_rows:
        thread_id = str(row["thread_id"])
        base_user_id = thread_id.split(":", 1)[0].strip()
        if base_user_id.isdigit():
            raw_user_ids.add(int(base_user_id))

    for row in active_rows + blocked_rows:
        raw_user_ids.add(int(row["user_id"]))

    return sorted(raw_user_ids)
