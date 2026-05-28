# Роль файла: Хранит и переключает активную дисциплину пользователя.
import sqlite3


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_courses (
                user_id INTEGER PRIMARY KEY,
                discipline TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def set_active_course(db_path: str, user_id: int, discipline: str) -> None:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO active_courses (user_id, discipline, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                discipline=excluded.discipline,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, discipline),
        )
        conn.commit()


def get_active_course(db_path: str, user_id: int) -> str | None:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT discipline FROM active_courses WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return row["discipline"]


def clear_active_course(db_path: str, user_id: int) -> None:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM active_courses WHERE user_id = ?", (user_id,))
        conn.commit()


def list_started_disciplines(db_path: str, user_id: int) -> list[str]:
    ensure_tables(db_path)
    prefix = f"{user_id}:"
    disciplines: list[str] = []

    with _connect(db_path) as conn:
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE thread_id LIKE ?
                ORDER BY thread_id
                """,
                (f"{prefix}%",),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

    for row in rows:
        thread_id = row["thread_id"]
        if not thread_id.startswith(prefix):
            continue
        discipline = thread_id[len(prefix):].strip()
        if discipline:
            disciplines.append(discipline)

    return disciplines
