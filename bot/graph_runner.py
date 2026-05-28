# Роль файла: Общий доступ к графу, локи по пользователям и операции с состоянием.
"""Общий экземпляр графа и обвязка для работы с состоянием пользователя."""
import asyncio
from functools import partial
from bot.agents.graph import build_graph
from bot.config import load_config
from bot.course_sessions import (
    clear_active_course,
    get_active_course,
    list_started_disciplines,
    set_active_course,
)

_graph = None
_checkpointer = None
_user_locks: dict[int, asyncio.Lock] = {}


def _get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


def get_graph():
    global _graph, _checkpointer
    if _graph is None:
        config = load_config()
        _graph, _checkpointer = build_graph(config.sqlite_path, return_checkpointer=True)
    return _graph


def _snapshot_for_thread(thread_id: str):
    g = get_graph()
    return g.get_state({"configurable": {"thread_id": thread_id}})


def _legacy_thread_id(user_id: int) -> str:
    return str(user_id)


def _course_thread_id(user_id: int, discipline: str) -> str:
    return f"{user_id}:{discipline}"


def _legacy_state_for_discipline(user_id: int, discipline: str) -> dict:
    snapshot = _snapshot_for_thread(_legacy_thread_id(user_id))
    values = snapshot.values if snapshot and snapshot.values else {}
    if values.get("discipline") == discipline:
        return values
    return {}


def _resolve_thread_id(user_id: int, discipline: str | None = None) -> str:
    # Если дисциплина уже известна, сначала ищем отдельный поток именно для неё.
    if discipline:
        snapshot = _snapshot_for_thread(_course_thread_id(user_id, discipline))
        if snapshot and snapshot.values:
            return _course_thread_id(user_id, discipline)
        # Если отдельного потока ещё нет, но есть старое состояние прежнего формата,
        # временно продолжаем работать через legacy-поток.
        if _legacy_state_for_discipline(user_id, discipline):
            return _legacy_thread_id(user_id)
        return _course_thread_id(user_id, discipline)

    config = load_config()
    active_discipline = get_active_course(config.sqlite_path, user_id)
    if active_discipline:
        return _resolve_thread_id(user_id, active_discipline)
    return _legacy_thread_id(user_id)


def set_active_discipline(user_id: int, discipline: str) -> None:
    config = load_config()
    set_active_course(config.sqlite_path, user_id, discipline)


def get_active_discipline(user_id: int) -> str | None:
    config = load_config()
    return get_active_course(config.sqlite_path, user_id)


def list_user_course_states(user_id: int) -> list[dict]:
    config = load_config()
    started = list_started_disciplines(config.sqlite_path, user_id)
    legacy = _snapshot_for_thread(_legacy_thread_id(user_id))
    legacy_state = legacy.values if legacy and legacy.values else {}
    legacy_discipline = legacy_state.get("discipline")
    if legacy_discipline and legacy_discipline not in started:
        started.append(legacy_discipline)

    active_discipline = get_active_course(config.sqlite_path, user_id)
    ordered = sorted(started, key=lambda item: (item != active_discipline, item))

    states = []
    for discipline in ordered:
        state = get_user_state(user_id, discipline=discipline)
        if state:
            states.append(state)
    return states


def make_initial_state(user_id: int, discipline: str) -> dict:
    """Создаёт пустое состояние для нового курса."""
    return {
        "user_id": user_id,
        "discipline": discipline,
        "level": "",
        "plan": [],
        "current_block_idx": 0,
        "history": [],
        "xp": 0,
        "achievements": [],
        "mode": "onboarding",
        "last_message": "",
        "response": "",
        "onboarding_step": 0,
        "onboarding_answers": [],
        "last_block_content": "",
        "last_block_quiz_context": "",
        "retry_count": 0,
        "quiz_questions": [],
        "quiz_current_idx": 0,
        "quiz_answers": [],
        "block_parts": [],
        "current_subtopic_idx": 0,
    }


def get_user_state(user_id: int, discipline: str | None = None) -> dict:
    snapshot = _snapshot_for_thread(_resolve_thread_id(user_id, discipline))
    return snapshot.values if snapshot and snapshot.values else {}


def is_user_busy(user_id: int) -> bool:
    return _get_user_lock(user_id).locked()


def invoke_graph(user_id: int, updates: dict, discipline: str | None = None) -> dict:
    """Запускает граф с частичным обновлением состояния."""
    g = get_graph()
    # Поток выбирается либо по явной дисциплине, либо по тому, что пришло в updates.
    thread_id = _resolve_thread_id(user_id, discipline or updates.get("discipline"))
    cfg = {"configurable": {"thread_id": thread_id}}
    result = g.invoke(updates, config=cfg)
    resolved_discipline = discipline or result.get("discipline") or updates.get("discipline")
    if resolved_discipline:
        set_active_discipline(user_id, resolved_discipline)
    return result


def reset_user(user_id: int, discipline: str | None = None) -> None:
    """Удаляет checkpoint активной дисциплины или всего legacy-потока пользователя."""
    get_graph()  # убеждаемся что checkpointer инициализирован
    target_thread_id = _resolve_thread_id(user_id, discipline)
    _checkpointer.delete_thread(target_thread_id)
    if discipline:
        active = get_active_discipline(user_id)
        if active == discipline:
            clear_active_course(load_config().sqlite_path, user_id)
    elif target_thread_id == _legacy_thread_id(user_id):
        clear_active_course(load_config().sqlite_path, user_id)


def update_user_state(user_id: int, updates: dict, discipline: str | None = None) -> None:
    """Напрямую обновляет поля состояния в checkpointer без запуска агентов."""
    g = get_graph()
    thread_id = _resolve_thread_id(user_id, discipline)
    cfg = {"configurable": {"thread_id": thread_id}}
    g.update_state(cfg, updates)


async def invoke_graph_async(user_id: int, updates: dict, discipline: str | None = None) -> dict:
    """Асинхронная обёртка вокруг синхронного запуска графа."""
    lock = _get_user_lock(user_id)
    if lock.locked():
        # Не запускаем второй запрос параллельно, чтобы не упереться в SQLite.
        return {}
    async with lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(invoke_graph, user_id, updates, discipline))
