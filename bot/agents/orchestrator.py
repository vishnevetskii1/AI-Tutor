# Роль файла: Выбирает следующий шаг графа по текущему state.
from bot.agents.state import State


def orchestrator(state: State) -> dict:
    """Определяет следующий шаг на основе текущего состояния."""
    if not state.get("level"):
        return {"mode": "onboarding"}
    if not state.get("plan"):
        return {"mode": "planning"}

    mode = state.get("mode", "learning")

    return {"mode": mode}
