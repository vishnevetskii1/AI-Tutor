# Роль файла: Собирает LangGraph-граф и маршрутизацию узлов.
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from bot.agents.state import State
from bot.agents.orchestrator import orchestrator
from bot.agents.profile_agent import profile_agent
from bot.agents.planner_agent import planner_agent
from bot.agents.content_agent import content_agent
from bot.agents.quiz_agent import quiz_agent
from bot.agents.qa_agent import qa_agent
from bot.agents.evaluator import evaluator


def _route(state: State) -> str:
    mode = state.get("mode", "")
    if mode == "onboarding":
        return "profile_agent"
    if mode == "planning":
        return "planner_agent"
    if mode == "learning":
        return "content_agent"
    if mode in {"qa", "lesson_chat", "lesson_chat_intro"}:
        return "qa_agent"
    if mode == "quiz":
        return "quiz_agent"
    if mode == "evaluate":
        return "evaluator"
    return END


def build_graph(db_path: str = "data/tutor.db", *, return_checkpointer: bool = False):
    builder = StateGraph(State)

    # Регистрируем все узлы графа один раз при сборке.
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("profile_agent", profile_agent)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("content_agent", content_agent)
    builder.add_node("qa_agent", qa_agent)
    builder.add_node("quiz_agent", quiz_agent)
    builder.add_node("evaluator", evaluator)

    builder.set_entry_point("orchestrator")

    # Оркестратор смотрит на текущий режим и выбирает следующий узел.
    builder.add_conditional_edges("orchestrator", _route, {
        "profile_agent": "profile_agent",
        "planner_agent": "planner_agent",
        "content_agent": "content_agent",
        "qa_agent": "qa_agent",
        "quiz_agent": "quiz_agent",
        "evaluator": "evaluator",
        END: END,
    })

    # После выполнения конкретного агента граф на этом проходе завершается.
    for node in ["profile_agent", "planner_agent", "content_agent", "qa_agent", "quiz_agent", "evaluator"]:
        builder.add_edge(node, END)

    # Состояние хранится в SQLite, чтобы пользователь мог продолжать курс между сообщениями.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)
    if return_checkpointer:
        return graph, checkpointer
    return graph
