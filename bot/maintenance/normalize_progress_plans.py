# Роль файла: Сервисный скрипт нормализации сохранённых учебных планов.
import argparse
import sqlite3
from collections.abc import Iterable

from bot.agents.graph import build_graph
from bot.config import load_config


def normalize_plan(plan: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    positions: dict[str, int] = {}

    for block in plan or []:
        if not isinstance(block, dict):
            continue

        topic = " ".join(str(block.get("topic", "")).split())
        key = topic.casefold() if topic else f"index:{block.get('index', len(normalized))}"

        if key not in positions:
            clean = dict(block)
            clean["index"] = len(normalized)
            clean["completed"] = bool(block.get("completed"))
            positions[key] = len(normalized)
            normalized.append(clean)
            continue

        if block.get("completed"):
            normalized[positions[key]] = {
                **normalized[positions[key]],
                "completed": True,
            }

    return normalized


def _thread_ids(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [str(row[0]) for row in rows]


def _clamp_current_idx(value: object, total: int) -> int:
    if not isinstance(value, int):
        return 0
    return max(0, min(value, total))


def normalize_checkpoint_plans(db_path: str, dry_run: bool = False) -> list[tuple[str, int, int, int]]:
    graph = build_graph(db_path)
    changed: list[tuple[str, int, int, int]] = []

    for thread_id in _thread_ids(db_path):
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(cfg)
        state = snapshot.values if snapshot and snapshot.values else {}
        plan = state.get("plan") or []
        normalized = normalize_plan(plan)

        if len(plan) == len(normalized):
            continue

        current_idx = _clamp_current_idx(state.get("current_block_idx", 0), len(normalized))
        changed.append((thread_id, len(plan), len(normalized), current_idx))

        if not dry_run:
            graph.update_state(cfg, {
                "plan": normalized,
                "current_block_idx": current_idx,
            })

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize duplicated course plans in LangGraph checkpoints.")
    parser.add_argument("--db-path", default=load_config().sqlite_path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = normalize_checkpoint_plans(args.db_path, dry_run=args.dry_run)
    for thread_id, before, after, current_idx in changed:
        action = "would update" if args.dry_run else "updated"
        print(f"{action}: {thread_id}: {before} -> {after}, current_block_idx={current_idx}")
    print(f"changed={len(changed)}")


if __name__ == "__main__":
    main()
