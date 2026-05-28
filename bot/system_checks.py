# Роль файла: Runtime-проверки окружения и состояния деплоя.
import subprocess
from pathlib import Path

import httpx

from bot.config import Config
from bot.disciplines import get_discipline_label


def build_health_report(config: Config) -> str:
    parts = [
        _check_qdrant(config),
        _check_ollama(config),
        _check_docker(),
        _check_pm2(),
        _check_local_data(config),
    ]
    return "\n\n".join(parts)


def build_qdrant_report(config: Config) -> str:
    return _check_qdrant(config)


def build_ollama_report(config: Config) -> str:
    return _check_ollama(config)


def build_docker_report() -> str:
    return _check_docker()


def build_pm2_report() -> str:
    return _check_pm2()


def _check_qdrant(config: Config) -> str:
    base = config.qdrant_url.rstrip("/")
    try:
        response = httpx.get(
            f"{base}/collections",
            timeout=httpx.Timeout(min(config.qdrant_timeout_seconds, 5.0), connect=2.0),
        )
        response.raise_for_status()
        payload = response.json()
        collections = [
            get_discipline_label(item["name"])
            for item in payload.get("result", {}).get("collections", [])
        ]
        if collections:
            listed = ", ".join(collections[:10])
            return f"Qdrant: OK\nURL: {config.qdrant_url}\nCollections: {listed}"
        return f"Qdrant: OK\nURL: {config.qdrant_url}\nCollections: empty"
    except Exception as exc:
        return f"Qdrant: FAIL\nURL: {config.qdrant_url}\nError: {exc}"


def _check_ollama(config: Config) -> str:
    base = config.ollama_url.rstrip("/")
    try:
        response = httpx.get(
            f"{base}/api/tags",
            timeout=httpx.Timeout(min(config.ollama_timeout_seconds, 5.0), connect=2.0),
        )
        response.raise_for_status()
        payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        if models:
            listed = ", ".join(models[:10])
            return f"Ollama: OK\nURL: {config.ollama_url}\nModels: {listed}"
        return f"Ollama: OK\nURL: {config.ollama_url}\nModels: empty"
    except Exception as exc:
        return f"Ollama: FAIL\nURL: {config.ollama_url}\nError: {exc}"


def _check_docker() -> str:
    return _run_command_report(
        title="Docker",
        command=["docker", "ps", "--format", "{{.Names}} | {{.Status}} | {{.Image}}"],
        empty_message="No running containers",
    )


def _check_pm2() -> str:
    return _run_command_report(
        title="PM2",
        command=["pm2", "ls", "--no-color"],
        empty_message="No PM2 processes",
    )


def _check_local_data(config: Config) -> str:
    db_path = Path(config.sqlite_path)
    disciplines_dir = Path(config.disciplines_dir)
    discipline_names = sorted(
        get_discipline_label(entry.name) for entry in disciplines_dir.iterdir()
        if disciplines_dir.exists() and entry.is_dir() and not entry.name.startswith(".")
    )
    disciplines_text = ", ".join(discipline_names) if discipline_names else "empty"
    return (
        "Local data: OK\n"
        f"SQLite: {'present' if db_path.exists() else 'missing'} ({db_path})\n"
        f"Disciplines: {disciplines_text}"
    )


def _run_command_report(title: str, command: list[str], empty_message: str) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return f"{title}: FAIL\nError: {exc}"

    if completed.returncode != 0:
        error = _summarize_error_output(completed.stderr or completed.stdout) or f"exit code {completed.returncode}"
        return f"{title}: FAIL\nError: {error}"

    output = completed.stdout.strip()
    if not output:
        return f"{title}: OK\n{empty_message}"

    lines = output.splitlines()[:12]
    return f"{title}: OK\n" + "\n".join(lines)


def _summarize_error_output(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""

    summary: list[str] = []
    for line in lines:
        if line.startswith("at "):
            continue
        if line.startswith("Emitted 'error' event"):
            continue
        summary.append(line)
        if len(summary) == 3:
            break

    if not summary:
        summary = lines[:1]

    return " | ".join(summary)
