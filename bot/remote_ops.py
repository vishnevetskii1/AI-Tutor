# Роль файла: Описывает удалённые сервисные операции и их обёртки.
import secrets
import subprocess
import time
from dataclasses import dataclass

from bot.config import Config
from bot.system_checks import (
    build_health_report,
    build_ollama_report,
    build_pm2_report,
    build_qdrant_report,
)


@dataclass
class PendingRemoteOp:
    token: str
    alias: str
    user_id: int
    created_at: float


@dataclass(frozen=True)
class RemoteOpSpec:
    description: str
    mode: str  # "report" | "command"
    command: list[str] | None = None


REMOTE_OPS: dict[str, RemoteOpSpec] = {
    "health": RemoteOpSpec("Проверить всё окружение", "report"),
    "qdrant": RemoteOpSpec("Проверить Qdrant", "report"),
    "ollama": RemoteOpSpec("Проверить Ollama", "report"),
    "pm2": RemoteOpSpec("Показать процессы PM2", "report"),
    "docker_ps": RemoteOpSpec(
        "Показать контейнеры Docker",
        "command",
        ["docker", "ps", "--format", "{{.Names}} | {{.Status}} | {{.Image}}"],
    ),
    "bot_logs": RemoteOpSpec(
        "Показать последние ошибки бота",
        "command",
        ["tail", "-n", "40", "/root/.pm2/logs/ai-tutor-bot-error.log"],
    ),
    "bot_restart": RemoteOpSpec(
        "Перезапустить ai-tutor-bot через PM2",
        "command",
        ["pm2", "restart", "ai-tutor-bot"],
    ),
}

PENDING_REMOTE_OPS: dict[str, PendingRemoteOp] = {}
PENDING_TTL_SECONDS = 300


def list_remote_ops() -> str:
    lines = ["Доступные remote ops:"]
    for alias, spec in REMOTE_OPS.items():
        lines.append(f"- {alias}: {spec.description}")
    return "\n".join(lines)


def create_pending_remote_op(alias: str, user_id: int) -> PendingRemoteOp:
    token = secrets.token_urlsafe(9)
    pending = PendingRemoteOp(token=token, alias=alias, user_id=user_id, created_at=time.time())
    PENDING_REMOTE_OPS[token] = pending
    _cleanup_pending()
    return pending


def get_pending_remote_op(token: str) -> PendingRemoteOp | None:
    _cleanup_pending()
    return PENDING_REMOTE_OPS.get(token)


def pop_pending_remote_op(token: str) -> PendingRemoteOp | None:
    _cleanup_pending()
    return PENDING_REMOTE_OPS.pop(token, None)


def execute_remote_op(alias: str, config: Config) -> str:
    spec = REMOTE_OPS[alias]
    if spec.mode == "report":
        if alias == "health":
            return build_health_report(config)
        if alias == "qdrant":
            return build_qdrant_report(config)
        if alias == "ollama":
            return build_ollama_report(config)
        if alias == "pm2":
            return build_pm2_report()
        raise ValueError(f"Unknown report op: {alias}")

    completed = subprocess.run(
        spec.command,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    output = (completed.stdout or completed.stderr or "").strip()
    if not output:
        output = "Command finished without output."
    status = "OK" if completed.returncode == 0 else f"FAIL (exit {completed.returncode})"
    return f"{alias}: {status}\n{output}"


def truncate_remote_output(text: str, limit: int = 3500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _cleanup_pending() -> None:
    now = time.time()
    expired = [
        token
        for token, pending in PENDING_REMOTE_OPS.items()
        if now - pending.created_at > PENDING_TTL_SECONDS
    ]
    for token in expired:
        PENDING_REMOTE_OPS.pop(token, None)
