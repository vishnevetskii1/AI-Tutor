# Роль файла: Единый слой вызова моделей для всех провайдеров.
import logging
import time
import httpx
from bot.config import load_config

logger = logging.getLogger(__name__)


def generate(
    prompt: str,
    system: str = "",
    label: str = "unknown",
    options: dict | None = None,
    timeout_seconds: float | None = None,
) -> str:
    config = load_config()

    if config.llm_provider == "openrouter":
        return _openrouter(
            prompt,
            system,
            config,
            label=label,
            options=options,
            timeout_seconds=timeout_seconds,
        )

    if config.llm_provider == "ollama":
        try:
            return _ollama(
                prompt,
                system,
                config,
                label=label,
                options=options,
                timeout_seconds=timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(_format_ollama_error(exc)) from exc

    raise ValueError(f"Неизвестный LLM_PROVIDER: {config.llm_provider!r}. Поддерживается: openrouter, ollama")


def _openrouter(
    prompt: str,
    system: str,
    config,
    label: str = "unknown",
    options: dict | None = None,
    timeout_seconds: float | None = None,
) -> str:
    from openrouter import OpenRouter

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    request_timeout = timeout_seconds or config.ollama_timeout_seconds
    request_options = options or {}
    completion_tokens = request_options.get("num_predict")
    continuation_limit = int(request_options.get("continue_on_length", 0))

    started_at = time.perf_counter()
    logger.info(
        "OpenRouter request started label=%s model=%s prompt_chars=%s system_chars=%s timeout=%.1fs",
        label,
        config.llm_model,
        len(prompt),
        len(system),
        request_timeout,
    )
    try:
        with OpenRouter(
            api_key=config.openrouter_api_key,
            timeout_ms=int(request_timeout * 1000),
        ) as client:
            parts: list[str] = []
            continuation_count = 0
            while True:
                payload: dict = {
                    "model": config.llm_model,
                    "messages": messages,
                }
                if "temperature" in request_options:
                    payload["temperature"] = request_options["temperature"]
                # num_predict — ollama-specific параметр; для OpenRouter не передаём,
                # чтобы модель (Gemini и др.) не обрезала текст на полуслове.

                response = client.chat.send(**payload)
                choice = response.choices[0]
                chunk = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                usage = getattr(response, "usage", None)
                completion_used = getattr(usage, "completion_tokens", None) if usage else None
                prompt_used = getattr(usage, "prompt_tokens", None) if usage else None
                logger.info(
                    "OpenRouter chunk label=%s model=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s chars=%s continuation=%s",
                    label,
                    config.llm_model,
                    finish_reason,
                    prompt_used,
                    completion_used,
                    len(chunk),
                    continuation_count,
                )
                parts.append(chunk)
                if finish_reason != "length" or continuation_count >= continuation_limit or not chunk.strip():
                    break

                messages.append({"role": "assistant", "content": chunk})
                messages.append({
                    "role": "user",
                    "content": "Продолжи с того места, где остановился. Не начинай заново и не повторяй уже сказанное.",
                })
                continuation_count += 1
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        logger.exception(
            "OpenRouter request failed label=%s model=%s after %.2fs",
            label,
            config.llm_model,
            elapsed,
        )
        raise RuntimeError(f"OpenRouter ошибка: {exc}") from exc

    elapsed = time.perf_counter() - started_at
    logger.info(
        "OpenRouter request finished label=%s model=%s in %.2fs",
        label,
        config.llm_model,
        elapsed,
    )
    return "".join(parts).strip()


def _ollama(
    prompt: str,
    system: str,
    config,
    label: str = "unknown",
    options: dict | None = None,
    timeout_seconds: float | None = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    request_timeout = timeout_seconds or config.ollama_timeout_seconds
    started_at = time.perf_counter()
    prompt_chars = len(prompt)
    system_chars = len(system)
    logger.info(
        "Ollama request started label=%s model=%s prompt_chars=%s system_chars=%s timeout=%.1fs",
        label,
        config.ollama_model,
        prompt_chars,
        system_chars,
        request_timeout,
    )
    try:
        response = httpx.post(
            f"{config.ollama_url}/api/chat",
            json={
                "model": config.ollama_model,
                "messages": messages,
                "stream": False,
                "options": options or {},
            },
            timeout=httpx.Timeout(request_timeout, connect=5.0),
        )
        elapsed = time.perf_counter() - started_at
        response.raise_for_status()
    except httpx.HTTPError:
        elapsed = time.perf_counter() - started_at
        logger.exception(
            "Ollama request failed label=%s model=%s after %.2fs prompt_chars=%s system_chars=%s",
            label,
            config.ollama_model,
            elapsed,
            prompt_chars,
            system_chars,
        )
        raise
    logger.info(
        "Ollama request finished label=%s model=%s in %.2fs with status=%s",
        label,
        config.ollama_model,
        elapsed,
        response.status_code,
    )
    return response.json()["message"]["content"]


def _format_ollama_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ReadTimeout):
        return "Локальная модель не ответила вовремя. Проверь Ollama или включи fallback на Google."
    if isinstance(exc, httpx.ConnectError):
        return "Не удалось подключиться к локальной модели. Проверь Ollama и OLLAMA_URL."
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 404:
            return (
                "Ollama вернул HTTP 404. Проверь OLLAMA_MODEL и OLLAMA_URL. "
                "Если бот запущен в Docker, для Ollama на хосте используй "
                "http://host.docker.internal:11434 вместо localhost."
            )
        return (
            f"Локальная модель вернула HTTP {exc.response.status_code}. "
            "Проверь OLLAMA_URL и OLLAMA_MODEL."
        )
    return f"Ошибка локальной модели: {exc}"
