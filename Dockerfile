FROM python:3.12-slim

# Ставим uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Сначала ставим зависимости, чтобы лучше работал кэш слоёв
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Копируем исходники
COPY bot/ ./bot/
COPY docs/ ./docs/
COPY main.py ingest.py ./

# Создаём папки для данных
RUN mkdir -p data/disciplines data/db

ENV PYTHONUNBUFFERED=1
ENV SQLITE_PATH=/app/data/db/tutor.db
ENV DISCIPLINES_DIR=/app/data/disciplines

CMD ["uv", "run", "python", "main.py"]
