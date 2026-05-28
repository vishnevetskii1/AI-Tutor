# Пакет bot

Здесь лежит основная рабочая логика проекта.

Лучший порядок чтения:

1. `config.py`
2. `graph_runner.py`
3. `handlers/commands.py`
4. `handlers/callbacks.py`
5. `agents/graph.py`
6. `agents/orchestrator.py`

Как устроена папка:

- `handlers/` - входные точки Telegram
- `agents/` - узлы графа и логика генерации
- `rag/` - поиск по PDF и векторная индексация
- `practice/` - извлечение практических задач
- `maintenance/` - разовые сервисные скрипты

Главная идея:

- handlers принимают события из Telegram
- graph_runner загружает и обновляет состояние
- agents генерируют план, урок, квиз и ответы
- rag даёт опорный контекст из PDF
