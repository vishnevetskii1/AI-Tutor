# Роль файла: Консольная точка входа для загрузки PDF в Qdrant.
"""Консольный запуск загрузки PDF-учебников в Qdrant.

Пример:
    uv run python ingest.py --discipline math
"""
import argparse
from pathlib import Path
from bot.rag.ingest import ingest
from bot.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Загрузить PDF по дисциплине в Qdrant")
    parser.add_argument("--discipline", required=True, help="Slug дисциплины (имя папки)")
    args = parser.parse_args()

    config = load_config()
    discipline_dir = Path(config.disciplines_dir) / args.discipline

    if not discipline_dir.exists():
        print(f"Папка не найдена: {discipline_dir}")
        return

    pdfs = list(discipline_dir.glob("*.pdf"))
    if not pdfs:
        print(f"PDF не найдены в {discipline_dir}")
        return

    for pdf in pdfs:
        print(f"Загружаю: {pdf.name} ...")
        ingest(pdf, args.discipline)
        print(f"  готово")

    print(f"Загружено {len(pdfs)} PDF в коллекцию '{args.discipline}'")


if __name__ == "__main__":
    main()
