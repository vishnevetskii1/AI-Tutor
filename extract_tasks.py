# Роль файла: CLI-помощник для извлечения практических задач из материалов.
import argparse
from pathlib import Path

from bot.practice import extract_tasks_from_pdf, save_task_collection
from bot.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Извлечь задачи из PDF и сохранить в JSON")
    parser.add_argument("--discipline", required=True, help="Slug дисциплины")
    parser.add_argument("--pdf", required=True, help="Имя PDF внутри папки дисциплины")
    parser.add_argument("--page-start", type=int, help="Первая страница диапазона (1-indexed)")
    parser.add_argument("--page-end", type=int, help="Последняя страница диапазона (1-indexed, inclusive)")
    parser.add_argument("--max-tasks", type=int, help="Максимум задач в выходном JSON")
    parser.add_argument("--output", help="Путь для JSON; по умолчанию data/practice/<discipline>/<stem>.json")
    args = parser.parse_args()

    config = load_config()
    pdf_path = Path(config.disciplines_dir) / args.discipline / args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    collection = extract_tasks_from_pdf(
        pdf_path,
        discipline=args.discipline,
        page_start=args.page_start,
        page_end=args.page_end,
        max_tasks=args.max_tasks,
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path("data/practice") / args.discipline / f"{pdf_path.stem}.json"
    )
    save_task_collection(collection, output_path)
    print(f"saved={output_path}")
    print(f"tasks={collection['task_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
