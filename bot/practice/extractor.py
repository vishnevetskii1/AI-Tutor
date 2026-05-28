# Роль файла: Извлекает наборы практических задач из материалов.
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bot.rag.pdf_audit import extract_pdf_page_texts

TASK_START_RE = re.compile(
    r"^(?P<kind>Задача|Упражнение|Пример)\s*(?P<number>\d+)?[\.\):]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
NUMBERED_TASK_RE = re.compile(r"^(?P<number>\d+)\*?[\.\)]\s*(?P<title>.*)$")
ANSWER_START_RE = re.compile(r"^(?:Ответ|Указание|Решение)\s*[:.]?\s*(?P<body>.*)$", re.IGNORECASE)
ANSWER_ITEM_RE = re.compile(r"(?<!\d)(?P<number>\d+)\.\s*")
SECTION_RE = re.compile(
    r"^(?:§\s*\d+\.?\s+.+|Глава\s+[A-ZА-ЯIVXLC0-9]+\.?\s+.+|ГЛАВА\s+[A-ZА-ЯIVXLC0-9]+\.?\s+.+)$"
)


@dataclass
class PracticeTask:
    task_id: str
    discipline: str
    topic: str
    difficulty: str
    problem_text: str
    answer_text: str
    hint_1: str
    hint_2: str
    source_pdf: str
    source_pages: list[int]


def extract_tasks_from_pdf(
    pdf_path: str | Path,
    discipline: str,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
    max_tasks: int | None = None,
) -> dict:
    path = Path(pdf_path)
    page_texts = extract_pdf_page_texts(path, page_start=page_start, page_end=page_end)
    page_offset = (page_start - 1) if page_start else 0
    tasks = _parse_tasks(
        page_texts,
        discipline=discipline,
        source_pdf=path.name,
        page_offset=page_offset,
    )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return {
        "schema_version": 1,
        "discipline": discipline,
        "source_pdf": path.name,
        "page_start": page_start,
        "page_end": page_end,
        "task_count": len(tasks),
        "tasks": [asdict(task) for task in tasks],
    }


def save_task_collection(collection: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n")


def _parse_tasks(
    page_texts: list[str],
    discipline: str,
    source_pdf: str,
    *,
    page_offset: int = 0,
) -> list[PracticeTask]:
    task_entries: list[dict] = []
    current_topic = discipline
    current: dict | None = None
    collecting_answer = False
    in_answers_section = False
    answer_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current, collecting_answer
        if not current:
            return
        problem_text = _normalize_block(current["problem"])
        answer_text = _normalize_block(current["answer"])
        if len(problem_text) < 40:
            current = None
            collecting_answer = False
            return
        if len(current["pages"]) > 4:
            current = None
            collecting_answer = False
            return
        task_entries.append({
            "number": current.get("number") or str(len(task_entries) + 1),
            "topic": current.get("topic") or discipline,
            "problem_text": problem_text,
            "answer_text": answer_text,
            "source_pages": sorted(current["pages"]),
        })
        current = None
        collecting_answer = False

    for page_idx, page_text in enumerate(page_texts, start=page_offset + 1):
        lines = [_normalize_line(line) for line in (page_text or "").splitlines()]
        for line in lines:
            if not line:
                continue
            if _is_answers_heading(line):
                flush_current()
                in_answers_section = True
                collecting_answer = False
                continue
            if in_answers_section:
                answer_lines.append(line)
                continue
            if _is_section_heading(line):
                flush_current()
                current_topic = line
                continue

            task_match = _match_task_start(line)
            if task_match:
                flush_current()
                current = {
                    "kind": task_match.groupdict().get("kind", "Задача"),
                    "number": task_match.group("number"),
                    "topic": current_topic,
                    "problem": [],
                    "answer": [],
                    "pages": {page_idx},
                }
                title = task_match.group("title").strip()
                if title:
                    current["problem"].append(title)
                collecting_answer = False
                continue

            if current is None:
                continue

            current["pages"].add(page_idx)
            answer_match = ANSWER_START_RE.match(line)
            if answer_match:
                collecting_answer = True
                body = answer_match.group("body").strip()
                if body:
                    current["answer"].append(body)
                continue

            if collecting_answer:
                current["answer"].append(line)
            else:
                current["problem"].append(line)

    flush_current()
    expected_numbers = {task_entry["number"] for task_entry in task_entries}
    answer_key = _parse_answer_key(answer_lines, expected_numbers)
    tasks: list[PracticeTask] = []
    for task_entry in task_entries:
        answer_text = _normalize_answer_text(task_entry["answer_text"] or answer_key.get(task_entry["number"], ""))
        if not answer_text:
            continue
        tasks.append(
            PracticeTask(
                task_id=f"{discipline}:{Path(source_pdf).stem}:{task_entry['number']}",
                discipline=discipline,
                topic=task_entry["topic"],
                difficulty=_estimate_difficulty(task_entry["problem_text"]),
                problem_text=task_entry["problem_text"],
                answer_text=answer_text,
                hint_1=_build_hint(task_entry["problem_text"], answer_text, strict=False),
                hint_2=_build_hint(task_entry["problem_text"], answer_text, strict=True),
                source_pdf=source_pdf,
                source_pages=task_entry["source_pages"],
            )
        )
    return tasks


def _match_task_start(line: str):
    return TASK_START_RE.match(line) or NUMBERED_TASK_RE.match(line)


def _is_answers_heading(line: str) -> bool:
    return line.casefold() == "ответы"


def _parse_answer_key(lines: list[str], expected_numbers: set[str]) -> dict[str, str]:
    answers: dict[str, list[str]] = {}
    current_number: str | None = None
    for raw_line in lines:
        line = _normalize_line(raw_line)
        if not line:
            continue
        if _is_section_heading(line) or _is_answers_heading(line):
            current_number = None
            continue
        matches = [
            match for match in ANSWER_ITEM_RE.finditer(line)
            if match.group("number") in expected_numbers
        ]
        if not matches:
            if current_number:
                answers.setdefault(current_number, []).append(line)
            continue
        for idx, match in enumerate(matches):
            current_number = match.group("number")
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            body = line[body_start:body_end].strip(" ;")
            if body:
                answers.setdefault(current_number, []).append(body)
    return {
        number: _normalize_block(parts)
        for number, parts in answers.items()
        if _normalize_block(parts)
    }


def _normalize_line(line: str) -> str:
    return " ".join((line or "").replace("\xa0", " ").split()).strip()


def _normalize_block(lines: list[str]) -> str:
    cleaned = [_normalize_line(line) for line in lines if _normalize_line(line)]
    return "\n".join(cleaned).strip()


def _is_section_heading(line: str) -> bool:
    if len(line) > 140:
        return False
    if SECTION_RE.match(line):
        return True
    if line.isupper() and len(line.split()) <= 12:
        return True
    return False


def _estimate_difficulty(problem_text: str) -> str:
    length = len(problem_text)
    lowered = problem_text.casefold()
    multi_step_markers = ["докажите", "найдите", "вычислите", "определите", "постройте"]
    marker_count = sum(marker in lowered for marker in multi_step_markers)
    if length < 220 and marker_count <= 1:
        return "basic"
    if length < 700 and marker_count <= 2:
        return "standard"
    return "advanced"


def _build_hint(problem_text: str, answer_text: str, strict: bool) -> str:
    source = answer_text or problem_text
    sentences = re.split(r"(?<=[\.\!\?])\s+", source)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return ""
    if strict or len(sentences) == 1:
        return sentences[0]
    return " ".join(sentences[:2])


def _normalize_answer_text(text: str) -> str:
    normalized = _normalize_block(text.splitlines())
    return re.sub(r";\s*([A-Za-zА-Яа-я]\))", r"\n\1", normalized)
