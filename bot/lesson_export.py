# Роль файла: Экспортирует уроки в Markdown и HTML.
from __future__ import annotations

import re
from datetime import datetime
from html import escape
from pathlib import Path

from bot.disciplines import get_discipline_label


EXPORT_ROOT = Path(__file__).resolve().parents[1] / "work" / "exports"
CTA_LINES = {
    "Если что-то осталось непонятным, задай уточняющий вопрос по текущей теме и подпунктам этого блока.",
    "Если хочешь разобрать этот блок в формате диалога, нажми кнопку «💬 Спросить по блоку».",
    "Готов к квизу? Нажми кнопку ниже.",
}
FRONT_MATTER = """---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: Inter, Arial, sans-serif;
    font-size: 26px;
    padding: 56px 72px;
    line-height: 1.35;
    background: #f8fafc;
    color: #0f172a;
  }
  h1, h2, h3 {
    color: #0f172a;
  }
  h1 {
    font-size: 2.0em;
    margin-bottom: 0.25em;
  }
  h2 {
    font-size: 1.4em;
    margin-bottom: 0.35em;
  }
  h3 {
    font-size: 1.1em;
    margin-bottom: 0.35em;
  }
  strong {
    color: #1d4ed8;
  }
  ul {
    margin-top: 0.3em;
  }
  section.lead {
    justify-content: center;
    background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
  }
  section.lead h1 {
    font-size: 2.3em;
    margin-bottom: 0.2em;
  }
---
"""


def _inline_html(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "lesson"


def _strip_wrapper(text: str) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    lines = body.splitlines()
    if lines and re.match(r"^[^\w#]*\*?.+\*?$", lines[0]) and len(lines) > 2 and not lines[0].startswith("#"):
        if lines[1].strip() == "":
            lines = lines[2:]
    kept = [line for line in lines if line.strip() not in CTA_LINES]
    return "\n".join(kept).strip()


def lesson_markdown(topic: str, source_text: str) -> str:
    body = _strip_wrapper(source_text)
    if not body:
        body = f"## {topic}\n\n### Основная часть\n\nМатериал блока отсутствует."
    if "## " not in body:
        body = f"## {topic}\n\n{body}"
    return body.strip() + "\n"


def _split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    lines = markdown.splitlines()
    title = "Учебный блок"
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            title = line[3:].strip()
            continue
        if line.startswith("### "):
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = line[4:].strip()
            current_lines = []
            continue
        if current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, current_lines))

    return title, [(heading, "\n".join(content).strip()) for heading, content in sections]


def _section_items(content: str) -> list[str]:
    items: list[str] = []
    for block in content.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if all(line.startswith("- ") for line in lines):
            items.extend(line[2:].strip() for line in lines)
            continue
        merged = " ".join(line[2:].strip() if line.startswith("- ") else line for line in lines)
        items.append(merged)
    return items


def _chunked(items: list[str], size: int = 4) -> list[list[str]]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)] or [[]]


def _build_slide_specs(markdown: str, discipline: str | None = None) -> list[dict[str, object]]:
    title, sections = _split_sections(markdown)
    slides: list[dict[str, object]] = []
    lead_slide = {
        "kind": "lead",
        "title": title,
        "discipline": discipline,
        "meta": "Формат: учебный блок для чтения и последующего квиза",
    }
    slides.append(lead_slide)
    if not sections:
        slides.append({"kind": "bullets", "title": "Основная часть", "items": ["Материал блока отсутствует."]})
        return slides

    for heading, content in sections:
        items = _section_items(content)
        for chunk_index, chunk in enumerate(_chunked(items, size=4)):
            slides.append({
                "kind": "bullets",
                "title": heading if chunk_index == 0 else f"{heading} (продолжение)",
                "items": chunk or ["Пустой раздел"],
            })

    return slides


def render_marp(markdown: str, discipline: str | None = None) -> str:
    slide_specs = _build_slide_specs(markdown, discipline)
    slides: list[str] = [FRONT_MATTER.strip()]

    lead = slide_specs[0]
    lead_slide = [
        "<!-- _class: lead -->",
        f"# {lead['title']}",
    ]
    if lead.get("discipline"):
        lead_slide.append(f"**Дисциплина:** {lead['discipline']}")
    lead_slide.append("")
    lead_slide.append(f"**{lead['meta']}**")
    slides.append("\n".join(lead_slide))

    for slide in slide_specs[1:]:
        body = "\n".join(f"- {item}" for item in slide["items"])
        slides.append(f"## {slide['title']}\n\n{body}")

    return "\n\n---\n\n".join(slides) + "\n"


def render_viewer_html(markdown: str, discipline: str | None = None) -> str:
    slide_specs = _build_slide_specs(markdown, discipline)
    sections: list[str] = []

    for index, slide in enumerate(slide_specs):
        if slide["kind"] == "lead":
            discipline_html = (
                f"<p class=\"slide-discipline\">Дисциплина: {_inline_html(str(slide['discipline']))}</p>"
                if slide.get("discipline")
                else ""
            )
            body_html = (
                f"<p class=\"slide-meta\">{_inline_html(str(slide['meta']))}</p>"
                if slide.get("meta")
                else ""
            )
            body = (
                f"<section class=\"slide lead{' active' if index == 0 else ''}\" data-slide-index=\"{index}\">"
                f"<div class=\"slide-shell\"><h1>{_inline_html(str(slide['title']))}</h1>{discipline_html}{body_html}</div></section>"
            )
        else:
            items = "".join(f"<li>{_inline_html(str(item))}</li>" for item in slide["items"])
            body = (
                f"<section class=\"slide{' active' if index == 0 else ''}\" data-slide-index=\"{index}\">"
                f"<div class=\"slide-shell\"><h2>{_inline_html(str(slide['title']))}</h2><ul>{items}</ul></div></section>"
            )
        sections.append(body)

    slides_html = "\n".join(sections)
    deck_title = _inline_html(str(slide_specs[0]["title"])) if slide_specs else "Учебный блок"
    total_slides = len(slide_specs)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{deck_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: radial-gradient(circle at top, #dbeafe 0%, #f8fafc 36%, #e2e8f0 100%);
      --surface: rgba(255, 255, 255, 0.88);
      --border: rgba(15, 23, 42, 0.08);
      --text: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
      --accent-soft: rgba(37, 99, 235, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .viewer {{
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 20px;
      gap: 16px;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 12px 14px;
      backdrop-filter: blur(16px);
    }}
    .toolbar-title {{
      font-size: 14px;
      color: var(--muted);
    }}
    .toolbar-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}
    button.secondary {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: default;
    }}
    .deck {{
      position: relative;
      flex: 1;
      min-height: 0;
    }}
    .slide {{
      display: none;
      min-height: calc(100vh - 120px);
      align-items: center;
      justify-content: center;
    }}
    .slide.active {{
      display: flex;
    }}
    .slide-shell {{
      width: min(1100px, 100%);
      min-height: min(78vh, 760px);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 32px;
      padding: clamp(24px, 4vw, 56px);
      box-shadow: 0 30px 80px rgba(15, 23, 42, 0.12);
    }}
    .lead .slide-shell {{
      background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(219,234,254,0.92));
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: clamp(32px, 5vw, 60px);
      line-height: 1.05;
    }}
    h2 {{
      margin: 0 0 20px;
      font-size: clamp(26px, 4vw, 42px);
      line-height: 1.1;
    }}
    .slide-discipline, .slide-meta, .toolbar-title {{
      margin: 0;
      line-height: 1.5;
    }}
    .slide-discipline {{
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .slide-meta {{
      color: var(--muted);
      max-width: 52ch;
      font-size: clamp(18px, 2vw, 24px);
    }}
    ul {{
      margin: 0;
      padding-left: 1.2em;
      display: grid;
      gap: 14px;
      font-size: clamp(20px, 2vw, 30px);
      line-height: 1.35;
    }}
    strong, code {{
      color: var(--accent);
    }}
    code {{
      background: rgba(37, 99, 235, 0.08);
      border-radius: 8px;
      padding: 0.08em 0.28em;
    }}
    @media (max-width: 700px) {{
      .viewer {{
        padding: 12px;
      }}
      .toolbar {{
        flex-direction: column;
        align-items: stretch;
      }}
      .toolbar-actions {{
        justify-content: space-between;
      }}
      button {{
        flex: 1;
      }}
      .slide {{
        min-height: calc(100vh - 164px);
      }}
      .slide-shell {{
        border-radius: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main class="viewer">
    <div class="toolbar">
      <p class="toolbar-title"><strong>Marp Viewer</strong> · <span id="page-indicator">1 / {total_slides}</span></p>
      <div class="toolbar-actions">
        <button class="secondary" id="prev-button">Назад</button>
        <button id="next-button">Дальше</button>
      </div>
    </div>
    <div class="deck">
      {slides_html}
    </div>
  </main>
  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    const pageIndicator = document.getElementById('page-indicator');
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    let currentIndex = 0;

    const render = () => {{
      slides.forEach((slide, index) => slide.classList.toggle('active', index === currentIndex));
      pageIndicator.textContent = `${{currentIndex + 1}} / ${{slides.length}}`;
      prevButton.disabled = currentIndex === 0;
      nextButton.disabled = currentIndex === slides.length - 1;
    }};

    prevButton.addEventListener('click', () => {{
      if (currentIndex > 0) {{
        currentIndex -= 1;
        render();
      }}
    }});

    nextButton.addEventListener('click', () => {{
      if (currentIndex < slides.length - 1) {{
        currentIndex += 1;
        render();
      }}
    }});

    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') {{
        prevButton.click();
      }}
      if (event.key === 'ArrowRight' || event.key === ' ') {{
        nextButton.click();
      }}
    }});

    render();
  </script>
</body>
</html>
"""


def _build_base_path(topic: str, export_format: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(topic)
    export_dir = EXPORT_ROOT / export_format
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir / f"{timestamp}-{slug}"


def export_marp(topic: str, source_text: str, discipline: str) -> Path:
    output = _build_base_path(topic, "marp").with_suffix(".marp.md")
    lesson_md = lesson_markdown(topic, source_text)
    output.write_text(render_marp(lesson_md, get_discipline_label(discipline)), encoding="utf-8")
    return output


def export_marp_viewer(topic: str, source_text: str, discipline: str) -> Path:
    output = _build_base_path(topic, "marp-viewer").with_suffix(".html")
    lesson_md = lesson_markdown(topic, source_text)
    output.write_text(render_viewer_html(lesson_md, get_discipline_label(discipline)), encoding="utf-8")
    return output
