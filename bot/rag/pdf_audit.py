# Роль файла: Проверяет, можно ли нормально использовать PDF в RAG.
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import fitz

from bot.rag.chunker import chunk_text

SCAN_LIKE_MIN_PAGES = 20
MIN_TEXT_PAGE_RATIO = 0.2
MIN_WORDS_PER_PAGE = 10.0


@dataclass(frozen=True)
class PdfAuditResult:
    pdf_path: Path
    file_size_bytes: int
    page_count: int
    nonempty_page_count: int
    extracted_word_count: int
    chunk_count: int
    extraction_method: str

    @property
    def words_per_page(self) -> float:
        if self.page_count == 0:
            return 0.0
        return self.extracted_word_count / self.page_count

    @property
    def text_page_ratio(self) -> float:
        if self.page_count == 0:
            return 0.0
        return self.nonempty_page_count / self.page_count

    @property
    def issues(self) -> list[str]:
        issues: list[str] = []
        if self.file_size_bytes == 0:
            issues.append("empty-file")
        if self.page_count == 0:
            issues.append("no-pages")
        if self.extracted_word_count == 0:
            issues.append("no-text-extracted")
        if self.chunk_count == 0:
            issues.append("no-chunks")
        if self.page_count >= SCAN_LIKE_MIN_PAGES and self.text_page_ratio < MIN_TEXT_PAGE_RATIO:
            issues.append(
                f"low-text-page-ratio={self.text_page_ratio:.2f} (< {MIN_TEXT_PAGE_RATIO:.2f})"
            )
        if self.page_count >= SCAN_LIKE_MIN_PAGES and self.words_per_page < MIN_WORDS_PER_PAGE:
            issues.append(
                f"low-words-per-page={self.words_per_page:.1f} (< {MIN_WORDS_PER_PAGE:.1f})"
            )
        return issues

    @property
    def is_suspicious(self) -> bool:
        return bool(self.issues)


def extract_pdf_chunks(
    pdf_path: str | Path,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> tuple[PdfAuditResult, list[str]]:
    path = Path(pdf_path)
    audit, chunks = _build_audit(path, _extract_page_texts_with_fitz(path), chunk_size, overlap, "fitz")
    if audit.is_suspicious:
        ocr_page_texts = _extract_page_texts_with_ocr(path)
        if ocr_page_texts is not None:
            ocr_audit, ocr_chunks = _build_audit(
                path,
                ocr_page_texts,
                chunk_size,
                overlap,
                "ocr",
            )
            if ocr_audit.extracted_word_count > audit.extracted_word_count:
                return ocr_audit, ocr_chunks
    return audit, chunks


def extract_pdf_page_texts(
    pdf_path: str | Path,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[str]:
    path = Path(pdf_path)
    return _extract_pdf_page_texts(path, page_start=page_start, page_end=page_end)


def _extract_pdf_page_texts(
    path: Path,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[str]:
    fitz_page_texts = _extract_page_texts_with_fitz(path, page_start=page_start, page_end=page_end)
    fitz_audit, _ = _build_audit(path, fitz_page_texts, 500, 50, "fitz")
    if fitz_audit.is_suspicious:
        ocr_page_texts = _extract_page_texts_with_ocr(path, page_start=page_start, page_end=page_end)
        if ocr_page_texts is not None:
            ocr_audit, _ = _build_audit(path, ocr_page_texts, 500, 50, "ocr")
            if ocr_audit.extracted_word_count > fitz_audit.extracted_word_count:
                return ocr_page_texts
    return fitz_page_texts


def _build_audit(
    path: Path,
    page_texts: list[str],
    chunk_size: int,
    overlap: int,
    extraction_method: str,
) -> tuple[PdfAuditResult, list[str]]:
    nonempty_page_count = sum(1 for text in page_texts if text)
    full_text = "\n".join(text for text in page_texts if text)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    audit = PdfAuditResult(
        pdf_path=path,
        file_size_bytes=path.stat().st_size,
        page_count=len(page_texts),
        nonempty_page_count=nonempty_page_count,
        extracted_word_count=len(full_text.split()),
        chunk_count=len(chunks),
        extraction_method=extraction_method,
    )
    return audit, chunks


def _extract_page_texts_with_fitz(
    pdf_path: Path,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[str]:
    doc = fitz.open(str(pdf_path))
    page_indices = _page_indices(doc.page_count, page_start=page_start, page_end=page_end)
    page_texts = [doc[idx].get_text().strip() for idx in page_indices]
    doc.close()
    return page_texts


def _extract_page_texts_with_ocr(
    pdf_path: Path,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[str] | None:
    if not _ocr_available():
        return None

    doc = fitz.open(str(pdf_path))
    page_indices = _page_indices(doc.page_count, page_start=page_start, page_end=page_end)
    doc.close()
    if not page_indices:
        return []

    page_texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rag-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        for page_idx in page_indices:
            page_number = page_idx + 1
            image_prefix = temp_path / f"page-{page_number:04d}"
            image_path = image_prefix.with_suffix(".png")
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-png",
                    "-singlefile",
                    str(pdf_path),
                    str(image_prefix),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            result = subprocess.run(
                [
                    "tesseract",
                    str(image_path),
                    "stdout",
                    "-l",
                    "rus+eng",
                    "--psm",
                    "6",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            page_texts.append(result.stdout.strip())
    return page_texts


def _page_indices(
    page_count: int,
    *,
    page_start: int | None,
    page_end: int | None,
) -> list[int]:
    start_idx = 0 if page_start is None else max(page_start - 1, 0)
    end_idx = page_count if page_end is None else min(page_end, page_count)
    return list(range(start_idx, end_idx))


def _ocr_available() -> bool:
    return shutil.which("pdftoppm") is not None and shutil.which("tesseract") is not None


def audit_disciplines(
    disciplines_dir: str | Path,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> dict[str, list[PdfAuditResult]]:
    base_dir = Path(disciplines_dir)
    results: dict[str, list[PdfAuditResult]] = {}
    for discipline_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        audits: list[PdfAuditResult] = []
        for pdf_path in sorted(discipline_dir.glob("*.pdf")):
            audit, _ = extract_pdf_chunks(pdf_path, chunk_size=chunk_size, overlap=overlap)
            audits.append(audit)
        results[discipline_dir.name] = audits
    return results


def assert_pdf_is_searchable(pdf_path: str | Path) -> PdfAuditResult:
    audit, _ = extract_pdf_chunks(pdf_path)
    if audit.is_suspicious:
        raise ValueError(
            f"PDF {audit.pdf_path.name} is not searchable enough for RAG: "
            + ", ".join(audit.issues)
        )
    return audit
