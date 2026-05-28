# Роль файла: CLI-помощник для проверки качества PDF и RAG.
import sys

from bot.config import load_config
from bot.rag.pdf_audit import audit_disciplines


def main() -> int:
    config = load_config()
    audits = audit_disciplines(config.disciplines_dir)
    suspicious_found = False

    for discipline, pdf_audits in audits.items():
        print(f"\n[{discipline}]")
        if not pdf_audits:
            print("  no pdfs found")
            continue
        for audit in pdf_audits:
            issues = ", ".join(audit.issues) if audit.issues else "ok"
            print(
                "  "
                f"{audit.pdf_path.name}: pages={audit.page_count}, "
                f"text_pages={audit.nonempty_page_count}, "
                f"words={audit.extracted_word_count}, "
                f"chunks={audit.chunk_count}, "
                f"method={audit.extraction_method}, "
                f"issues={issues}"
            )
            suspicious_found = suspicious_found or audit.is_suspicious

    if suspicious_found:
        print("\nRAG audit failed: suspicious PDFs detected.")
        return 1

    print("\nRAG audit passed: all PDFs look searchable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
