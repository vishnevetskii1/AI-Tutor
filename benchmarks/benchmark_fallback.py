#!/usr/bin/env python3
"""
Benchmark comparing fallback search algorithms for the RAG module.

Algorithms:
  1. Baseline  — current custom keyword scorer (search.py)
  2. BM25Okapi — classic BM25 with TF saturation + IDF (requires: rank-bm25)
  3. BM25L     — BM25L variant, softer TF saturation for long docs (requires: rank-bm25)
  4. BM25Morph — BM25Okapi + pymorphy3 Russian lemmatization (requires: rank-bm25, pymorphy3)

Evaluation — self-retrieval benchmark:
  For each sampled chunk a query is generated from its most frequent terms.
  Metrics measure whether the source chunk appears in the top-k results.
  This gives an upper-bound on absolute performance but a fair relative comparison.

  Known limitation: BM25Morph advantage is underestimated here because synthetic
  queries are generated from the same token forms as the chunk text. Real student
  queries use varied Russian word forms — BM25Morph will outperform more in production.

Install optional dependencies:
  uv add --optional bench rank-bm25 pymorphy3

Usage:
  uv run python benchmark_fallback.py <discipline>
  uv run python benchmark_fallback.py math --samples 200 --seed 42 --top-k 10
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi, BM25L  # type: ignore
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import pymorphy3  # type: ignore
    HAS_PYMORPHY = True
except ImportError:
    HAS_PYMORPHY = False

from bot.rag.pdf_audit import extract_pdf_chunks

# ── Constants ─────────────────────────────────────────────────────────────────

STOPWORDS = {
    "и", "или", "над", "при", "для", "это", "что", "как", "где", "если",
    "они", "она", "оно", "них", "ними", "его", "ее", "её", "the",
}

ChunkRecord = tuple[str, str]  # (source_file, text)

# ── Corpus loader ─────────────────────────────────────────────────────────────

def load_chunks(disciplines_dir: str, discipline: str) -> list[ChunkRecord]:
    discipline_dir = Path(disciplines_dir) / discipline
    if not discipline_dir.exists():
        return []
    chunks: list[ChunkRecord] = []
    for pdf_path in sorted(discipline_dir.glob("*.pdf")):
        try:
            audit, pdf_chunks = extract_pdf_chunks(pdf_path, chunk_size=220, overlap=40)
        except Exception as exc:
            print(f"  Warning: failed to read {pdf_path.name}: {exc}")
            continue
        if audit.is_suspicious:
            print(f"  Skipping suspicious PDF {pdf_path.name}: {', '.join(audit.issues)}")
            continue
        chunks.extend((pdf_path.name, chunk) for chunk in pdf_chunks)
    return chunks

# ── Tokenizers ────────────────────────────────────────────────────────────────

def _tokenize_simple(text: str) -> list[str]:
    return [
        t.casefold()
        for t in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", text or "")
    ]

def _tokenize_filtered(text: str) -> list[str]:
    return [t for t in _tokenize_simple(text) if t not in STOPWORDS]

_morph_analyzer = None

def _tokenize_morph(text: str) -> list[str]:
    global _morph_analyzer
    if _morph_analyzer is None:
        _morph_analyzer = pymorphy3.MorphAnalyzer()
    tokens = _tokenize_simple(text)
    result = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        parses = _morph_analyzer.parse(token)
        lemma = parses[0].normal_form if parses else token
        result.append(lemma)
    return result

# ── Query generator ───────────────────────────────────────────────────────────

def _generate_query(chunk: str, n_terms: int = 4) -> str:
    """Extracts the n most frequent non-stopword terms (≥4 chars) from a chunk."""
    terms = [t for t in _tokenize_filtered(chunk) if len(t) >= 4]
    if not terms:
        return ""
    freq = Counter(terms)
    return " ".join(term for term, _ in freq.most_common(n_terms))

# ── Searcher implementations ──────────────────────────────────────────────────

class BaselineSearcher:
    """Current keyword scorer from bot/rag/search.py — reproduces _score_chunk logic."""

    name = "Baseline (current)"

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[str]:
        query_terms = {t for t in _tokenize_simple(query) if t not in STOPWORDS}
        ordered_terms = [t for t in _tokenize_simple(query) if t not in STOPWORDS]
        lowered_query = query.casefold()

        scored: list[tuple[int, str]] = []
        for _, chunk in self._chunks:
            if not chunk:
                continue
            lowered = chunk.casefold()
            chunk_terms = set(_tokenize_simple(chunk))
            overlap = len(query_terms & chunk_terms)
            if overlap == 0 and lowered_query not in lowered:
                continue

            score = overlap * 4
            if lowered_query in lowered:
                score += 20
            for left, right in zip(ordered_terms, ordered_terms[1:]):
                if f"{left} {right}" in lowered:
                    score += 6
            if (lowered.startswith("содержание")
                    or lowered.count("глава") >= 3
                    or lowered.count("§") >= 3
                    or ". . ." in lowered):
                score -= 12

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        result: list[str] = []
        for _, chunk in scored:
            if chunk not in seen:
                seen.add(chunk)
                result.append(chunk)
            if len(result) >= top_k:
                break
        return result


class BM25OkapiSearcher:
    """Standard BM25Okapi: probabilistic TF saturation + IDF weighting."""

    name = "BM25Okapi"

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._texts = [text for _, text in chunks]
        tokenized = [_tokenize_filtered(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        tokens = _tokenize_filtered(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        seen: set[str] = set()
        result: list[str] = []
        for i in indices:
            if scores[i] <= 0:
                break
            text = self._texts[i]
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result


class BM25LSearcher:
    """BM25L: lower TF saturation ceiling — better for longer documents."""

    name = "BM25L"

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._texts = [text for _, text in chunks]
        tokenized = [_tokenize_filtered(t) for t in self._texts]
        self._bm25 = BM25L(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        tokens = _tokenize_filtered(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        seen: set[str] = set()
        result: list[str] = []
        for i in indices:
            if scores[i] <= 0:
                break
            text = self._texts[i]
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result


class BM25MorphSearcher:
    """BM25Okapi + pymorphy3 lemmatization: matches Russian word forms across cases."""

    name = "BM25+Morph"

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._texts = [text for _, text in chunks]
        print("    Lemmatizing corpus... ", end="", flush=True)
        tokenized = [_tokenize_morph(t) for t in self._texts]
        print("done")
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        tokens = _tokenize_morph(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        seen: set[str] = set()
        result: list[str] = []
        for i in indices:
            if scores[i] <= 0:
                break
            text = self._texts[i]
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result

# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(
    searcher: BaselineSearcher | BM25OkapiSearcher | BM25LSearcher | BM25MorphSearcher,
    samples: list[str],
    top_k: int,
) -> dict[str, float]:
    hits = {1: 0, 3: 0, 5: 0, top_k: 0}
    rr_sum = 0.0
    evaluated = 0

    for chunk_text in samples:
        query = _generate_query(chunk_text)
        if not query:
            continue

        results = searcher.search(query, top_k=top_k)
        rank: int | None = None
        for i, result in enumerate(results, 1):
            if result.strip() == chunk_text.strip():
                rank = i
                break

        if rank is not None:
            for k in hits:
                if rank <= k:
                    hits[k] += 1
            rr_sum += 1.0 / rank

        evaluated += 1

    if evaluated == 0:
        return {f"Hit@{k}": 0.0 for k in [1, 3, 5, top_k]} | {"MRR": 0.0, "n": 0}

    return {
        "Hit@1":      hits[1]    / evaluated,
        "Hit@3":      hits[3]    / evaluated,
        "Hit@5":      hits[5]    / evaluated,
        f"Hit@{top_k}": hits[top_k] / evaluated,
        "MRR":        rr_sum     / evaluated,
        "n":          evaluated,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RAG fallback search algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("discipline", help="Discipline slug (e.g. math)")
    parser.add_argument("--samples", type=int, default=150, help="Chunks to sample for evaluation")
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed")
    parser.add_argument("--top-k",  type=int, default=10,  help="Retrieval depth")
    args = parser.parse_args()

    disciplines_dir = os.getenv("DISCIPLINES_DIR", "data/disciplines")

    print(f"\nLoading chunks: discipline={args.discipline!r}, dir={disciplines_dir!r}")
    chunks = load_chunks(disciplines_dir, args.discipline)
    if not chunks:
        print(f"ERROR: no readable chunks found for discipline '{args.discipline}'.")
        print(f"       Check DISCIPLINES_DIR in .env and that PDFs exist there.")
        sys.exit(1)

    random.seed(args.seed)
    sample_size = min(args.samples, len(chunks))
    sample_texts = [text for _, text in random.sample(chunks, sample_size)]

    print(f"Chunks: {len(chunks)} | Samples: {sample_size} | Seed: {args.seed} | Top-K: {args.top_k}\n")

    # ── Build searchers ────────────────────────────────────────────────────────

    searchers = []

    print("Building Baseline...")
    searchers.append(BaselineSearcher(chunks))

    if HAS_BM25:
        print("Building BM25Okapi...")
        searchers.append(BM25OkapiSearcher(chunks))
        print("Building BM25L...")
        searchers.append(BM25LSearcher(chunks))
    else:
        print("  rank-bm25 not installed — skipping BM25Okapi and BM25L")
        print("  Install: uv add rank-bm25")

    if HAS_BM25 and HAS_PYMORPHY:
        print("Building BM25+Morph...")
        searchers.append(BM25MorphSearcher(chunks))
    elif HAS_BM25 and not HAS_PYMORPHY:
        print("  pymorphy3 not installed — skipping BM25+Morph")
        print("  Install: uv add pymorphy3")

    # ── Run evaluation ─────────────────────────────────────────────────────────

    top_k = args.top_k
    metric_keys = ["Hit@1", "Hit@3", "Hit@5", f"Hit@{top_k}", "MRR", "Time(s)"]
    col_algo  = 20
    col_metric = 10

    print()
    header = f"{'Algorithm':<{col_algo}}" + "".join(f"{k:>{col_metric}}" for k in metric_keys)
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    results: list[tuple[str, dict, float]] = []
    for searcher in searchers:
        t0 = time.perf_counter()
        metrics = _evaluate(searcher, sample_texts, top_k)
        elapsed = time.perf_counter() - t0
        results.append((searcher.name, metrics, elapsed))

        row = f"{searcher.name:<{col_algo}}"
        for k in metric_keys[:-1]:
            row += f"{metrics.get(k, 0.0):>{col_metric}.3f}"
        row += f"{elapsed:>{col_metric}.1f}"
        print(row)

    print(sep)

    # ── Summary ────────────────────────────────────────────────────────────────

    if len(results) > 1:
        best_name, best_metrics, _ = max(results, key=lambda r: r[1]["MRR"])
        baseline_metrics = results[0][1]
        mrr_gain = best_metrics["MRR"] - baseline_metrics["MRR"]
        hit5_gain = best_metrics.get("Hit@5", 0) - baseline_metrics.get("Hit@5", 0)

        print(f"\nBest algorithm by MRR: {best_name}")
        if best_name != results[0][0]:
            print(f"  MRR gain vs Baseline:   +{mrr_gain:.3f}")
            print(f"  Hit@5 gain vs Baseline: +{hit5_gain:.3f}")

    print()
    print("Note: self-retrieval benchmark overestimates absolute scores but")
    print("gives a fair relative comparison. BM25+Morph advantage is")
    print("underestimated here — real Russian queries use varied word forms.")


if __name__ == "__main__":
    main()
