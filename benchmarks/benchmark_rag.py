#!/usr/bin/env python3
"""
Full RAG pipeline benchmark — compares six retrieval architectures end-to-end.

Architectures:
  1. Baseline       — current pipeline: dense + keyword fallback + _rerank_chunks
  2. DenseOnly      — cosine-similarity retrieval only, no re-ranking
  3. BM25Only       — BM25Okapi, no vectors
  4. HybridRRF      — Reciprocal Rank Fusion of dense + BM25
  5. HybridRRF+CE   — HybridRRF candidates re-ranked by a cross-encoder
  6. HybridRRF+MMR  — HybridRRF candidates diversified by Maximum Marginal Relevance

All architectures operate fully in-memory (no Qdrant required).
Dense vectors are pre-computed once with all-MiniLM-L6-v2 and shared.

Evaluation: self-retrieval benchmark (same as benchmark_fallback.py) + Diversity score.
  For each sampled chunk, a query is generated from its most frequent terms.
  Diversity = 1 - avg pairwise cosine similarity of the top-k results.

Install dependencies:
  uv add rank-bm25      # needed for BM25Only, HybridRRF, HybridRRF+CE, HybridRRF+MMR
  # cross-encoder model is downloaded automatically by sentence-transformers

Usage:
  uv run python benchmark_rag.py <discipline>
  uv run python benchmark_rag.py math --samples 100 --top-k 5 --skip-ce
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_ST = True
except ImportError:
    HAS_ST = False

from bot.rag.pdf_audit import extract_pdf_chunks
from bot.rag.search import _rerank_chunks  # safe: does not call load_config()

# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE, OVERLAP = 220, 40   # consistent with fallback benchmark

STOPWORDS = {
    "и", "или", "над", "при", "для", "это", "что", "как", "где", "если",
    "они", "она", "оно", "них", "ними", "его", "ее", "её", "the",
}

ChunkRecord = tuple[str, str]  # (source_file, text)

# ── Lazy model singletons ──────────────────────────────────────────────────────

_embedding_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# ── Corpus loader ──────────────────────────────────────────────────────────────

def load_chunks(disciplines_dir: str, discipline: str) -> list[ChunkRecord]:
    discipline_dir = Path(disciplines_dir) / discipline
    if not discipline_dir.exists():
        return []
    chunks: list[ChunkRecord] = []
    for pdf_path in sorted(discipline_dir.glob("*.pdf")):
        try:
            audit, pdf_chunks = extract_pdf_chunks(
                pdf_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP
            )
        except Exception as exc:
            print(f"  Warning: {pdf_path.name}: {exc}")
            continue
        if audit.is_suspicious:
            print(f"  Skipping {pdf_path.name}: {', '.join(audit.issues)}")
            continue
        chunks.extend((pdf_path.name, chunk) for chunk in pdf_chunks)
    return chunks


# ── Tokeniser & query generator ───────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [t.casefold() for t in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", text or "")]


def _tokenize_filtered(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in STOPWORDS]


def _generate_query(chunk: str, n_terms: int = 4) -> str:
    terms = [t for t in _tokenize_filtered(chunk) if len(t) >= 4]
    if not terms:
        return ""
    return " ".join(t for t, _ in Counter(terms).most_common(n_terms))


def query_length_category(query: str) -> str:
    n = len(_tokenize_filtered(query))
    if n <= 2:
        return "short (1-2)"
    if n <= 4:
        return "medium (3-4)"
    return "long (5+)"


# ── Shared pre-computed resources ─────────────────────────────────────────────

class SharedResources:
    """
    Builds dense vectors and BM25 index once.
    All architectures share the same underlying data — no redundant computation.
    """

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks  = chunks
        self.texts   = [t for _, t in chunks]
        self.sources = [s for s, _ in chunks]
        self._vectors:     np.ndarray | None  = None
        self._bm25:        BM25Okapi | None   = None
        self._text_to_idx: dict[str, int] | None = None

    @property
    def vectors(self) -> np.ndarray:
        """L2-normalised dense vectors, shape (N, 384). Cosine sim = dot product."""
        if self._vectors is None:
            model = _get_embedding_model()
            print(f"  Encoding {len(self.texts)} chunks with {EMBEDDING_MODEL}...", flush=True)
            raw = model.encode(
                self.texts,
                show_progress_bar=True,
                batch_size=64,
                normalize_embeddings=True,
            ).astype(np.float32)
            self._vectors = raw
        return self._vectors

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            if not HAS_BM25:
                raise RuntimeError("rank-bm25 not installed. Run: pip install rank-bm25")
            print("  Building BM25 index...", flush=True)
            self._bm25 = BM25Okapi([_tokenize_filtered(t) for t in self.texts])
        return self._bm25

    @property
    def text_to_idx(self) -> dict[str, int]:
        if self._text_to_idx is None:
            self._text_to_idx = {t: i for i, t in enumerate(self.texts)}
        return self._text_to_idx


# ── Low-level retrieval primitives ────────────────────────────────────────────

def _cosine_top_k(shared: SharedResources, query: str, k: int) -> list[int]:
    q_vec = _get_embedding_model().encode(
        query, normalize_embeddings=True
    ).astype(np.float32)
    sims = shared.vectors @ q_vec          # (N,) — dot product = cosine sim
    return np.argsort(sims)[::-1][:k].tolist()


def _bm25_top_k(shared: SharedResources, query: str, k: int) -> list[int]:
    scores = np.array(shared.bm25.get_scores(_tokenize_filtered(query)))
    return np.argsort(scores)[::-1][:k].tolist()


def _rrf_merge(rank_lists: list[list[int]], rrf_k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion. Returns indices sorted by descending RRF score."""
    scores: dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=lambda x: -scores[x])


def _mmr_select(
    shared: SharedResources,
    query: str,
    candidate_indices: list[int],
    top_k: int,
    lambda_: float = 0.5,
) -> list[int]:
    """
    Maximum Marginal Relevance.
    Iteratively picks the candidate that maximises:
        lambda * sim(query, doc) - (1 - lambda) * max_sim(doc, already_selected)
    """
    if not candidate_indices:
        return []

    q_vec  = _get_embedding_model().encode(query, normalize_embeddings=True).astype(np.float32)
    vecs   = shared.vectors[candidate_indices]     # (M, D), already normalised
    q_sims = vecs @ q_vec                          # (M,) relevance scores

    selected: list[int] = []   # local indices into candidate_indices
    remaining = list(range(len(candidate_indices)))

    while len(selected) < top_k and remaining:
        if not selected:
            best_local = remaining[int(np.argmax(q_sims[remaining]))]
        else:
            sel_vecs = vecs[selected]              # (S, D)
            scores   = np.empty(len(remaining))
            for j, loc in enumerate(remaining):
                redundancy   = float(np.max(sel_vecs @ vecs[loc]))
                scores[j]    = lambda_ * q_sims[loc] - (1 - lambda_) * redundancy
            best_local = remaining[int(np.argmax(scores))]

        selected.append(best_local)
        remaining.remove(best_local)

    return [candidate_indices[i] for i in selected]


# ── Diversity metric ──────────────────────────────────────────────────────────

def diversity_score(shared: SharedResources, result_texts: list[str]) -> float:
    """1 - avg pairwise cosine similarity. 1.0 = perfectly diverse, 0.0 = all identical."""
    indices = [shared.text_to_idx[t] for t in result_texts if t in shared.text_to_idx]
    if len(indices) < 2:
        return 1.0
    vecs = shared.vectors[indices]          # already L2-normalised
    gram = vecs @ vecs.T                    # cosine-sim matrix
    np.fill_diagonal(gram, 0.0)
    n    = len(indices)
    return float(1.0 - gram.sum() / (n * (n - 1)))


# ── Architecture 1: Baseline ──────────────────────────────────────────────────

class BaselineRAG:
    """
    Mirrors current search.py logic, fully in-memory:
    dense candidates + BM25 fallback candidates → merged → _rerank_chunks.
    """
    name = "Baseline (current)"

    def __init__(self, shared: SharedResources) -> None:
        self._s = shared

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        pool = top_k * 4
        dense_idx = _cosine_top_k(self._s, query, pool)
        bm25_idx  = _bm25_top_k(self._s, query, pool) if HAS_BM25 else []

        seen: set[int] = set()
        merged: list[int] = []
        for idx in dense_idx + bm25_idx:
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)

        records = [(self._s.sources[i], self._s.texts[i]) for i in merged]
        return _rerank_chunks(query, records, top_k)


# ── Architecture 2: Dense Only ────────────────────────────────────────────────

class DenseOnlyRAG:
    """Pure cosine-similarity retrieval — no keyword fallback, no re-ranking."""
    name = "DenseOnly"

    def __init__(self, shared: SharedResources) -> None:
        self._s = shared

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        indices = _cosine_top_k(self._s, query, top_k * 2)
        seen: set[str] = set()
        result: list[str] = []
        for idx in indices:
            text = self._s.texts[idx]
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result


# ── Architecture 3: BM25 Only ─────────────────────────────────────────────────

class BM25OnlyRAG:
    """Pure BM25Okapi retrieval — no dense vectors, no re-ranking."""
    name = "BM25Only"

    def __init__(self, shared: SharedResources) -> None:
        self._s = shared

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        tokens = _tokenize_filtered(query)
        if not tokens:
            return []
        scores  = np.array(self._s.bm25.get_scores(tokens))
        indices = np.argsort(scores)[::-1]
        seen: set[str] = set()
        result: list[str] = []
        for idx in indices:
            if scores[idx] <= 0:
                break
            text = self._s.texts[idx]
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result


# ── Architecture 4: Hybrid RRF ────────────────────────────────────────────────

class HybridRRFRAG:
    """
    Reciprocal Rank Fusion of dense + BM25.
    RRF score = Σ 1/(k + rank_i) — no tuning required, proven across benchmarks.
    """
    name = "HybridRRF"

    def __init__(self, shared: SharedResources, pool_factor: int = 6) -> None:
        self._s    = shared
        self._pool = pool_factor

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        pool = top_k * self._pool
        dense_ranked = _cosine_top_k(self._s, query, pool)
        bm25_ranked  = _bm25_top_k(self._s, query, pool)
        rrf_ranked   = _rrf_merge([dense_ranked, bm25_ranked])

        seen: set[str] = set()
        result: list[str] = []
        for idx in rrf_ranked:
            text = self._s.texts[idx]
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result


# ── Architecture 5: Hybrid RRF + Cross-Encoder ───────────────────────────────

class HybridRRFCrossEncoderRAG:
    """
    HybridRRF retrieves a wider candidate pool; a cross-encoder re-ranks by
    query-document joint score (much more accurate than bi-encoder similarity).

    Note: uses cross-encoder/ms-marco-MiniLM-L-6-v2 (English model).
    For Russian production use, replace with a Russian-specific cross-encoder.
    """
    name = "HybridRRF+CE"

    def __init__(self, shared: SharedResources, n_candidates: int = 20) -> None:
        self._s    = shared
        self._n    = n_candidates

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        pool       = max(self._n, top_k * 4)
        dense      = _cosine_top_k(self._s, query, pool)
        bm25       = _bm25_top_k(self._s, query, pool)
        candidates = _rrf_merge([dense, bm25])[:pool]

        texts      = [self._s.texts[i] for i in candidates]
        ce_scores  = _get_cross_encoder().predict([(query, t) for t in texts])

        ranked = sorted(zip(ce_scores, texts), key=lambda x: -x[0])
        seen: set[str] = set()
        result: list[str] = []
        for _, text in ranked:
            if text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= top_k:
                break
        return result


# ── Architecture 6: Hybrid RRF + MMR ─────────────────────────────────────────

class HybridRRFMMRRAG:
    """
    HybridRRF retrieval followed by Maximum Marginal Relevance selection.
    MMR trades a small amount of relevance for higher result diversity —
    useful when the LLM context window must cover different subtopics.
    lambda_=0.5 balances relevance and diversity equally.
    """
    name = "HybridRRF+MMR"

    def __init__(self, shared: SharedResources, lambda_: float = 0.5, pool_factor: int = 6) -> None:
        self._s       = shared
        self._lambda  = lambda_
        self._pool    = pool_factor

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        pool       = top_k * self._pool
        dense      = _cosine_top_k(self._s, query, pool)
        bm25       = _bm25_top_k(self._s, query, pool)
        candidates = _rrf_merge([dense, bm25])[:pool]

        selected   = _mmr_select(self._s, query, candidates, top_k, lambda_=self._lambda)
        return [self._s.texts[i] for i in selected]


# ── LLM-as-Judge evaluation ───────────────────────────────────────────────────

def evaluate_with_judge(
    arch: BaselineRAG | DenseOnlyRAG | BM25OnlyRAG | HybridRRFRAG
         | HybridRRFCrossEncoderRAG | HybridRRFMMRRAG,
    shared: SharedResources,
    samples: list[str],
    top_k: int,
    judge: object,                    # llm_judge.LLMJudge — kept untyped to avoid circular import
    compute_sufficiency: bool = True,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Evaluate a RAG architecture using LLM-as-judge metrics.

    The judge rates each retrieved chunk on a 0-3 relevance scale and optionally
    rates the entire context for sufficiency (0-2).  All calls are batched (one
    LLM request per query) and cached inside the judge object.

    Returns:
        LLM-Prec@k     — fraction of top-k chunks rated relevant (grade ≥ 2)
        LLM-NDCG@k     — Normalised DCG using grades as relevance weights
        LLM-MRR        — MRR where "hit" = first chunk with grade ≥ 2
        LLM-Sufficiency — mean normalised sufficiency (0-1), if compute_sufficiency
        n              — number of samples actually evaluated
    """
    from llm_judge import ndcg as _ndcg   # lazy import — llm_judge is optional

    all_grades:      list[list[int]] = []
    all_sufficiency: list[int]       = []
    evaluated = 0

    for i, chunk_text in enumerate(samples):
        query = _generate_query(chunk_text)
        if not query:
            continue

        results = arch.retrieve(query, top_k=top_k)
        if not results:
            continue

        try:
            grades = judge.rate_chunks(query, results)
            all_grades.append(grades)

            if compute_sufficiency:
                suff = judge.rate_sufficiency(query, results)
                all_sufficiency.append(suff)

            evaluated += 1
            if verbose:
                print(f"  [{i + 1}/{len(samples)}] grades={grades}")

        except Exception as exc:
            logger.warning("Judge evaluation failed for sample %d: %s", i, exc)

    if evaluated == 0:
        return {}

    # LLM-Precision@k: fraction of top-k chunks rated relevant (grade >= 2)
    prec = sum(
        sum(g >= 2 for g in gs) / len(gs) for gs in all_grades
    ) / evaluated

    # LLM-NDCG@k: normalised DCG using 0-3 grades as relevance
    avg_ndcg = sum(_ndcg(gs) for gs in all_grades) / evaluated

    # LLM-MRR: reciprocal rank of first chunk with grade >= 2
    rr_sum = 0.0
    for gs in all_grades:
        for rank, g in enumerate(gs, 1):
            if g >= 2:
                rr_sum += 1.0 / rank
                break

    out: dict[str, float] = {
        "LLM-Prec@k": prec,
        "LLM-NDCG@k": avg_ndcg,
        "LLM-MRR":    rr_sum / evaluated,
        "n":          float(evaluated),
    }
    if all_sufficiency:
        # Normalise 0-2 → 0-1
        out["LLM-Sufficiency"] = sum(s / 2 for s in all_sufficiency) / len(all_sufficiency)

    return out


# ── Self-retrieval evaluation ──────────────────────────────────────────────────

def evaluate(
    arch: BaselineRAG | DenseOnlyRAG | BM25OnlyRAG | HybridRRFRAG
         | HybridRRFCrossEncoderRAG | HybridRRFMMRRAG,
    shared: SharedResources,
    samples: list[str],
    top_k: int = 10,
    compute_diversity: bool = True,
) -> dict[str, float]:
    """
    Self-retrieval benchmark.
    Returns: Hit@{1,3,5,top_k}, MRR, Diversity (optional), n.
    """
    hits       = {1: 0, 3: 0, 5: 0, top_k: 0}
    rr_sum     = 0.0
    div_scores: list[float] = []
    evaluated  = 0

    for chunk_text in samples:
        query = _generate_query(chunk_text)
        if not query:
            continue

        results = arch.retrieve(query, top_k=top_k)

        rank: int | None = None
        for i, r in enumerate(results, 1):
            if r.strip() == chunk_text.strip():
                rank = i
                break

        if rank is not None:
            for k in hits:
                if rank <= k:
                    hits[k] += 1
            rr_sum += 1.0 / rank

        if compute_diversity and len(results) >= 2:
            div_scores.append(diversity_score(shared, results))

        evaluated += 1

    if evaluated == 0:
        return {}

    out: dict[str, float] = {
        "Hit@1":         hits[1]     / evaluated,
        "Hit@3":         hits[3]     / evaluated,
        "Hit@5":         hits[5]     / evaluated,
        f"Hit@{top_k}":  hits[top_k] / evaluated,
        "MRR":           rr_sum      / evaluated,
        "n":             float(evaluated),
    }
    if div_scores:
        out["Diversity"] = float(np.mean(div_scores))
    return out


def evaluate_by_query_length(
    arch: BaselineRAG | DenseOnlyRAG | BM25OnlyRAG | HybridRRFRAG
         | HybridRRFCrossEncoderRAG | HybridRRFMMRRAG,
    shared: SharedResources,
    samples: list[str],
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Run evaluate() separately for short / medium / long queries.
    Returns a dict: category → metrics dict.
    """
    buckets: dict[str, list[str]] = {"short (1-2)": [], "medium (3-4)": [], "long (5+)": []}
    for chunk_text in samples:
        q    = _generate_query(chunk_text)
        cat  = query_length_category(q) if q else "short (1-2)"
        buckets[cat].append(chunk_text)

    return {
        cat: evaluate(arch, shared, texts, top_k=top_k, compute_diversity=False)
        for cat, texts in buckets.items()
        if texts
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full RAG pipeline architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("discipline", help="Discipline slug (e.g. math)")
    parser.add_argument("--samples",  type=int,  default=100, help="Evaluation sample size")
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--top-k",    type=int,  default=5)
    parser.add_argument("--skip-ce",  action="store_true",
                        help="Skip HybridRRF+CE (avoids cross-encoder download)")
    args = parser.parse_args()

    disciplines_dir = os.getenv("DISCIPLINES_DIR", "data/disciplines")

    print(f"\nLoading corpus: discipline={args.discipline!r}")
    chunks = load_chunks(disciplines_dir, args.discipline)
    if not chunks:
        print(f"ERROR: no readable chunks found for '{args.discipline}'.")
        sys.exit(1)

    sources = {s for s, _ in chunks}
    print(f"Chunks: {len(chunks)}  |  PDFs: {len(sources)}")

    if not HAS_ST:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    # ── Build shared resources ─────────────────────────────────────────────────
    shared = SharedResources(chunks)
    print()
    _ = shared.vectors          # triggers dense encoding (slow — done once)
    if HAS_BM25:
        _ = shared.bm25         # triggers BM25 build (fast)
    else:
        print("WARNING: rank-bm25 not installed → BM25-based architectures skipped.")

    # ── Assemble architectures ────────────────────────────────────────────────
    architectures = [BaselineRAG(shared), DenseOnlyRAG(shared)]

    if HAS_BM25:
        architectures.append(BM25OnlyRAG(shared))
        architectures.append(HybridRRFRAG(shared))
        architectures.append(HybridRRFMMRRAG(shared))

    if HAS_BM25 and HAS_ST and not args.skip_ce:
        try:
            print("\nLoading cross-encoder model (download on first run)...")
            _ = _get_cross_encoder()
            architectures.append(HybridRRFCrossEncoderRAG(shared))
        except Exception as exc:
            print(f"  Cross-encoder unavailable: {exc}")

    # ── Sample ────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    sample_size  = min(args.samples, len(chunks))
    sample_texts = [t for _, t in random.sample(chunks, sample_size)]

    top_k = args.top_k
    print(f"\nSamples: {sample_size}  |  Seed: {args.seed}  |  Top-K: {top_k}\n")

    # ── Evaluate & print table ────────────────────────────────────────────────
    col_a, col_m = 22, 11
    keys = ["Hit@1", "Hit@3", "Hit@5", f"Hit@{top_k}", "MRR", "Diversity", "Time(s)"]
    sep  = "─" * (col_a + col_m * len(keys))
    hdr  = f"{'Architecture':<{col_a}}" + "".join(f"{k:>{col_m}}" for k in keys)

    print(sep)
    print(hdr)
    print(sep)

    results = []
    for arch in architectures:
        t0      = time.perf_counter()
        metrics = evaluate(arch, shared, sample_texts, top_k=top_k)
        elapsed = time.perf_counter() - t0
        results.append((arch.name, metrics, elapsed))

        row = f"{arch.name:<{col_a}}"
        for k in keys[:-1]:
            row += f"{metrics.get(k, 0.0):>{col_m}.3f}"
        row += f"{elapsed:>{col_m}.1f}"
        print(row)

    print(sep)

    if len(results) > 1:
        best_name, best_m, _ = max(results, key=lambda r: r[1].get("MRR", 0))
        base_mrr = results[0][1].get("MRR", 0)
        print(f"\nBest by MRR: {best_name}  (Δ vs Baseline = {best_m['MRR'] - base_mrr:+.3f})")

    print()
    print("Note: HybridRRF+CE uses an English cross-encoder — advantage may be underestimated")
    print("      for Russian corpora. Swap model for a Russian cross-encoder in production.")


if __name__ == "__main__":
    main()
