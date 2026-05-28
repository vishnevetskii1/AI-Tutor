#!/usr/bin/env python3
"""
LLM-as-Judge — evaluates RAG retrieval quality with a language model.

Replaces the self-retrieval proxy metric with genuine LLM relevance judgements.
Works standalone: reads LLM provider / API key from environment without requiring
TELEGRAM_BOT_TOKEN.  Supports the same providers as the bot (OpenRouter, Ollama).

── Scoring ──────────────────────────────────────────────────────────────────────

  Relevance grade (per retrieved chunk, 0-3):
    0 = Not relevant — chunk does not address the question
    1 = Marginally relevant — loosely related, not useful
    2 = Relevant — useful context, partially answers the question
    3 = Highly relevant — directly answers or strongly supports the answer

  Sufficiency (entire top-k context, 0-2):
    0 = Insufficient — key information is missing
    1 = Partial — some useful info, but gaps remain
    2 = Sufficient — enough to fully answer the question

── Derived metrics ───────────────────────────────────────────────────────────────

  LLM-Precision@k  fraction of top-k chunks with grade ≥ 2
  LLM-NDCG@k      Normalised Discounted Cumulative Gain (grades as relevance)
  LLM-MRR         MRR where "hit" = first chunk with grade ≥ 2
  LLM-Sufficiency  mean normalised sufficiency (0-1)

── Caching ───────────────────────────────────────────────────────────────────────

  All LLM responses are cached in a local JSON file (default: judge_cache.json).
  Rerunning the benchmark reuses cached judgements — no redundant API calls.

── Cost estimate ─────────────────────────────────────────────────────────────────

  Each query generates ONE batched relevance call (all k chunks at once)
  plus optionally ONE sufficiency call.
  For 20 samples × 6 architectures × 2 calls ≈ 240 calls.
  At Gemini Flash rates (~$0.075/M tokens) with ~300 tokens/call ≈ $0.005 total.

Usage:
  from llm_judge import LLMJudge
  judge = LLMJudge.from_env()
  grades = judge.rate_chunks("What is a derivative?", ["chunk1 text", "chunk2 text"])
  suff   = judge.rate_sufficiency("What is a derivative?", ["chunk1 text", "chunk2 text"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_RELEVANCE = (
    "You are a relevance judge for an educational RAG (Retrieval-Augmented Generation) system.\n"
    "Your task: given a student question and a list of retrieved text chunks, rate how relevant\n"
    "each chunk is to the question.\n\n"
    "Relevance scale:\n"
    "  0 = Not relevant — the chunk does not address the question at all\n"
    "  1 = Marginally relevant — loosely related but not useful for answering\n"
    "  2 = Relevant — provides useful context or partially addresses the question\n"
    "  3 = Highly relevant — directly answers or strongly supports answering the question\n\n"
    "Rules:\n"
    "- Respond with EXACTLY N lines (one per chunk), each containing a SINGLE digit (0, 1, 2, or 3).\n"
    "- Do NOT include explanations, labels, bullet points, or any other text.\n"
    "- Assess semantic relevance even if the chunk is in Russian or another language."
)

_TEMPLATE_RELEVANCE = (
    "QUESTION: {query}\n\n"
    "Retrieved chunks ({n} total):\n"
    "{chunks_block}\n\n"
    "Rate each chunk's relevance. Respond with exactly {n} lines, each a single digit 0-3."
)

_SYSTEM_SUFFICIENCY = (
    "You are a RAG quality evaluator for an educational tutor system.\n"
    "Given a student's question and a set of retrieved context passages, decide whether\n"
    "the context collectively contains enough information to answer the question.\n\n"
    "Sufficiency scale:\n"
    "  0 = Insufficient — the context lacks the information needed to answer\n"
    "  1 = Partial — some relevant information is present but key gaps remain\n"
    "  2 = Sufficient — the context contains enough information to answer the question well\n\n"
    "Rules:\n"
    "- Respond with EXACTLY ONE digit (0, 1, or 2). Nothing else.\n"
    "- Evaluate only whether the information is present, not writing quality."
)

_TEMPLATE_SUFFICIENCY = (
    "QUESTION: {query}\n\n"
    "RETRIEVED CONTEXT:\n"
    "{context_block}\n\n"
    "Does this context collectively provide enough information to answer the question?\n"
    "Respond with a single digit: 0 (insufficient), 1 (partial), or 2 (sufficient)."
)

# ── NDCG helper ───────────────────────────────────────────────────────────────

def _dcg(grades: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg(grades: list[int]) -> float:
    """Normalised DCG for a single query. grades are relevance values (0-3)."""
    ideal = sorted(grades, reverse=True)
    idcg  = _dcg(ideal)
    return _dcg(grades) / idcg if idcg > 0 else 0.0


# ── LLMJudge ─────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    LLM-as-Judge for RAG retrieval quality.

    Supports OpenRouter and Ollama.  All responses are cached to a local JSON
    file so repeated benchmark runs never re-query the API for the same inputs.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        ollama_url: str = "http://localhost:11434",
        cache_path: str | Path = "judge_cache.json",
        timeout: float = 45.0,
        max_retries: int = 3,
        chunk_preview: int = 600,    # max chars per chunk sent to the LLM
    ) -> None:
        self.provider      = provider
        self.model         = model
        self.api_key       = api_key
        self.ollama_url    = ollama_url.rstrip("/")
        self.timeout       = timeout
        self.max_retries   = max_retries
        self.chunk_preview = chunk_preview
        self._cache_path   = Path(cache_path)
        self._cache: dict[str, str] = self._load_cache()

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        cache_path: str | Path = "judge_cache.json",
        judge_model: str | None = None,
    ) -> "LLMJudge":
        """
        Create a judge from environment variables.

        Reads: LLM_PROVIDER, OPENROUTER_API_KEY, OLLAMA_URL, OLLAMA_MODEL.
        Set JUDGE_MODEL to override the model used for judging (e.g. a cheaper
        model than the bot's primary generation model).

        Args:
            cache_path:   Path to the JSON cache file.
            judge_model:  Override model name.  None = use JUDGE_MODEL env var,
                          falling back to LLM_MODEL or provider defaults.
        """
        provider   = os.getenv("LLM_PROVIDER", "openrouter")
        api_key    = os.getenv("OPENROUTER_API_KEY", "")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        if judge_model:
            model = judge_model
        elif provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        else:
            # Prefer a dedicated cheap/fast judge model; fall back to primary model
            model = os.getenv(
                "JUDGE_MODEL",
                os.getenv("LLM_MODEL", "google/gemini-2.5-flash"),
            )

        return cls(
            provider   = provider,
            model      = model,
            api_key    = api_key,
            ollama_url = ollama_url,
            cache_path = cache_path,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def rate_chunks(self, query: str, chunks: list[str]) -> list[int]:
        """
        Rate the relevance of each chunk for the given query in ONE LLM call.

        All k chunks are sent together — much cheaper than k separate calls.
        Returns a list of int grades (0-3), same length as chunks.
        Malformed responses are handled gracefully (defaulting to grade 1).
        """
        if not chunks:
            return []

        cache_key = self._key("rel", query, *chunks)
        if cache_key in self._cache:
            return self._parse_grades(self._cache[cache_key], len(chunks))

        chunks_block = "\n\n".join(
            f"[{i + 1}] {c[: self.chunk_preview]}" for i, c in enumerate(chunks)
        )
        prompt = _TEMPLATE_RELEVANCE.format(
            query=query, n=len(chunks), chunks_block=chunks_block
        )

        raw = self._call(prompt, _SYSTEM_RELEVANCE)
        self._cache[cache_key] = raw
        self._save_cache()
        return self._parse_grades(raw, len(chunks))

    def rate_sufficiency(self, query: str, chunks: list[str]) -> int:
        """
        Rate whether the top-k context collectively answers the query (0-2).

        0 = insufficient, 1 = partial, 2 = sufficient.
        """
        if not chunks:
            return 0

        cache_key = self._key("suf", query, *chunks)
        if cache_key in self._cache:
            return self._parse_sufficiency(self._cache[cache_key])

        context_block = "\n---\n".join(
            f"[{i + 1}] {c[: self.chunk_preview]}" for i, c in enumerate(chunks)
        )
        prompt = _TEMPLATE_SUFFICIENCY.format(query=query, context_block=context_block)

        raw = self._call(prompt, _SYSTEM_SUFFICIENCY)
        self._cache[cache_key] = raw
        self._save_cache()
        return self._parse_sufficiency(raw)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def flush_cache(self) -> None:
        """Delete all cached responses from memory and disk."""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()

    # ── LLM backends ─────────────────────────────────────────────────────────

    def _call(self, prompt: str, system: str) -> str:
        """Dispatch to the configured provider with retry + exponential back-off."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "ollama":
                    return self._call_ollama(prompt, system)
                return self._call_openrouter(prompt, system)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Judge LLM call failed (attempt %d/%d): %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(
            f"Judge LLM call failed after {self.max_retries} attempts"
        ) from last_exc

    def _call_openrouter(self, prompt: str, system: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ai-tutor-bot",
            },
            json={
                "model":       self.model,
                "messages":    messages,
                "max_tokens":  80,       # grades are very short
                "temperature": 0.0,      # deterministic → stable scores
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, prompt: str, system: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = httpx.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model":    self.model,
                "messages": messages,
                "stream":   False,
                "options":  {"temperature": 0, "num_predict": 80},
            },
            timeout=httpx.Timeout(self.timeout, connect=5.0),
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_grades(raw: str, n: int) -> list[int]:
        """
        Extract n relevance grades (0-3) from a raw LLM response.

        Strategy:
          1. Try to read one grade per line (the expected format)
          2. Fall back to scanning for all digits 0-3 in the string
          3. Pad with 1 (marginally relevant) if the model returned too few digits
        """
        # Per-line parsing (expected format: "2\n1\n3\n...")
        line_grades: list[int] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and stripped[0] in "0123":
                line_grades.append(int(stripped[0]))

        if len(line_grades) >= n:
            return line_grades[:n]

        # Fallback: scan for any digit 0-3 in the raw string
        all_digits = [int(d) for d in re.findall(r"[0-3]", raw)]
        if len(all_digits) >= n:
            return all_digits[:n]

        # Pad if too few
        base = all_digits if all_digits else []
        return (base + [1] * n)[:n]

    @staticmethod
    def _parse_sufficiency(raw: str) -> int:
        """Extract a single sufficiency grade (0-2) from the raw response."""
        digits = re.findall(r"[0-2]", raw)
        return int(digits[0]) if digits else 1

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _key(self, *parts: str) -> str:
        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def _load_cache(self) -> dict[str, str]:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load judge cache: %s", exc)
        return {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save judge cache: %s", exc)

    def __repr__(self) -> str:
        return (
            f"LLMJudge(provider={self.provider!r}, model={self.model!r}, "
            f"cache={self.cache_size} entries)"
        )
