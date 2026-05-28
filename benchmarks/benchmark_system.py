#!/usr/bin/env python3
"""
Benchmark — 4-architecture ablation study for QA answer quality.

Proves how much RAG and multi-agent prompt engineering each contribute
to precision metrics in the AI-tutor system.

  ┌─────────────────────┬──────────────┬───────────────┐
  │ Architecture        │ RAG context  │ Agent prompts │
  ├─────────────────────┼──────────────┼───────────────┤
  │ LLM Only            │      ✗       │      ✗        │
  │ LLM + RAG           │      ✓       │      ✗        │
  │ LLM + Agents        │      ✗       │      ✓        │
  │ LLM + RAG + Agents  │      ✓       │      ✓        │
  └─────────────────────┴──────────────┴───────────────┘

LLM-as-Judge rates each answer on three dimensions:
  Correctness  (0–3) — factual accuracy vs. reference / general knowledge
  Completeness (0–2) — whether all aspects of the question are addressed
  Clarity      (0–2) — student-friendly explanation quality

Composite (0–1) = Correctness/3 × 0.50 + Completeness/2 × 0.30 + Clarity/2 × 0.20

All LLM calls (generation + judging) are cached to JSON files.
Reruns do not cost API credits.

Dependencies:
  Required:  httpx, python-dotenv
  Optional:  sentence-transformers (dense RAG), rank-bm25 (BM25 RAG)
             At least one is needed for LLM+RAG architectures.

Usage:
  uv run python benchmark_system.py math --samples 20 --level beginner
  uv run python benchmark_system.py math --questions-file questions.json
  uv run python benchmark_system.py math --samples 20 --level all --output results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean as _mean

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Optional heavy dependencies ───────────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True and HAS_NUMPY  # dense retrieval needs both
except ImportError:
    HAS_ST = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# ── Project imports (safe — none of these call load_config()) ─────────────────

try:
    from benchmark_fallback import load_chunks, _tokenize_filtered, STOPWORDS
    HAS_CORPUS = True
except ImportError:
    HAS_CORPUS = False
    print("⚠  benchmark_fallback.py not found — place it next to this file.")

# ── Architecture names ─────────────────────────────────────────────────────────

ARCH_NAMES = [
    "LLM Only",
    "LLM + RAG",
    "LLM + Agents",
    "LLM + RAG + Agents",
]

# Extended set — includes two fixed architectures that address cold-QA mismatch:
#   "LLM + Agents (Light)"            — lightweight prompt, no in-session restrictions
#   "LLM + RAG + Agents (In-Session)" — pre-generated lesson simulates real tutor flow
ARCH_NAMES_EXTENDED = ARCH_NAMES + [
    "LLM + Agents (Light)",
    "LLM + RAG + Agents (In-Session)",
]

# ── QA agent prompts (mirrors bot/agents/qa_agent.py exactly) ─────────────────
# Copied here to avoid importing from bot.agents.llm which calls load_config().

_QA_SYSTEM_PROMPT = (
    "Ты персональный AI-тьютор, закреплённый за одним конкретным учебным блоком. "
    "Твоя цель — вести живой учебный диалог и помогать студенту глубоко разобраться именно в текущей теме. "
    "Ты не просто отвечаешь на вопросы, а ведёшь студента к пониманию как сильный репетитор один на один.\n\n"
    "Главные правила:\n"
    "1. Отвечай только в рамках текущей темы. Если вопрос не по теме, мягко верни студента.\n"
    "2. Если вопрос показывает пробел, сначала уточни — что студент уже понимает.\n"
    "3. Корректируй ошибки мягко: 'почти, но есть нюанс' вместо 'ты не прав'.\n"
    "4. Если точного ответа нет — честно обозначь границу и верни к главной теме.\n"
    "5. Говори как живой преподаватель: прямо, спокойно, уверенно, без шаблонных вступлений.\n"
    "6. Не употребляй слова 'учебник', 'материал', 'контекст', 'текст выше', 'источник'.\n"
    "7. Отвечай на русском языке.\n\n"
    "Формат: обычно 2–5 предложений; ключевые термины жирным; "
    "по возможности завершай вопросом или предложением следующего шага."
)

_LEVEL_GUIDANCE: dict[str, str] = {
    "beginner": (
        "Студенту тема даётся тяжело. Объясняй максимально просто, без перегруза терминами, "
        "через бытовые аналогии и очень короткие шаги."
    ),
    "intermediate": (
        "Студент понимает базу. Объясняй уверенно, связывай смысл с формулой и показывай, "
        "как идея работает на типичном примере."
    ),
    "advanced": (
        "Студент сильный. Отвечай более строго, подчёркивай нюансы, границы применимости "
        "и типичные ловушки."
    ),
}

_SIMPLE_SYSTEM = (
    "Ты опытный преподаватель. Отвечай на вопросы студентов точно и понятно на русском языке."
)

# ── Light agent system prompt ─────────────────────────────────────────────────
# Isolates pure pedagogical prompt engineering: level-awareness, structured
# explanation style, Socratic ending — WITHOUT in-session restrictions.
# No "assigned to specific block", no "admit limits and redirect", no "ask
# clarifying question instead of answering".  Proves that agent VALUE comes
# from prompting structure, not from restrictive guardrails.

_QA_SYSTEM_LIGHT = (
    "Ты опытный преподаватель-репетитор для студентов университета. "
    "Твоя задача — давать точные, понятные и педагогически выверенные ответы.\n\n"
    "Правила:\n"
    "1. Всегда давай содержательный ответ — не уклоняйся и не проси уточнений без крайней необходимости.\n"
    "2. Адаптируй глубину и язык под уровень студента (beginner / intermediate / advanced).\n"
    "3. Структурируй ответ: сначала суть, потом обоснование или пример.\n"
    "4. Выделяй **ключевые термины и понятия** жирным.\n"
    "5. Завершай ответ коротким вопросом или предложением следующего шага — это поддерживает диалог.\n"
    "6. Ответ 3–6 предложений. Без шаблонных вступлений вроде 'Конечно!' или 'Отличный вопрос!'.\n"
    "7. Отвечай на русском языке."
)

# ── LLM Backend ───────────────────────────────────────────────────────────────


class _Backend:
    """
    Minimal LLM client that reads env vars directly.
    No TELEGRAM_BOT_TOKEN required. Supports OpenRouter and Ollama.
    All responses cached to JSON so reruns cost nothing.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        ollama_url: str = "http://localhost:11434",
        timeout: float = 90.0,
        max_retries: int = 3,
        cache_path: str | Path = "system_bench_gen.json",
    ) -> None:
        self.provider   = provider
        self.model      = model
        self.api_key    = api_key
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout    = timeout
        self.max_retries = max_retries
        self._cache_path = Path(cache_path)
        self._cache: dict[str, str] = self._load_cache()

    @classmethod
    def from_env(cls, cache_path: str | Path = "system_bench_gen.json") -> "_Backend":
        provider   = os.getenv("LLM_PROVIDER", "openrouter")
        api_key    = os.getenv("OPENROUTER_API_KEY", "")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = (
            os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            if provider == "ollama"
            else os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
        )
        return cls(provider=provider, model=model, api_key=api_key,
                   ollama_url=ollama_url, cache_path=cache_path)

    def generate(
        self, prompt: str, system: str = "", max_tokens: int = 600, temperature: float = 0.18
    ) -> str:
        key = hashlib.sha256(
            f"{self.model}|{system[:200]}|{prompt}".encode("utf-8")
        ).hexdigest()[:24]
        if key in self._cache:
            return self._cache[key]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = (
                    self._call_ollama(prompt, system, max_tokens, temperature)
                    if self.provider == "ollama"
                    else self._call_openrouter(prompt, system, max_tokens, temperature)
                )
                self._cache[key] = result
                self._save_cache()
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning("Generation attempt %d/%d: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 16))
        logger.error("Generation failed after %d attempts: %s", self.max_retries, last_exc)
        return ""

    def _call_openrouter(self, prompt, system, max_tokens, temperature):
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
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, prompt, system, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = httpx.post(
            f"{self.ollama_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens}},
            timeout=httpx.Timeout(self.timeout, connect=5.0),
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _load_cache(self) -> dict[str, str]:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text("utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), "utf-8"
            )
        except Exception:
            pass

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"_Backend(provider={self.provider!r}, model={self.model!r}, cache={self.cache_size})"


# ── Standalone RAG Retriever ──────────────────────────────────────────────────


class _RAGRetriever:
    """
    PDF-based HybridRRF retriever — no Qdrant, no load_config() needed.
    Dense (sentence-transformers) + BM25, fused with Reciprocal Rank Fusion.
    Falls back gracefully if optional deps are missing.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, chunks: list[tuple[str, str]]) -> None:
        self._texts = [text for _, text in chunks]
        self._dense_vecs  = None   # np.ndarray, L2-normalised, shape (N, 384)
        self._bm25_index  = None
        self._model       = None
        self._init_dense()
        self._init_bm25()

    def _init_dense(self) -> None:
        if not HAS_ST or not self._texts:
            return
        try:
            self._model = SentenceTransformer(self._MODEL_NAME, local_files_only=True)
            vecs = self._model.encode(self._texts, show_progress_bar=False, batch_size=64)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._dense_vecs = (vecs / norms).astype("float32")
        except Exception as exc:
            logger.warning("Dense index failed: %s", exc)

    def _init_bm25(self) -> None:
        if not HAS_BM25 or not self._texts:
            return
        tokenized = [_tokenize_filtered(t) for t in self._texts]
        self._bm25_index = BM25Okapi(tokenized)

    def _dense_top_k(self, query: str, k: int) -> list[int]:
        if self._dense_vecs is None or self._model is None:
            return []
        qv = self._model.encode([query], show_progress_bar=False)[0].astype("float32")
        norm = float(np.linalg.norm(qv)) or 1.0
        qv = qv / norm
        scores = self._dense_vecs @ qv
        return list(np.argsort(scores)[::-1][:k])

    def _bm25_top_k(self, query: str, k: int) -> list[int]:
        if self._bm25_index is None:
            return []
        tokens = _tokenize_filtered(query)
        if not tokens:
            return []
        scores = self._bm25_index.get_scores(tokens)
        if HAS_NUMPY:
            return list(np.argsort(scores)[::-1][:k])
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    @staticmethod
    def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for pos, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
        return sorted(scores, key=lambda i: -scores[i])

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        pool = top_k * 4
        dense_ranks = self._dense_top_k(query, pool)
        bm25_ranks  = self._bm25_top_k(query, pool)

        rank_lists = [r for r in [dense_ranks, bm25_ranks] if r]
        if not rank_lists:
            return []

        merged = self._rrf(rank_lists)[:top_k]
        return [self._texts[i] for i in merged if i < len(self._texts)]

    @property
    def is_ready(self) -> bool:
        return self._dense_vecs is not None or self._bm25_index is not None


# ── Judge ─────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an educational QA evaluator for a university AI tutor system.\n"
    "Rate the quality of a system-generated answer to a student question.\n\n"
    "Dimensions:\n\n"
    "Correctness (0–3):\n"
    "  0 = Wrong or misleading — factual errors\n"
    "  1 = Partially correct — key facts right, but errors or crucial gaps present\n"
    "  2 = Mostly correct — minor inaccuracies or omissions only\n"
    "  3 = Fully correct — factually accurate and complete\n\n"
    "Completeness (0–2):\n"
    "  0 = Incomplete — major aspects of the question not addressed\n"
    "  1 = Partial — some aspects covered, key points missing\n"
    "  2 = Complete — all aspects of the question addressed\n\n"
    "Clarity (0–2):\n"
    "  0 = Unclear — confusing or hard to follow for a student\n"
    "  1 = Adequate — understandable but could be clearer\n"
    "  2 = Clear — well-structured, student-friendly\n\n"
    "Rules:\n"
    "- Evaluate even if the answer is in Russian or another language.\n"
    "- A concise but accurate answer can score 3 on correctness.\n"
    "- Use the reference answer to check correctness; allow equivalent phrasings.\n"
    "- Respond with EXACTLY 3 lines: correctness, completeness, clarity (one digit each).\n"
    "- No explanations, labels, or other text."
)

_JUDGE_TEMPLATE = (
    "STUDENT QUESTION: {question}\n\n"
    "TOPIC: {topic}\n\n"
    "REFERENCE ANSWER:\n{reference}\n\n"
    "SYSTEM ANSWER:\n{answer}\n\n"
    "Rate (3 lines only — correctness 0-3, completeness 0-2, clarity 0-2):"
)


def _composite(correctness: int, completeness: int, clarity: int) -> float:
    return correctness / 3 * 0.50 + completeness / 2 * 0.30 + clarity / 2 * 0.20


class AnswerJudge:
    """
    Rates system-generated answers on 3 dimensions using an LLM judge.
    Responses cached — no duplicate API calls on reruns.
    """

    def __init__(self, backend: _Backend) -> None:
        self._backend = backend

    def rate(
        self,
        question: str,
        topic: str,
        answer: str,
        reference: str = "",
    ) -> dict[str, float]:
        """
        Returns:
            correctness  (0-3)
            completeness (0-2)
            clarity      (0-2)
            composite    (0-1)
        """
        prompt = _JUDGE_TEMPLATE.format(
            question=question,
            topic=topic,
            reference=reference or "(no reference — use general knowledge)",
            answer=answer[:1200],
        )
        raw = self._backend.generate(prompt, _JUDGE_SYSTEM, max_tokens=20, temperature=0.0)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict[str, float]:
        digits = re.findall(r"[0-3]", raw)
        c3  = int(digits[0]) if len(digits) > 0 else 1
        c2a = int(digits[1]) if len(digits) > 1 else 1
        c2b = int(digits[2]) if len(digits) > 2 else 1
        # Clamp to valid ranges
        correctness  = min(c3,  3)
        completeness = min(c2a, 2)
        clarity      = min(c2b, 2)
        return {
            "correctness":  correctness,
            "completeness": completeness,
            "clarity":      clarity,
            "composite":    round(_composite(correctness, completeness, clarity), 4),
        }


# ── Architecture prompt builders ──────────────────────────────────────────────


def _build_simple_prompt(question: str, topic: str) -> str:
    return (
        f"Тема: {topic}\n\n"
        f"Вопрос студента: {question}\n\n"
        "Дай точный и понятный ответ."
    )


def _build_rag_prompt(question: str, topic: str, context: str) -> str:
    return (
        f"Тема: {topic}\n\n"
        f"Опорный материал:\n{context}\n\n"
        f"Вопрос студента: {question}\n\n"
        "Ответь точно, опираясь на предоставленный материал."
    )


def _build_agent_prompt(question: str, topic: str, level: str, context: str = "") -> str:
    guidance = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["beginner"])
    ctx_block = (
        f"Опора по теме (используй молча, без ссылок на источник):\n{context}\n\n"
        if context
        else "Опоры по теме нет. Ответь аккуратно только на основе своих знаний.\n\n"
    )
    return (
        f"Тема: {topic}\n"
        f"Уровень студента: {level}\n"
        f"Подсказка по уровню: {guidance}\n\n"
        f"{ctx_block}"
        f"Вопрос студента: {question}\n\n"
        "Как отвечать:\n"
        "- сначала определи, можно ли ответить сразу или нужно уточнить понимание;\n"
        "- если вопрос неясный, задай один уточняющий вопрос вместо длинного ответа;\n"
        "- если вопрос содержит ошибку, мягко поправь и объясни верный ход мысли;\n"
        "- если точного ответа нет, честно скажи и верни к главной теме.\n\n"
        "Требования:\n"
        "- выдели ключевые понятия **жирным**;\n"
        "- ответ 2–5 предложений, без лишних вступлений;\n"
        "- завершай вопросом или предложением следующего шага."
    )


# ── Architecture answer functions ─────────────────────────────────────────────


def _answer_llm_only(question: str, topic: str, level: str, backend: _Backend) -> str:
    """Arch 1: bare LLM, minimal prompt, no context."""
    prompt = _build_simple_prompt(question, topic)
    return backend.generate(prompt, _SIMPLE_SYSTEM, max_tokens=400, temperature=0.2)


def _answer_llm_rag(
    question: str, topic: str, level: str,
    retriever: _RAGRetriever, discipline: str,
    backend: _Backend, top_k: int = 5,
) -> str:
    """Arch 2: LLM + retrieved chunks, simple context injection (no agent prompts)."""
    chunks = retriever.retrieve(f"{question} {topic}", top_k=top_k)
    context = "\n\n".join(chunks[:top_k]) if chunks else ""
    if context:
        prompt = _build_rag_prompt(question, topic, context)
    else:
        prompt = _build_simple_prompt(question, topic)
    return backend.generate(prompt, _SIMPLE_SYSTEM, max_tokens=400, temperature=0.2)


def _answer_llm_agents(question: str, topic: str, level: str, backend: _Backend) -> str:
    """Arch 3: full qa_agent prompt engineering, NO RAG context."""
    prompt = _build_agent_prompt(question, topic, level, context="")
    return backend.generate(prompt, _QA_SYSTEM_PROMPT, max_tokens=500, temperature=0.2)


def _answer_llm_rag_agents(
    question: str, topic: str, level: str,
    retriever: _RAGRetriever, discipline: str,
    backend: _Backend, top_k: int = 5,
) -> str:
    """Arch 4: full production — agent prompts + RAG chunks."""
    chunks = retriever.retrieve(f"{question} {topic}", top_k=top_k)
    context = "\n\n".join(c[:600] for c in chunks[:top_k]) if chunks else ""
    prompt = _build_agent_prompt(question, topic, level, context=context)
    return backend.generate(prompt, _QA_SYSTEM_PROMPT, max_tokens=500, temperature=0.2)


# ── Fixed architecture helpers ────────────────────────────────────────────────


def _build_light_agent_prompt(question: str, topic: str, level: str, context: str = "") -> str:
    """
    Light prompt — same level guidance and output format as the full agent,
    but WITHOUT the in-session restrictions (no block-assignment, no refusal triggers).
    """
    guidance = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["beginner"])
    ctx_block = (
        f"Справочный материал по теме:\n{context}\n\n"
        if context
        else ""
    )
    return (
        f"Уровень студента: {level}. {guidance}\n\n"
        f"Тема вопроса: {topic}\n\n"
        f"{ctx_block}"
        f"Вопрос студента: {question}\n\n"
        "Объясни точно и структурированно. "
        "Начни с сути, затем дай обоснование или пример. "
        "Завершай коротким вопросом или предложением следующего шага."
    )


def _generate_lesson_block(
    topic: str,
    discipline: str,
    level: str,
    retriever: _RAGRetriever,
    backend: _Backend,
    top_k: int = 5,
) -> str:
    """
    Pre-generates a lesson block simulating what content_agent would have shown
    the student before a QA session begins.  Cached by (topic, discipline, level).

    In the real system the student first reads a lesson, then asks questions.
    qa_agent expects `last_block_content` to already be set.  This function
    replicates that precondition so the benchmark is fair.
    """
    chunks = retriever.retrieve(f"{topic} {discipline}", top_k=top_k)
    context = "\n\n".join(c[:500] for c in chunks) if chunks else ""
    ctx_section = f"Опорный материал:\n{context}\n\n" if context else ""

    guidance = {
        "beginner":     "Объясни максимально просто, через аналогии, шаг за шагом.",
        "intermediate": "Объясни чётко, свяжи смысл с формулой, дай типовой пример.",
        "advanced":     "Объясни строго, укажи нюансы и границы применимости.",
    }.get(level, "Объясни чётко и структурированно.")

    prompt = (
        f"Создай учебный блок по теме «{topic}» (дисциплина: {discipline}) "
        f"для студента уровня «{level}».\n\n"
        f"{ctx_section}"
        f"Указание по стилю: {guidance}\n\n"
        "Структура блока (обязательная):\n"
        "1. Краткое введение — суть темы и зачем она нужна (1–2 абзаца)\n"
        "2. Основные понятия — ключевые определения и их связи\n"
        "3. Пример — конкретный разбор применения темы\n"
        "4. Ключевые выводы — 2–3 главных тезиса\n\n"
        "Объём: 300–500 слов. Язык: русский. "
        "Не упоминай 'учебник', 'контекст', 'материал выше'. "
        "Выдели **ключевые термины** жирным."
    )
    return backend.generate(
        prompt,
        "Ты опытный преподаватель. Создай чёткий, структурированный учебный блок.",
        max_tokens=700,
        temperature=0.18,
    )


def _answer_llm_agents_light(
    question: str, topic: str, level: str, backend: _Backend
) -> str:
    """
    Arch 5 — LLM + Agents (Light).
    Lightweight pedagogical prompt: level adaptation + structured output style.
    No in-session restrictions, no block-assignment, no refusal triggers.
    Isolates the pure value of prompt engineering over plain LLM.
    """
    prompt = _build_light_agent_prompt(question, topic, level, context="")
    return backend.generate(prompt, _QA_SYSTEM_LIGHT, max_tokens=400, temperature=0.2)


def _answer_llm_rag_agents_insession(
    question: str,
    topic: str,
    level: str,
    retriever: _RAGRetriever,
    discipline: str,
    backend: _Backend,
    top_k: int = 5,
    lesson_cache: dict | None = None,
) -> str:
    """
    Arch 6 — LLM + RAG + Agents (In-Session).
    Proper simulation of the real multi-agent tutor flow:

      1. Generate (or cache) a lesson block for the topic
         → simulates content_agent having already taught this topic
      2. Retrieve additional RAG chunks for the specific question
         → simulates qa_agent's runtime search
      3. Inject lesson + chunks as context into the full qa_agent prompt
         → the agent now behaves as it does in production

    This is the architecturally correct comparison: plain LLM vs. the full
    pipeline INCLUDING the in-session context that agents are designed to use.
    """
    if lesson_cache is None:
        lesson_cache = {}

    cache_key = f"{topic}|{discipline}|{level}"
    if cache_key not in lesson_cache:
        print(f"    [pre-gen lesson for '{topic[:40]}']", end=" ", flush=True)
        lesson_cache[cache_key] = _generate_lesson_block(
            topic, discipline, level, retriever, backend, top_k
        )
        print("ok")

    lesson_block = lesson_cache[cache_key]

    # Additional retrieval for this specific question (mirrors qa_agent search)
    chunks = retriever.retrieve(f"{question} {topic}", top_k=top_k)
    rag_ctx = "\n\n".join(c[:500] for c in chunks) if chunks else ""

    full_context = lesson_block
    if rag_ctx:
        full_context += f"\n\nДополнительный материал по вопросу:\n{rag_ctx}"

    prompt = _build_agent_prompt(question, topic, level, context=full_context)
    return backend.generate(prompt, _QA_SYSTEM_PROMPT, max_tokens=500, temperature=0.2)


# ── Test question generation ──────────────────────────────────────────────────

_QG_SYSTEM = (
    "You generate educational QA pairs for benchmarking. "
    "Output only valid JSON, nothing else."
)

_QG_TEMPLATE = (
    "Read the educational text below and output a JSON object with two fields:\n"
    "  \"question\": a natural student question that this text answers (in Russian)\n"
    "  \"reference\": a concise reference answer based strictly on this text (in Russian)\n\n"
    "Text:\n{chunk}\n\n"
    "Output only the JSON object, no other text."
)


def generate_test_questions(
    backend: _Backend,
    chunks: list[tuple[str, str]],
    n: int = 20,
    seed: int = 42,
    discipline_label: str = "discipline",
) -> list[dict]:
    """
    Generate n QA pairs from corpus chunks using the LLM.
    Returns list of dicts: {question, topic, reference, source_chunk}.
    Cached — regeneration on reruns costs nothing.
    """
    random.seed(seed)
    sample = random.sample(chunks, min(n, len(chunks)))
    pairs: list[dict] = []

    for i, (source, chunk_text) in enumerate(sample, 1):
        print(f"  Generating question {i}/{len(sample)} from {source[:30]}...", end=" ", flush=True)
        prompt = _QG_TEMPLATE.format(chunk=chunk_text[:700])
        raw = backend.generate(prompt, _QG_SYSTEM, max_tokens=250, temperature=0.25)

        question = reference = ""
        try:
            # Try to extract JSON even if model adds surrounding text
            match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                question  = obj.get("question", "").strip()
                reference = obj.get("reference", "").strip()
        except (json.JSONDecodeError, AttributeError):
            pass

        if not question:
            # Fallback: extract lines starting with q:/r:
            for line in raw.splitlines():
                if line.lower().startswith("вопрос") or line.lower().startswith("q:"):
                    question = re.sub(r"^[^:]+:\s*", "", line).strip()
                elif line.lower().startswith("ответ") or line.lower().startswith("r:"):
                    reference = re.sub(r"^[^:]+:\s*", "", line).strip()

        if not question:
            print("skip (no question extracted)")
            continue

        # Infer topic from source filename (strip extension, prettify)
        topic = re.sub(r"[-_]", " ", Path(source).stem).strip() or discipline_label

        pairs.append({
            "question":     question,
            "topic":        topic,
            "reference":    reference,
            "source_chunk": chunk_text[:150],
        })
        print(f"ok  ({len(question)} chars)")

    return pairs


def load_questions_from_file(path: str) -> list[dict]:
    """Load pre-written questions from JSON file."""
    data = json.loads(Path(path).read_text("utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("questions", [])
    raise ValueError("questions file must be a JSON list or {\"questions\": [...]}")


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_all(
    questions: list[dict],
    discipline: str,
    level: str,
    retriever: _RAGRetriever,
    backend: _Backend,
    judge: AnswerJudge,
    top_k: int = 5,
    verbose: bool = False,
    arch_names: list[str] | None = None,
    lesson_cache: dict | None = None,
) -> dict[str, list[dict]]:
    """
    Runs architectures on every question and rates answers with the judge.

    Args:
        arch_names:    subset of ARCH_NAMES_EXTENDED to evaluate.
                       Defaults to ARCH_NAMES (original 4).
        lesson_cache:  shared dict for pre-generated lesson blocks.
                       Pass the same dict across calls so blocks are not
                       regenerated when evaluating multiple question sets.

    Returns:
        {arch_name: [{"question", "answer", "correctness", "completeness",
                      "clarity", "composite"}, ...]}
    """
    if arch_names is None:
        arch_names = ARCH_NAMES
    if lesson_cache is None:
        lesson_cache = {}

    results: dict[str, list[dict]] = {name: [] for name in arch_names}

    # Build the full dispatcher once (avoids per-iteration lambda rebinding)
    _dispatch: dict[str, object] = {
        "LLM Only":
            lambda qu, to, le: _answer_llm_only(qu, to, le, backend),
        "LLM + RAG":
            lambda qu, to, le: _answer_llm_rag(qu, to, le, retriever, discipline, backend, top_k),
        "LLM + Agents":
            lambda qu, to, le: _answer_llm_agents(qu, to, le, backend),
        "LLM + RAG + Agents":
            lambda qu, to, le: _answer_llm_rag_agents(qu, to, le, retriever, discipline, backend, top_k),
        "LLM + Agents (Light)":
            lambda qu, to, le: _answer_llm_agents_light(qu, to, le, backend),
        "LLM + RAG + Agents (In-Session)":
            lambda qu, to, le: _answer_llm_rag_agents_insession(
                qu, to, le, retriever, discipline, backend, top_k, lesson_cache
            ),
    }

    for qi, q in enumerate(questions, 1):
        question  = q["question"]
        topic     = q.get("topic", discipline)
        reference = q.get("reference", "")

        print(f"\n[Q{qi:02d}/{len(questions)}] {question[:70]}...")

        for arch_name in arch_names:
            fn = _dispatch[arch_name]
            answer = fn(question, topic, level)
            scores = judge.rate(question, topic, answer, reference)

            results[arch_name].append({
                "question":     question,
                "topic":        topic,
                "answer":       answer,
                **scores,
            })
            if verbose:
                print(
                    f"  [{arch_name:35s}]  "
                    f"corr={scores['correctness']}  "
                    f"comp={scores['completeness']}  "
                    f"clar={scores['clarity']}  "
                    f"composite={scores['composite']:.2f}"
                )
            else:
                print(f"  [{arch_name:35s}] composite={scores['composite']:.2f}", end="  ")

        print()

    return results


def compute_metrics(results: dict[str, list[dict]]) -> dict[str, dict]:
    """
    Aggregate per-question results into mean metrics per architecture.

    Returns:
        {arch_name: {mean_correctness, mean_completeness, mean_clarity,
                     mean_composite, pct_good (≥0.6), pct_excellent (≥0.8), n}}
    """
    agg: dict[str, dict] = {}
    for arch, rows in results.items():
        if not rows:
            agg[arch] = {}
            continue
        n = len(rows)
        agg[arch] = {
            "mean_correctness":  round(_mean(r["correctness"]  for r in rows), 4),
            "mean_completeness": round(_mean(r["completeness"] for r in rows), 4),
            "mean_clarity":      round(_mean(r["clarity"]      for r in rows), 4),
            "mean_composite":    round(_mean(r["composite"]     for r in rows), 4),
            "pct_good":          round(sum(r["composite"] >= 0.60 for r in rows) / n, 4),
            "pct_excellent":     round(sum(r["composite"] >= 0.80 for r in rows) / n, 4),
            "n": n,
        }
    return agg


def compute_gains(metrics: dict[str, dict]) -> dict[str, float]:
    """
    Decompose improvement into:
      rag_gain    = (LLM+RAG)         - (LLM Only)
      agent_gain  = (LLM+Agents)      - (LLM Only)
      full_gain   = (LLM+RAG+Agents)  - (LLM Only)
      synergy     = full_gain - rag_gain - agent_gain
    Uses mean_composite as the primary metric.
    """
    def _c(name): return metrics.get(name, {}).get("mean_composite", 0.0)
    base        = _c("LLM Only")
    rag_gain    = round(_c("LLM + RAG")          - base, 4)
    agent_gain  = round(_c("LLM + Agents")       - base, 4)
    full_gain   = round(_c("LLM + RAG + Agents") - base, 4)
    synergy     = round(full_gain - rag_gain - agent_gain, 4)
    return {
        "base":       round(base, 4),
        "rag_gain":   rag_gain,
        "agent_gain": agent_gain,
        "full_gain":  full_gain,
        "synergy":    synergy,   # positive = superadditive effect
    }


def compute_gains_extended(metrics: dict[str, dict]) -> dict[str, float]:
    """
    Extended gain table including the two fixed architectures.

    Additional keys:
      light_gain       = (LLM+Agents Light)      - (LLM Only)
      insession_gain   = (LLM+RAG+Agents InSess) - (LLM Only)
      insession_vs_rag = (LLM+RAG+Agents InSess) - (LLM+RAG)
                         positive → agent adds value ON TOP of pure RAG
      light_vs_agents  = (LLM+Agents Light)      - (LLM+Agents)
                         quantifies how much the restrictive prompt hurt
    """
    base = compute_gains(metrics)   # reuse original 4-arch gains

    def _c(name): return metrics.get(name, {}).get("mean_composite", 0.0)
    b               = _c("LLM Only")
    light_gain      = round(_c("LLM + Agents (Light)")              - b,                      4)
    insession_gain  = round(_c("LLM + RAG + Agents (In-Session)")   - b,                      4)
    insession_vs_rag= round(_c("LLM + RAG + Agents (In-Session)")   - _c("LLM + RAG"),        4)
    light_vs_agents = round(_c("LLM + Agents (Light)")              - _c("LLM + Agents"),     4)

    return {
        **base,
        "light_gain":       light_gain,
        "insession_gain":   insession_gain,
        "insession_vs_rag": insession_vs_rag,   # agent value on top of RAG
        "light_vs_agents":  light_vs_agents,    # how much restrictive prompt hurt
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_table(metrics: dict[str, dict], gains: dict[str, float]) -> None:
    cols = ["mean_correctness", "mean_completeness", "mean_clarity",
            "mean_composite", "pct_good", "pct_excellent"]
    col_w = 15
    arch_w = 38
    sep = "─" * (arch_w + col_w * len(cols))

    print()
    print(sep)
    header = f"{'Architecture':<{arch_w}}" + "".join(f"{c[:col_w]:>{col_w}}" for c in cols)
    print(header)
    print(sep)
    for arch in metrics:
        m = metrics.get(arch, {})
        if not m:
            continue
        row = f"{arch:<{arch_w}}"
        for col in cols:
            row += f"{m.get(col, 0.0):>{col_w}.3f}"
        print(row)
    print(sep)

    print(f"\n  Composite gain over 'LLM Only':")
    print(f"    RAG alone        : {gains['rag_gain']:+.3f}")
    print(f"    Agents (orig)    : {gains['agent_gain']:+.3f}")
    print(f"    RAG + Agents     : {gains['full_gain']:+.3f}")
    print(f"    Synergy          : {gains['synergy']:+.3f}  "
          f"({'superadditive ✓' if gains['synergy'] > 0 else 'subadditive'})")
    if "light_gain" in gains:
        print(f"    Agents (light)   : {gains['light_gain']:+.3f}")
        print(f"    RAG+Agents InSess: {gains['insession_gain']:+.3f}")
        print(f"    InSession vs RAG : {gains['insession_vs_rag']:+.3f}  "
              f"({'agents add value on top of RAG ✓' if gains['insession_vs_rag'] > 0 else 'RAG already sufficient'})")
        print(f"    Light vs Orig    : {gains['light_vs_agents']:+.3f}  "
              f"(how much restrictive prompt cost)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="6-architecture QA benchmark with LLM-as-Judge (incl. fixed architectures)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("discipline", help="Discipline slug (e.g. math)")
    parser.add_argument("--samples",        type=int, default=20,
                        help="Number of questions to generate (ignored if --questions-file)")
    parser.add_argument("--level",          default="beginner",
                        choices=["beginner", "intermediate", "advanced", "all"],
                        help="Student level ('all' runs all 3 levels)")
    parser.add_argument("--top-k",          type=int, default=5,  help="RAG top-k chunks")
    parser.add_argument("--seed",           type=int, default=42, help="Random seed")
    parser.add_argument("--questions-file", default="",
                        help="Path to pre-written questions JSON (skips auto-generation)")
    parser.add_argument("--output",         default="system_bench_results.json",
                        help="Output JSON path")
    parser.add_argument("--verbose",        action="store_true", help="Print per-answer scores")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # ── Load corpus ────────────────────────────────────────────────────────────
    if not HAS_CORPUS:
        print("ERROR: benchmark_fallback.py not found."); sys.exit(1)

    disciplines_dir = os.getenv("DISCIPLINES_DIR", "data/disciplines")
    print(f"\nLoading corpus: discipline={args.discipline!r}, dir={disciplines_dir!r}")
    chunks = load_chunks(disciplines_dir, args.discipline)
    if not chunks:
        print(f"ERROR: no chunks found for '{args.discipline}'."); sys.exit(1)
    print(f"Chunks: {len(chunks)}")

    # ── Build retriever ────────────────────────────────────────────────────────
    print("Building RAG retriever (dense + BM25)...")
    retriever = _RAGRetriever(chunks)
    if not retriever.is_ready:
        print("WARNING: Neither sentence-transformers nor rank-bm25 available.")
        print("         Install: uv add sentence-transformers rank-bm25")
        print("         LLM+RAG architectures will return empty context.\n")

    # ── LLM backend ───────────────────────────────────────────────────────────
    backend = _Backend.from_env(cache_path="system_bench_gen.json")
    judge   = AnswerJudge(_Backend.from_env(cache_path="system_bench_judge.json"))
    print(f"Backend: {backend}")

    # ── Questions ──────────────────────────────────────────────────────────────
    if args.questions_file:
        print(f"Loading questions from {args.questions_file}")
        questions = load_questions_from_file(args.questions_file)
    else:
        print(f"Generating {args.samples} questions from corpus...")
        questions = generate_test_questions(backend, chunks, n=args.samples, seed=args.seed)

    if not questions:
        print("ERROR: no questions available."); sys.exit(1)
    print(f"Questions ready: {len(questions)}\n")

    # ── Run evaluation (all 6 architectures) ──────────────────────────────────
    levels = ["beginner", "intermediate", "advanced"] if args.level == "all" else [args.level]
    all_output: dict = {"discipline": args.discipline, "top_k": args.top_k,
                        "questions": questions, "levels": {}}
    lesson_cache: dict = {}   # shared across levels to avoid repeated generation

    for level in levels:
        print(f"\n{'='*68}")
        print(f"Level: {level.upper()}")
        print(f"{'='*68}")

        results = evaluate_all(
            questions, args.discipline, level, retriever,
            backend, judge, top_k=args.top_k, verbose=args.verbose,
            arch_names=ARCH_NAMES_EXTENDED, lesson_cache=lesson_cache,
        )
        metrics = compute_metrics(results)
        gains   = compute_gains_extended(metrics)

        _print_table(metrics, gains)

        all_output["levels"][level] = {
            "metrics": metrics,
            "gains":   gains,
            "results": results,
        }

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.write_text(json.dumps(all_output, ensure_ascii=False, indent=2), "utf-8")
    print(f"Results saved → {out_path}")
    print(f"Gen cache  → system_bench_gen.json   ({backend.cache_size} entries)")
    print(f"Judge cache → system_bench_judge.json")


if __name__ == "__main__":
    main()
