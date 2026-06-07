"""
Step 6 & 7 of the pipeline - the Retrieval-Augmented Generation (RAG) layer.

Given a natural-language question:
  1. Encode the question with the same Sentence-Transformers model.
  2. Retrieve the top-k most similar transcript chunks from FAISS.
  3. Send the chunks + the question to an LLM served by OpenRouter
     (OpenAI-compatible API) with a strict "answer from context only"
     system prompt.
  4. Return the answer, the timestamp of the best chunk, the transcript
     segment, and a clickable YouTube URL that deep-links to that moment.

Usage from CLI:
    python chatbot.py "What is first-principles thinking?"
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    APP_NAME,
    APP_URL,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LLM_MODEL,
    LLM_TEMPERATURE,
    METADATA_PATH,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TOP_K,
    YOUTUBE_URL,
)
from utils import format_timestamp, youtube_url_with_timestamp


# Make sure stdout can print any Unicode the LLM might return (em-dashes,
# smart quotes, non-Latin characters, etc.) even when running on Windows
# with the default cp1252 console encoding.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass


SYSTEM_PROMPT = """You are a precise Q&A assistant for a single podcast transcript.

Rules you must follow:
1. Use ONLY the transcript context provided. Do not use any outside knowledge.
2. If the answer is not contained in the context, reply EXACTLY with:
   "I could not find the answer in the podcast."
3. Keep answers short (2-4 sentences). Quote the speaker when it helps.
4. Never invent names, numbers, dates, or timestamps.
5. Attribute statements to the speaker when obvious (e.g. "Elon Musk says...").
"""


@dataclass
class QAResult:
    """Structured response from PodcastQA.ask()."""

    answer: str
    timestamp: str
    start_seconds: float
    end_seconds: float
    transcript: str
    youtube_url: str
    chunks: list[dict]


class PodcastQA:
    """Encapsulates the FAISS index, the embedding model, and the LLM client."""

    def __init__(self) -> None:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env "
                "and fill in your OpenRouter API key from "
                "https://openrouter.ai/keys"
            )
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "FAISS index or metadata is missing. "
                "Run `python transcribe.py` and `python embed.py` first."
            )

        # OpenRouter exposes an OpenAI-compatible chat-completions endpoint,
        # so we can reuse the official `openai` Python SDK by just pointing
        # the base_url at OpenRouter. The optional HTTP-Referer / X-Title
        # headers are recommended by OpenRouter for app identification.
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": APP_URL,
                "X-Title": APP_NAME,
            },
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index: faiss.Index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)
        self.youtube_url = YOUTUBE_URL

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        """Embed `question` and return the top-k most relevant chunks."""
        q_emb = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _build_context(self, chunks: list[dict]) -> str:
        """Format the retrieved chunks into a context block for the LLM."""
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(
                f"[Segment {i} | {format_timestamp(c['start'])} - "
                f"{format_timestamp(c['end'])}]\n{c['text']}"
            )
        return "\n\n".join(blocks)

    def generate_answer(self, question: str, context_chunks: list[dict]) -> str:
        """Call the LLM with the retrieved context to produce a grounded answer."""
        context = self._build_context(context_chunks)
        user_prompt = (
            f"Transcript context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer (use only the context above):"
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # End-to-end
    # ------------------------------------------------------------------ #
    def ask(self, question: str, k: int = TOP_K) -> QAResult:
        """Run the full RAG pipeline for a single question."""
        chunks = self.retrieve(question, k=k)

        if not chunks:
            return QAResult(
                answer="I could not find the answer in the podcast.",
                timestamp="N/A",
                start_seconds=0.0,
                end_seconds=0.0,
                transcript="",
                youtube_url=self.youtube_url,
                chunks=[],
            )

        answer = self.generate_answer(question, chunks)
        best = chunks[0]

        return QAResult(
            answer=answer,
            timestamp=format_timestamp(best["start"]),
            start_seconds=float(best["start"]),
            end_seconds=float(best["end"]),
            transcript=best["text"],
            youtube_url=youtube_url_with_timestamp(self.youtube_url, best["start"]),
            chunks=chunks,
        )


def _print_result(result: QAResult) -> None:
    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== TIMESTAMP ===")
    print(result.timestamp)
    print("\n=== TRANSCRIPT SEGMENT ===")
    print(result.transcript)
    print("\n=== WATCH ===")
    print(result.youtube_url)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "What is first-principles thinking?"
        print(f"[INFO] No question provided, using default: {question!r}")

    try:
        qa = PodcastQA()
        result = qa.ask(question)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] chatbot.py failed: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_result(result)


if __name__ == "__main__":
    main()
