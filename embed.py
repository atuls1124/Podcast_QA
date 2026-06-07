"""
Step 3, 4 & 5 of the pipeline.

Loads the JSON transcript produced by transcribe.py, merges short segments
into reasonably-sized chunks (without losing timestamps), embeds them with
Sentence-Transformers, and persists a FAISS index together with the chunk
metadata.

Output:
    embeddings/faiss.index   - the vector index
    embeddings/metadata.pkl  - list[dict] parallel to the index, each dict
                               is {start, end, text}
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_MAX_CHARS,
    EMBEDDING_MODEL,
    EMBEDDINGS_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TRANSCRIPT_DIR,
)


def load_transcript(path: Path) -> list[dict]:
    """Load the JSON transcript produced by transcribe.py."""
    with open(path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            "transcript.json is empty or malformed. "
            "Run `python transcribe.py` first."
        )
    return segments


def build_chunks(segments: list[dict], max_chars: int = CHUNK_MAX_CHARS) -> list[dict]:
    """
    Concatenate adjacent Whisper segments until the running text length
    exceeds `max_chars`. The chunk's `start` is the earliest segment start
    and its `end` is the latest segment end, so timestamps stay accurate.
    """
    chunks: list[dict] = []
    buffer_text = ""
    buffer_start: float | None = None
    buffer_end: float | None = None

    def flush() -> None:
        nonlocal buffer_text, buffer_start, buffer_end
        if buffer_text and buffer_start is not None and buffer_end is not None:
            chunks.append(
                {
                    "start": float(buffer_start),
                    "end": float(buffer_end),
                    "text": buffer_text,
                }
            )
        buffer_text = ""
        buffer_start = None
        buffer_end = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        if buffer_start is None:
            buffer_start = float(seg["start"])

        # If the new text would overflow the chunk, flush the buffer first.
        if buffer_text and len(buffer_text) + len(text) + 1 > max_chars:
            flush()
            buffer_start = float(seg["start"])

        buffer_text = (buffer_text + " " + text).strip() if buffer_text else text
        buffer_end = float(seg["end"])

    flush()
    print(
        f"[INFO] Merged {len(segments)} segments into {len(chunks)} chunks "
        f"(max {max_chars} chars each)."
    )
    return chunks


def build_faiss_index(chunks: list[dict], model_name: str) -> faiss.Index:
    """
    Embed every chunk with Sentence-Transformers and return a FAISS index
    that uses inner-product similarity on L2-normalized vectors (= cosine).
    """
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"[INFO] Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # we still call faiss.normalize_L2 as a guard
    ).astype("float32")

    # Defensive: make sure vectors are unit-normalized before inner-product.
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index contains {index.ntotal} vectors of dim {dim}.")
    return index


def main() -> None:
    transcript_path = TRANSCRIPT_DIR / "transcript.json"
    if not transcript_path.exists():
        print(
            f"[ERROR] {transcript_path} not found. "
            f"Run `python transcribe.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        segments = load_transcript(transcript_path)
        chunks = build_chunks(segments)

        index = build_faiss_index(chunks, EMBEDDING_MODEL)

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(chunks, f)

        print(f"[INFO] Saved FAISS index to: {FAISS_INDEX_PATH}")
        print(f"[INFO] Saved metadata to:    {METADATA_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] embed.py failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
