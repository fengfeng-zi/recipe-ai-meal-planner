from __future__ import annotations

from dataclasses import dataclass
import re

from .documents import Document


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    strategy: str
    position: int
    source_path: str


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_document(document: Document, strategy: str = "hybrid", window_size: int = 3, overlap: int = 1) -> list[Chunk]:
    if strategy == "paragraph":
        units = _paragraphs(document.text)
    elif strategy == "sliding":
        units = _sentences(document.text)
    else:
        units = _paragraphs(document.text) if len(_paragraphs(document.text)) >= 2 else _sentences(document.text)

    chunks: list[Chunk] = []
    if strategy == "sliding":
        step = max(1, window_size - overlap)
        for idx in range(0, len(units), step):
            text = " ".join(units[idx: idx + window_size]).strip()
            if not text:
                continue
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}::chunk::{len(chunks)}",
                doc_id=document.doc_id,
                title=document.title,
                text=text,
                strategy=strategy,
                position=len(chunks),
                source_path=document.source_path,
            ))
        return chunks

    for idx, text in enumerate(units):
        chunks.append(Chunk(
            chunk_id=f"{document.doc_id}::chunk::{idx}",
            doc_id=document.doc_id,
            title=document.title,
            text=text,
            strategy=strategy,
            position=idx,
            source_path=document.source_path,
        ))
    return chunks
