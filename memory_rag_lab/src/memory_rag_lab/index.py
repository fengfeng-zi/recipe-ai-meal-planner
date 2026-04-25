from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import math
import re

from .chunking import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class SparseIndex:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: dict[str, Chunk] = {}
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freq: dict[str, Counter] = {}
        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            counts = Counter(tokenize(chunk.text))
            self.term_freq[chunk.chunk_id] = counts
            for token in counts.keys():
                self.doc_freq[token] += 1

    def idf(self, token: str) -> float:
        total_docs = max(1, len(self.chunks))
        return math.log((1 + total_docs) / (1 + self.doc_freq.get(token, 0))) + 1.0

    def sparse_score(self, query: str, chunk_id: str) -> float:
        counts = self.term_freq.get(chunk_id, Counter())
        score = 0.0
        for token in tokenize(query):
            score += counts.get(token, 0) * self.idf(token)
        return score

    def to_dict(self) -> dict:
        return {
            "chunks": {chunk_id: asdict(chunk) for chunk_id, chunk in self.chunks.items()},
        }
