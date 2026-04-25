from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .answering import build_grounded_answer
from .chunking import chunk_document
from .documents import load_documents
from .evals import ensure_sample_docs
from .index import SparseIndex
from .memory import SessionMemoryStore
from .rerank import rerank_hits
from .retrieval import serialize_hits, hybrid_retrieve
from .vision import analyze_meal_image, analysis_to_text


@dataclass(slots=True)
class ServiceConfig:
    project_root: Path
    docs_root: Path
    memory_root: Path
    chunk_strategy: str = "hybrid"
    top_k: int = 5
    rerank_top_k: int = 3


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _line_value(text: str, prefix: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(prefix)}\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _hit_candidates(index: SparseIndex, hit_dicts: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for item in hit_dicts:
        chunk_id = str(item.get("chunk_id", ""))
        if not chunk_id or chunk_id not in index.chunks:
            continue
        chunk = index.chunks[chunk_id]
        text = chunk.text
        candidates.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "source_path": chunk.source_path,
                "meal_type": _line_value(text, "Meal type:") or "meal",
                "prep_time": _line_value(text, "Prep time:") or "",
                "calories": _line_value(text, "Calories:") or "",
                "protein": _line_value(text, "Protein:") or "",
                "ingredients": _line_value(text, "Ingredients:") or "",
                "score": item.get("score", 0.0),
                "components": {
                    "sparse_score": item.get("sparse_score", 0.0),
                    "ingredient_overlap": item.get("ingredient_overlap", 0.0),
                    "meal_type_match": item.get("meal_type_match", 0.0),
                    "habit_alignment": item.get("habit_alignment", 0.0),
                    "log_reuse_bonus": item.get("log_reuse_bonus", 0.0),
                    "vision_alignment": item.get("vision_alignment", 0.0),
                    "visual_memory_alignment": item.get("visual_memory_alignment", 0.0),
                    "rerank_bonus": item.get("rerank_bonus", 0.0),
                },
            }
        )
    return candidates


class RecipeQueryService:
    def __init__(
        self,
        *,
        project_root_path: str | Path | None = None,
        docs_root: str | Path | None = None,
        memory_root: str | Path | None = None,
        chunk_strategy: str = "hybrid",
        top_k: int = 5,
        rerank_top_k: int = 3,
        ensure_default_docs: bool = True,
    ):
        root = Path(project_root_path) if project_root_path is not None else project_root()
        resolved_docs = Path(docs_root) if docs_root is not None else (ensure_sample_docs(root) if ensure_default_docs else root / "examples")
        resolved_memory = Path(memory_root) if memory_root is not None else root / "data" / "memories"
        self.config = ServiceConfig(
            project_root=root,
            docs_root=resolved_docs,
            memory_root=resolved_memory,
            chunk_strategy=chunk_strategy,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )
        self.memory_store = SessionMemoryStore(self.config.memory_root)

    def build_index(self) -> tuple[SparseIndex, dict]:
        documents = load_documents(self.config.docs_root)
        chunks = []
        for document in documents:
            chunks.extend(chunk_document(document, strategy=self.config.chunk_strategy))
        index = SparseIndex(chunks)
        meta = {
            "docs_root": str(self.config.docs_root),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_strategy": self.config.chunk_strategy,
        }
        return index, meta

    def query(
        self,
        *,
        query: str,
        session_id: str = "demo",
        image_path: str | Path | None = None,
        image_analysis: dict | None = None,
        vision_provider: str | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> dict:
        resolved_top_k = int(top_k if top_k is not None else self.config.top_k)
        resolved_rerank_k = int(rerank_top_k if rerank_top_k is not None else self.config.rerank_top_k)
        effective_query = query
        used_image_analysis = image_analysis
        image_path_value = str(image_path) if image_path else ""
        if used_image_analysis is None and image_path:
            used_image_analysis = analyze_meal_image(image_path, provider=vision_provider)
        if used_image_analysis is not None:
            effective_query = f"{query}\nVisual context: {analysis_to_text(used_image_analysis)}"

        index, index_meta = self.build_index()
        hits, trace = hybrid_retrieve(
            effective_query,
            index,
            top_k=resolved_top_k,
            memory_store=self.memory_store,
            session_id=session_id,
            image_analysis=used_image_analysis,
        )
        reranked_hits = rerank_hits(effective_query, hits, index, top_k=resolved_rerank_k)
        reranked_hit_dicts = serialize_hits(reranked_hits)
        trace["reranked_hits"] = reranked_hit_dicts
        answer = build_grounded_answer(effective_query, reranked_hits, index, trace=trace)
        return {
            "query": query,
            "effective_query": effective_query,
            "session_id": session_id,
            "image_path": image_path_value,
            "vision_provider": vision_provider or "",
            "image_analysis": used_image_analysis,
            "hits": reranked_hit_dicts,
            "trace": trace,
            "answer": answer["answer"],
            "answer_trace": answer["trace"],
            "citations": answer["citations"],
            "candidates": _hit_candidates(index, reranked_hit_dicts),
            "profile_summary": self.memory_store.build_profile_summary(session_id),
            "meta": {
                **index_meta,
                "top_k": resolved_top_k,
                "rerank_top_k": resolved_rerank_k,
            },
        }


def run_recipe_query(
    *,
    query: str,
    session_id: str = "demo",
    image_path: str | Path | None = None,
    image_analysis: dict | None = None,
    vision_provider: str | None = None,
    project_root_path: str | Path | None = None,
    docs_root: str | Path | None = None,
    memory_root: str | Path | None = None,
    chunk_strategy: str = "hybrid",
    top_k: int = 5,
    rerank_top_k: int = 3,
    ensure_default_docs: bool = True,
) -> dict:
    service = RecipeQueryService(
        project_root_path=project_root_path,
        docs_root=docs_root,
        memory_root=memory_root,
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        ensure_default_docs=ensure_default_docs,
    )
    return service.query(
        query=query,
        session_id=session_id,
        image_path=image_path,
        image_analysis=image_analysis,
        vision_provider=vision_provider,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )
