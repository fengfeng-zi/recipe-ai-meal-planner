from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analytics import build_report
from .answering import build_grounded_answer
from .chunking import chunk_document
from .documents import load_documents
from .evals import ensure_sample_docs, run_eval_suite
from .index import SparseIndex
from .memory import SessionMemoryStore
from .rerank import rerank_hits
from .retrieval import hybrid_retrieve, serialize_hits
from .vision import analysis_to_text, analyze_meal_image

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
MEMORY_TYPES = ("preference", "goal", "allergy", "dislike", "note")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_index(root: Path) -> SparseIndex:
    docs_root = ensure_sample_docs(root)
    documents = load_documents(docs_root)
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy="hybrid"))
    return SparseIndex(chunks)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recipe query loop with grounded recipe retrieval and food-habit memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask")
    ask.add_argument("--query", required=True)
    ask.add_argument("--session-id", default="demo")
    ask.add_argument("--image-path", default="")
    ask.add_argument("--vision-provider", default="")

    remember = subparsers.add_parser("remember")
    remember.add_argument("--session-id", default="demo")
    remember.add_argument("--text", required=True)
    remember.add_argument("--memory-type", choices=MEMORY_TYPES, default="preference")
    remember.add_argument("--tags", default="")

    log_meal = subparsers.add_parser("log-meal")
    log_meal.add_argument("--session-id", default="demo")
    log_meal.add_argument("--meal-name", required=True)
    log_meal.add_argument("--meal-type", choices=MEAL_TYPES, required=True)
    log_meal.add_argument("--calories", type=int)
    log_meal.add_argument("--ingredients", default="")
    log_meal.add_argument("--notes", default="")

    show_profile = subparsers.add_parser("show-profile", aliases=["profile"])
    show_profile.add_argument("--session-id", default="demo")

    analyze_image = subparsers.add_parser("analyze-image")
    analyze_image.add_argument("--image-path", required=True)
    analyze_image.add_argument("--session-id", default="demo")
    analyze_image.add_argument("--vision-provider", default="")
    analyze_image.add_argument("--remember", action="store_true")

    subparsers.add_parser("eval")
    subparsers.add_parser("report")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = project_root()
    memory_store = SessionMemoryStore(root / "data" / "memories")

    if args.command == "remember":
        saved = memory_store.save_preference(
            args.session_id,
            memory_text=args.text,
            memory_type=args.memory_type,
            tags=_split_csv(args.tags),
        )
        print(json.dumps(saved, ensure_ascii=False, indent=2))
        return

    if args.command == "log-meal":
        logged = memory_store.log_meal(
            args.session_id,
            meal_name=args.meal_name,
            meal_type=args.meal_type,
            calories=args.calories,
            ingredients=_split_csv(args.ingredients),
            notes=args.notes,
        )
        print(json.dumps(logged, ensure_ascii=False, indent=2))
        return

    if args.command in {"show-profile", "profile"}:
        payload = {
            "profile_summary": memory_store.build_profile_summary(args.session_id),
            "memories": memory_store.list_memories(args.session_id),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "analyze-image":
        ensure_sample_docs(root)
        analysis = analyze_meal_image(args.image_path, provider=args.vision_provider or None)
        payload = {"analysis": analysis}
        if args.remember:
            payload["saved_memory"] = memory_store.save_visual_analysis(args.session_id, analysis)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "eval":
        report = run_eval_suite(root)
        print(f"passed {report['passed']} / {report['total']}")
        for item in report["results"]:
            print(item)
        return

    if args.command == "report":
        print(json.dumps(build_report(root), ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        query = args.query
        image_analysis = None
        if args.image_path:
            ensure_sample_docs(root)
            image_analysis = analyze_meal_image(args.image_path, provider=args.vision_provider or None)
            query = f"{query}\nVisual context: {analysis_to_text(image_analysis)}"

        index = build_index(root)
        hits, trace = hybrid_retrieve(
            query,
            index,
            top_k=5,
            memory_store=memory_store,
            session_id=args.session_id,
            image_analysis=image_analysis,
        )
        hits = rerank_hits(query, hits, index, top_k=3)
        trace["reranked_hits"] = serialize_hits(hits)
        answer = build_grounded_answer(query, hits, index, trace=trace)
        print(answer["answer"])
        print(json.dumps({
            "profile_summary": memory_store.build_profile_summary(args.session_id),
            "trace": trace,
            "answer_trace": answer["trace"],
            "citations": answer["citations"],
        }, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()

