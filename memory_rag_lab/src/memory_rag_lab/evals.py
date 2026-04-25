from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .answering import build_grounded_answer
from .chunking import chunk_document
from .documents import load_documents
from .index import SparseIndex
from .memory import SessionMemoryStore
from .rerank import rerank_hits
from .retrieval import hybrid_retrieve, serialize_hits
from .vision import analysis_to_text, analyze_meal_image

EVAL_SESSION_ID = "eval_recipe_default"


@dataclass(slots=True)
class EvalCase:
    case_id: str
    query: str
    required_terms: list[str]
    requires_memory: bool = False
    image_path: str = ""
    requires_image_trace: bool = False
    expected_top_meal_type: str = ""
    requires_visual_grounding: bool = False


DEFAULT_EVAL_CASES = [
    EvalCase(
        case_id="allergy_guardrail",
        query="I am allergic to peanuts. Suggest a quick dinner with chicken and vegetables.",
        required_terms=["peanut", "vegetable"],
        requires_memory=True,
    ),
    EvalCase(
        case_id="muscle_gain_goal",
        query="Plan a high-protein lunch idea for my muscle gain goal.",
        required_terms=["protein", "lunch"],
        requires_memory=True,
        expected_top_meal_type="lunch",
    ),
    EvalCase(
        case_id="breakfast_log_recall",
        query="What did I log for breakfast and what should I adjust?",
        required_terms=["breakfast", "adjust"],
        requires_memory=True,
        expected_top_meal_type="breakfast",
    ),
    EvalCase(
        case_id="trace_explainability",
        query="Which trace fields explain the ranking decision for this recipe answer?",
        required_terms=["trace", "ranking"],
    ),
    EvalCase(
        case_id="visual_query_offline",
        query="Give me a dinner suggestion from this plate with high protein and vegetables.",
        required_terms=["dinner", "protein", "visual"],
        image_path="salmon_broccoli_dinner.jpg",
        requires_image_trace=True,
        expected_top_meal_type="dinner",
        requires_visual_grounding=True,
    ),
    EvalCase(
        case_id="meal_intent_dinner_precision",
        query="I want a quick high-protein dinner. Avoid breakfast and lunch suggestions.",
        required_terms=["dinner", "protein"],
        requires_memory=True,
        expected_top_meal_type="dinner",
    ),
]


def ensure_sample_docs(project_root: str | Path) -> Path:
    root = Path(project_root) / "examples"
    root.mkdir(parents=True, exist_ok=True)
    sample_a = root / "agent_runtime.md"
    sample_b = root / "memory_notes.md"
    sample_c = root / "retrieval_trace.md"
    sample_image = root / "salmon_broccoli_dinner.jpg"
    if not sample_a.exists():
        sample_a.write_text(
            """Recipe: Lemon Chicken Tray Dinner\nMeal type: dinner\nPrep time: 20 minutes\nCalories: 520 kcal\nProtein: 41g protein\nIngredients: chicken breast, broccoli, carrot, lemon, garlic, olive oil\nBenefits: Quick peanut-free dinner with vegetables and strong satiety.\nUse when: You need a fast weeknight dinner after work.\n""",
            encoding="utf-8",
        )
    if not sample_b.exists():
        sample_b.write_text(
            """Recipe: Greek Yogurt Berry Oats\nMeal type: breakfast\nPrep time: 8 minutes\nCalories: 410 kcal\nProtein: 31g protein\nIngredients: greek yogurt, oats, berries, chia seeds, cinnamon\nBenefits: High-protein breakfast that helps reduce rebound hunger later in the morning.\nUse when: The user often skips breakfast or wants better satiety.\n""",
            encoding="utf-8",
        )
    if not sample_c.exists():
        sample_c.write_text(
            """Recipe: Tofu Edamame Grain Bowl\nMeal type: lunch\nPrep time: 18 minutes\nCalories: 520 kcal\nProtein: 32g protein\nIngredients: firm tofu, edamame, brown rice, cucumber, sesame\nBenefits: Plant-forward lunch that still delivers high protein.\nUse when: The user wants variety without losing protein density.\n""",
            encoding="utf-8",
        )
    if not sample_image.exists():
        sample_image.write_bytes(b"offline-visual-demo")
    return root


def _top_hit_meal_type(hits: list, index: SparseIndex) -> str:
    if not hits:
        return ""
    text = index.chunks[hits[0].chunk_id].text
    match = re.search(r"meal type:\s*(breakfast|lunch|dinner|snack)", text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).lower()


def run_eval_suite(project_root: str | Path) -> dict:
    docs_root = ensure_sample_docs(project_root)
    documents = load_documents(docs_root)
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy="hybrid"))
    index = SparseIndex(chunks)
    memory_store = SessionMemoryStore(Path(project_root) / "data" / "memories")
    memory_store.clear_memories(EVAL_SESSION_ID)

    memory_store.save_preference(
        EVAL_SESSION_ID,
        memory_text="Avoid peanuts and peanut sauces.",
        memory_type="allergy",
        tags=["peanut"],
    )
    memory_store.save_preference(
        EVAL_SESSION_ID,
        memory_text="Need high-protein lunches for muscle gain.",
        memory_type="goal",
        tags=["high-protein", "lunch", "muscle"],
    )
    memory_store.save_preference(
        EVAL_SESSION_ID,
        memory_text="Prefer quick weekday dinners with vegetables.",
        memory_type="preference",
        tags=["quick", "dinner", "vegetable"],
    )
    memory_store.log_meal(
        EVAL_SESSION_ID,
        meal_name="banana oatmeal",
        meal_type="breakfast",
        calories=320,
        ingredients=["oats", "banana"],
        notes="Hungry again in 2 hours",
    )

    results = []
    for case in DEFAULT_EVAL_CASES:
        effective_query = case.query
        image_analysis = None
        if case.image_path:
            image_path = docs_root / case.image_path
            image_analysis = analyze_meal_image(image_path)
            effective_query = f"{effective_query}\nVisual context: {analysis_to_text(image_analysis)}"

        hits, trace = hybrid_retrieve(
            effective_query,
            index,
            top_k=5,
            memory_store=memory_store,
            session_id=EVAL_SESSION_ID,
            image_analysis=image_analysis,
        )
        hits = rerank_hits(effective_query, hits, index, top_k=3)
        trace["reranked_hits"] = serialize_hits(hits)

        answer = build_grounded_answer(effective_query, hits, index, trace=trace)
        missing = [term for term in case.required_terms if term.lower() not in answer["answer"].lower()]
        memory_ok = not case.requires_memory or answer["trace"]["memory_augmented_hits"] > 0
        image_ok = not case.requires_image_trace or bool(trace.get("image_analysis"))
        top_meal_type = _top_hit_meal_type(hits, index)
        meal_intent_ok = not case.expected_top_meal_type or top_meal_type == case.expected_top_meal_type.lower()
        visual_grounding_ok = True
        if case.requires_visual_grounding:
            visual_grounding_ok = bool(
                trace.get("image_analysis")
                and any(
                    float(item.get("vision_alignment", 0.0)) > 0.0
                    or float(item.get("visual_memory_alignment", 0.0)) > 0.0
                    for item in trace.get("reranked_hits", [])
                )
            )
        results.append(
            {
                "case_id": case.case_id,
                "passed": (
                    not missing
                    and len(answer["citations"]) > 0
                    and memory_ok
                    and image_ok
                    and meal_intent_ok
                    and visual_grounding_ok
                ),
                "missing_terms": missing,
                "citation_count": len(answer["citations"]),
                "memory_hit_count": len(trace["memory_hits"]),
                "memory_augmented_hits": answer["trace"]["memory_augmented_hits"],
                "image_used": image_analysis is not None,
                "image_provider": (image_analysis or {}).get("provider", ""),
                "top_meal_type": top_meal_type,
                "expected_top_meal_type": case.expected_top_meal_type,
                "meal_intent_ok": meal_intent_ok,
                "visual_grounding_ok": visual_grounding_ok,
            }
        )
    return {"total": len(results), "passed": sum(1 for item in results if item["passed"]), "results": results}
