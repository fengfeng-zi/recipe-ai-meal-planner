# Memory RAG Lab

Memory-aware recipe retrieval with grounded answers and offline-first meal image support.

## What It Does

- loads recipe-style documents from `.txt`, `.md`, `.json`
- stores typed food memory such as `allergy`, `dislike`, `preference`, `goal`, `meal_log`
- retrieves candidates with explicit score components
- produces citation-backed answers and retrieval traces
- supports meal image analysis through sidecars, filename fallback, or optional live API mode

## Quick Start

```powershell
cd <repo-root>
$root = (Get-Location).Path
$env:PYTHONPATH = "$root\memory_rag_lab\src"

python -m memory_rag_lab.cli remember --session-id demo --memory-type allergy --tags peanut --text "Avoid peanuts and peanut sauces."
python -m memory_rag_lab.cli log-meal --session-id demo --meal-name "banana oatmeal" --meal-type breakfast --calories 320 --ingredients oats,banana --notes "Hungry again in 2 hours"
python -m memory_rag_lab.cli analyze-image --image-path ".\memory_rag_lab\examples\salmon_broccoli_dinner.jpg" --session-id demo --remember
python -m memory_rag_lab.cli ask --query "I need a quick high-protein dinner with vegetables and no peanuts" --image-path ".\memory_rag_lab\examples\salmon_broccoli_dinner.jpg" --session-id demo
python -m memory_rag_lab.cli eval
python -m memory_rag_lab.cli report
```

## Key Files

- `src/memory_rag_lab/service.py`: end-to-end query orchestration
- `src/memory_rag_lab/retrieval.py`: score computation and retrieval trace
- `src/memory_rag_lab/memory.py`: typed session memory store
- `src/memory_rag_lab/vision.py`: image analysis and sidecar fallback
- `src/memory_rag_lab/evals.py`: demo fixtures and regression cases

## Examples

- `examples/salmon_broccoli_dinner.jpg`
- `examples/visual_fixtures/`
- `examples/images/`

