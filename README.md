# Recipe AI Meal Planner

A standalone project for meal-image understanding, recipe retrieval, and personalized meal planning.

The main user flow is:

1. Upload a meal image or provide an existing image path.
2. Analyze the dish, ingredients, meal type, and nutrition cues.
3. Retrieve grounded recipe evidence with user memory such as allergies, dislikes, goals, and meal logs.
4. Generate a multi-day meal plan and shopping list.

## Repo Layout

- `memory_rag_lab`: memory-aware recipe retrieval, grounded answers, vision helpers, evals
- `agent_runtime_harness`: planning/runtime layer with candidate picking, meal-plan drafting, checkpoints, evals
- `recipe_studio_web`: web demo for upload -> analyze -> retrieve -> plan
- `beefmeal.jpg`, `valid_meal_like.png`: sample local images for demos

## Quick Start

Requirements:

- Python 3.11+

Web demo:

```powershell
cd <repo-root>
python recipe_studio_web\server.py
```

Then open:

```text
http://127.0.0.1:8787
```

CLI evals:

```powershell
cd <repo-root>
$root = (Get-Location).Path
$env:PYTHONPATH = "$root\memory_rag_lab\src;$root\agent_runtime_harness\src"
python -m memory_rag_lab.cli eval
python -m agent_runtime_harness.cli eval
python recipe_studio_web\tests.py
```

Sample image paths you can paste into the UI:

- `valid_meal_like.png`
- `beefmeal.jpg`
- `memory_rag_lab\examples\salmon_broccoli_dinner.jpg`

## What This Project Demonstrates

- Offline-first meal image analysis with optional OpenAI-compatible vision calls
- Typed food memory: allergy, dislike, preference, goal, meal log
- Retrieval traces and citation-backed recipe evidence
- Personalized meal planning and shopping-list generation
- A lightweight web UI for end-to-end demo flows

## Notes

- Runtime outputs such as uploads, run logs, and checkpoints are ignored by default.
- Example fixtures are kept in the repo so the demo remains reproducible.

