from __future__ import annotations

from agent_runtime_harness.tools import draft_meal_plan


def _meal_lines(answer: str, meal_type: str) -> list[str]:
    prefix = f"- {meal_type.title()}: "
    return [line for line in answer.splitlines() if line.startswith(prefix)]


def _meal_names(answer: str, meal_type: str) -> list[str]:
    names: list[str] = []
    for line in _meal_lines(answer, meal_type):
        names.append(line.split(": ", 1)[1].split(" (", 1)[0].strip())
    return names


def test_draft_meal_plan_respects_full_requested_horizon() -> None:
    payload = {
        "task": "Plan 7-day meals",
        "input_text": "Need quick high-protein meals without peanuts.",
        "planning_days": 7,
        "goal": "fat loss",
        "user_profile": {"allergies": ["peanut"], "dislikes": []},
        "keywords": ["quick", "high", "protein"],
    }
    answer = draft_meal_plan(payload)["answer"]
    day_count = sum(1 for line in answer.splitlines() if line.startswith("Day "))
    assert day_count == 7


def test_long_horizon_rotates_dinner_choices() -> None:
    payload = {
        "task": "Plan 6-day meals",
        "input_text": "Need quick high-protein dinners for weekdays.",
        "planning_days": 6,
        "goal": "fat loss",
        "user_profile": {"allergies": [], "dislikes": []},
        "keywords": ["quick", "high", "protein", "dinner"],
    }
    answer = draft_meal_plan(payload)["answer"]
    dinners = _meal_names(answer, "dinner")
    assert len(dinners) == 6
    assert len(set(dinners)) >= 2


def test_constraints_still_filter_unsafe_recipe() -> None:
    payload = {
        "task": "Plan 5-day meals",
        "input_text": "Need a balanced plan without salmon.",
        "planning_days": 5,
        "goal": "general healthy eating",
        "user_profile": {"allergies": ["salmon"], "dislikes": []},
        "keywords": ["balanced"],
    }
    answer = draft_meal_plan(payload)["answer"]
    assert "Garlic Salmon Veggie Tray" not in answer
