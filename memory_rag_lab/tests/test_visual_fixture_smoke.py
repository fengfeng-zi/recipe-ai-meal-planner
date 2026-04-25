from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISION_PATH = PROJECT_ROOT / "src" / "memory_rag_lab" / "vision.py"
FIXTURE_DIR = PROJECT_ROOT / "examples" / "visual_fixtures"

spec = importlib.util.spec_from_file_location("memory_rag_lab_vision", VISION_PATH)
vision = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(vision)


class TestMemoryVisualFixtures(unittest.TestCase):
    def test_manifest_fixtures(self) -> None:
        manifest = json.loads((FIXTURE_DIR / "fixture_manifest.json").read_text(encoding="utf-8-sig"))
        for item in manifest["fixtures"]:
            with self.subTest(image=item["image"]):
                image_path = FIXTURE_DIR / item["image"]
                analysis = vision.analyze_meal_image(image_path)
                self.assertEqual(analysis.get("provider"), item["expected_provider"])

                tags = set(vision.analysis_tags(analysis))
                for expected in item.get("expected_tags", []):
                    self.assertIn(expected, tags)

    def test_rich_recipe_fields_are_backward_compatible(self) -> None:
        image_path = FIXTURE_DIR / "salmon_plate_demo.jpg"
        analysis = vision.analyze_meal_image(image_path)

        # Backward-compatible required keys.
        for key in (
            "dish_name",
            "meal_type",
            "visible_ingredients",
            "cuisine_tags",
            "nutrition_signals",
            "caution_tags",
            "summary",
            "confidence",
        ):
            self.assertIn(key, analysis)

        # New recipe-oriented fields.
        self.assertIn("cooking_method", analysis)
        self.assertIn("estimated_portions", analysis)
        self.assertIn("recipe_cues", analysis)
        self.assertIsInstance(analysis["recipe_cues"], list)
        self.assertGreaterEqual(float(analysis["estimated_portions"]), 1.0)
        self.assertLessEqual(float(analysis["confidence"]), 1.0)
        self.assertGreaterEqual(float(analysis["confidence"]), 0.0)

    def test_filename_fallback_populates_recipe_cues(self) -> None:
        image_path = FIXTURE_DIR / "peanut_tofu_lunch_demo.jpg"
        analysis = vision.analyze_meal_image(image_path)
        self.assertEqual(analysis.get("provider"), "filename-fallback")
        self.assertIn("recipe_cues", analysis)
        self.assertTrue(analysis["recipe_cues"])
        tags = set(vision.analysis_tags(analysis))
        self.assertIn("lunch", tags)
        self.assertIn("peanut", tags)


if __name__ == "__main__":
    unittest.main()
