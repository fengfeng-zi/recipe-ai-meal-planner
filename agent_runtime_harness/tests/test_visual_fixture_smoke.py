from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISION_PATH = PROJECT_ROOT / "src" / "agent_runtime_harness" / "vision.py"
FIXTURE_DIR = PROJECT_ROOT / "examples" / "visual_fixtures"

spec = importlib.util.spec_from_file_location("agent_runtime_harness_vision", VISION_PATH)
vision = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(vision)


class TestHarnessVisualFixtures(unittest.TestCase):
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

    def test_structured_fields_present_in_sidecar_and_fallback(self) -> None:
        sidecar_image = FIXTURE_DIR / "chicken_rice_bowl_demo.jpg"
        sidecar_analysis = vision.analyze_meal_image(sidecar_image)
        self.assertIn("cooking_method", sidecar_analysis)
        self.assertIn("estimated_portions", sidecar_analysis)
        self.assertIn("recipe_cues", sidecar_analysis)
        self.assertIsInstance(sidecar_analysis["estimated_portions"], float)
        self.assertIsInstance(sidecar_analysis["recipe_cues"], list)

        fallback_image = FIXTURE_DIR / "peanut-noodle-dinner-demo.jpg"
        fallback_analysis = vision.analyze_meal_image(fallback_image)
        self.assertIn("cooking_method", fallback_analysis)
        self.assertIn("estimated_portions", fallback_analysis)
        self.assertIn("recipe_cues", fallback_analysis)
        self.assertGreaterEqual(float(fallback_analysis["estimated_portions"]), 1.0)

    def test_extract_structured_analysis_from_tool_calls(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": json.dumps(
                                        {
                                            "dish_name": "Salmon Tray",
                                            "meal_type": "supper",
                                            "visible_ingredients": ["salmon", "asparagus"],
                                            "cuisine_tags": ["tray-bake"],
                                            "nutrition_signals": ["high-protein"],
                                            "caution_tags": [],
                                            "summary": "Sheet pan salmon dinner",
                                            "confidence": "high",
                                            "cooking_method": "baked",
                                            "estimated_portions": "2",
                                            "recipe_cues": ["sheet-pan-friendly"],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                }
            ]
        }
        parsed = vision._extract_structured_analysis(payload)
        self.assertIsNotNone(parsed)
        analysis = vision._build_analysis(parsed, FIXTURE_DIR / "salmon_tray.jpg", "test-provider")
        self.assertEqual(analysis["meal_type"], "dinner")
        self.assertEqual(analysis["cooking_method"], "baked")
        self.assertEqual(analysis["confidence"], 0.8)
        self.assertEqual(analysis["estimated_portions"], 2.0)
        self.assertIn("sheet-pan-friendly", analysis["recipe_cues"])


if __name__ == "__main__":
    unittest.main()
