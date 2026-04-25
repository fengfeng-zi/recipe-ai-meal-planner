from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_rag_lab.vision import analysis_tags, analyze_meal_image  # noqa: E402


def main() -> None:
    fixtures_dir = Path(__file__).resolve().parent
    manifest = json.loads((fixtures_dir / "fixture_manifest.json").read_text(encoding="utf-8-sig"))
    failures: list[str] = []

    for item in manifest.get("fixtures", []):
        image_path = fixtures_dir / item["image"]
        expected_provider = item["expected_provider"]
        expected_tags = set(item.get("expected_tags", []))

        analysis = analyze_meal_image(image_path)
        tags = set(analysis_tags(analysis))

        if analysis.get("provider") != expected_provider:
            failures.append(
                f"{image_path.name}: provider={analysis.get('provider')} expected={expected_provider}"
            )

        missing = sorted(expected_tags - tags)
        if missing:
            failures.append(f"{image_path.name}: missing tags {missing}; actual={sorted(tags)}")

    if failures:
        print("visual fixture check: FAILED")
        for failure in failures:
            print("- " + failure)
        raise SystemExit(1)

    print("visual fixture check: PASSED")


if __name__ == "__main__":
    main()
