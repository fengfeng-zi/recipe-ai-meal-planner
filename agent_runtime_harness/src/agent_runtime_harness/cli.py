from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analytics import build_report
from .evals import run_eval_suite
from .runtime import AgentRuntime


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_input_text(args: argparse.Namespace) -> str:
    return args.input_text or Path(args.input_file).read_text(encoding="utf-8")


def _parse_profile(args: argparse.Namespace) -> dict:
    profile: dict = {}
    if args.profile_json:
        profile.update(json.loads(args.profile_json))
    if args.allergies:
        profile["allergies"] = [item.strip() for item in args.allergies.split(",") if item.strip()]
    if args.dislikes:
        profile["dislikes"] = [item.strip() for item in args.dislikes.split(",") if item.strip()]
    if args.preferences:
        profile["preferences"] = [item.strip() for item in args.preferences.split(",") if item.strip()]
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recipe multi-agent meal-planning harness with optional meal-photo analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a meal-planning assistant task")
    run_parser.add_argument("--task", required=True)
    group = run_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-text")
    group.add_argument("--input-file")
    run_parser.add_argument("--required-terms", default="")
    run_parser.add_argument("--session-id", default="demo")
    run_parser.add_argument("--goal", default="")
    run_parser.add_argument("--planning-days", type=int, default=7)
    run_parser.add_argument("--daily-kcal-target", type=int, default=2000)
    run_parser.add_argument("--max-output-chars", type=int, default=2600)
    run_parser.add_argument("--stop-after-step", type=int, default=0)
    run_parser.add_argument("--allergies", default="")
    run_parser.add_argument("--dislikes", default="")
    run_parser.add_argument("--preferences", default="")
    run_parser.add_argument("--profile-json", default="")
    run_parser.add_argument("--vision-provider", default="")
    run_parser.add_argument("--image-path", action="append", default=[])
    run_parser.add_argument("--meal-log", action="append", default=[])

    resume_parser = subparsers.add_parser("resume", help="Load a checkpointed run or continue execution")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--continue-execution", action="store_true")

    subparsers.add_parser("list-runs", help="List saved runs")
    subparsers.add_parser("eval", help="Run the recipe meal-planning eval suite")
    subparsers.add_parser("report", help="Summarize historical meal-planning runtime metrics")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runtime = AgentRuntime(project_root())

    if args.command == "run":
        input_text = _load_input_text(args)
        required_terms = [term.strip() for term in args.required_terms.split(",") if term.strip()]
        run_context = {
            "session_id": args.session_id,
            "goal": args.goal,
            "planning_days": args.planning_days,
            "daily_kcal_target": args.daily_kcal_target,
            "stop_after_step": args.stop_after_step,
            "meal_logs": args.meal_log,
            "image_paths": args.image_path,
            "vision_provider": args.vision_provider,
            "user_profile": _parse_profile(args),
        }
        run = runtime.run_task(
            args.task,
            input_text,
            required_terms=required_terms,
            max_output_chars=args.max_output_chars,
            run_context=run_context,
        )
        print(f"run_id={run.id}")
        print(f"status={run.status}")
        print(f"policy_version={run.policy_version}")
        print(f"checkpoint_stage={run.checkpoint_state.get('stage', '')}")
        print(run.final_output)
        return

    if args.command == "resume":
        run = runtime.resume_task(args.run_id, continue_execution=args.continue_execution)
        print(f"run_id={run.id}")
        print(f"status={run.status}")
        print(f"checkpoint_stage={run.checkpoint_state.get('stage', '')}")
        print(run.final_output)
        return

    if args.command == "list-runs":
        for item in runtime.list_runs():
            print(
                f"{item['run_id']} | {item['status']} | {item['checkpoint_stage']} | "
                f"{item['failure_type'] or 'ok'} | {item['task']}"
            )
        return

    if args.command == "eval":
        report = run_eval_suite(project_root())
        print(f"passed {report['passed']} / {report['total']}")
        for item in report["results"]:
            print(item)
        return

    if args.command == "report":
        print(json.dumps(build_report(project_root()), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
