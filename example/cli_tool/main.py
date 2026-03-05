from __future__ import annotations

import argparse
import asyncio
from datetime import date
import os
from pathlib import Path
from typing import Any

from nlvm import NLRuntime, build_runtime_from_provider_config

from example.cli_tool.helpers import configure_logic_caller


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlvm-task-cli",
        description="Runtime-intelligent TODO scheduler",
    )
    parser.add_argument(
        "request",
        nargs="*",
        help="Natural language plan request. If omitted, interactive input will be used.",
    )
    parser.add_argument(
        "--workspace",
        default="./.nlvm_cli_workspace",
        help="Workspace directory for todo storage",
    )
    parser.add_argument(
        "--providers",
        default="./providers.json",
        help="Provider config JSON path for LLM runtime",
    )
    parser.add_argument(
        "--today",
        default=date.today().isoformat(),
        help="Planning date in YYYY-MM-DD",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Max number of recommended tasks in schedule",
    )

    return parser


def _modules_root() -> Path:
    return Path(__file__).resolve().parents[1] / "nl_modules"


def _configure_nested_logic_env(args: argparse.Namespace) -> None:
    os.environ["NLVM_MODULES_ROOT"] = str(_modules_root().resolve())
    os.environ["NLVM_EXECUTOR"] = "llm"
    os.environ["NLVM_PROVIDERS_PATH"] = str(Path(args.providers).expanduser().resolve())
    os.environ.pop("NLVM_DEFAULT_PROVIDER", None)
    os.environ.pop("NLVM_DEFAULT_MODEL", None)


def _build_runtime(args: argparse.Namespace) -> NLRuntime:
    _configure_nested_logic_env(args)

    runtime = build_runtime_from_provider_config(
        modules_root=_modules_root(),
        providers_path=Path(args.providers).expanduser().resolve(),
        include_pyrepl_tools=True,
    )

    async def _call_logic(logic_ref: str, payload: dict[str, Any]) -> Any:
        logic_func = runtime.use(logic_ref)
        return await logic_func(payload)

    configure_logic_caller(_call_logic)
    return runtime


def _read_user_request(args: argparse.Namespace) -> str:
    merged = " ".join(args.request).strip()
    if merged:
        return merged

    while True:
        line = input("请输入你的自然语言计划: ").strip()
        if line:
            return line


def _print_schedule(payload: dict[str, Any]) -> None:
    created_tasks = payload.get("created_tasks", [])
    plan_items = payload.get("plan", [])

    print("\n=== Assistant ===")
    print(payload.get("assistant_message", ""))

    if created_tasks:
        print("\n=== 新增任务 ===")
        for index, task in enumerate(created_tasks, start=1):
            due_date = task.get("due_date") or "未指定"
            print(
                f"{index}. {task.get('title', '')} | due: {due_date} | effort: {task.get('effort', '')}"
            )

    print("\n=== 智能排期 ===")
    if not plan_items:
        print("暂无可推荐任务")
    else:
        for index, item in enumerate(plan_items, start=1):
            print(
                f"{index}. {item.get('title', '')} "
                f"(score={item.get('score', 0)}, reason={item.get('reason', 'score_based')})"
            )

    print(f"\n待办总数: {payload.get('total_pending', 0)}")
    narrative = str(payload.get("narrative", "")).strip()
    if narrative:
        print(f"说明: {narrative}")


async def _run(args: argparse.Namespace) -> None:
    runtime = _build_runtime(args)
    smart_schedule = runtime.use("tasks.assistant#smart_schedule")
    user_text = _read_user_request(args)
    context_notes: list[str] = []
    max_rounds = 3

    try:
        for _ in range(max_rounds):
            result = await smart_schedule(
                {
                    "workspace": args.workspace,
                    "today": args.today,
                    "user_text": user_text,
                    "context_notes": list(context_notes),
                    "max_items": args.max_items,
                }
            )
            payload = result.model_dump(mode="json")

            status = payload.get("status")
            if status == "needs_clarification":
                print("\n=== Assistant ===")
                print(payload.get("assistant_message", ""))

                questions = payload.get("questions", [])
                if not questions:
                    print("未收到澄清问题，结束。")
                    return

                answered = 0
                for question in questions:
                    answer = input(f"\n{question}\n> ").strip()
                    if not answer:
                        continue
                    context_notes.append(f"{question} => {answer}")
                    answered += 1

                if answered == 0:
                    print("未提供澄清信息，结束。")
                    return
                continue

            _print_schedule(payload)
            return

        print("达到最大澄清轮次，建议补充更具体的计划描述后重试。")
    finally:
        configure_logic_caller(None)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
