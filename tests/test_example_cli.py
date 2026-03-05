from __future__ import annotations

from pathlib import Path

import pytest

from nlvm import NLRuntime

from example.cli_tool.helpers import configure_logic_caller
from example.cli_tool.invoker import TaskCliInvoker
from example.cli_tool.types import BuildPlanOutput, CreateTaskOutput


def _modules_root() -> Path:
    return Path(__file__).resolve().parents[1] / "example" / "nl_modules"


def _build_runtime() -> NLRuntime:
    runtime = NLRuntime(
        modules_root=_modules_root(),
        invoker=TaskCliInvoker(),
        include_pyrepl_tools=False,
    )

    async def _call_logic(logic_ref: str, payload: dict[str, object]) -> object:
        logic_func = runtime.use(logic_ref)
        return await logic_func(payload)

    configure_logic_caller(_call_logic)
    return runtime


@pytest.mark.asyncio
async def test_example_cli_runtime_supports_nested_logic_calls(tmp_path: Path) -> None:
    runtime = _build_runtime()
    create_task = runtime.use("tasks.create#create_task")
    list_tasks = runtime.use("tasks.list#list_tasks")
    build_plan = runtime.use("tasks.plan#build_day_plan")
    complete_task = runtime.use("tasks.complete#complete_task")

    workspace = str(tmp_path / "workspace")

    try:
        first = await create_task(
            {
                "workspace": workspace,
                "title": "Ship changelog",
                "details": "Capture highlights for this week",
                "effort": 2,
                "tags": ["urgent", "quick"],
                "due_date": "2026-03-01",
            }
        )
        second = await create_task(
            {
                "workspace": workspace,
                "title": "Refactor CLI parser",
                "details": "Split parser by command",
                "effort": 4,
                "tags": ["techdebt"],
                "due_date": None,
            }
        )

        assert isinstance(first, CreateTaskOutput)
        assert isinstance(second, CreateTaskOutput)

        listed = await list_tasks({"workspace": workspace, "include_done": False})
        listed_payload = listed.model_dump(mode="json")
        assert listed_payload["total"] == 2

        plan_before = await build_plan(
            {
                "workspace": workspace,
                "today": "2026-03-01",
                "max_items": 2,
            }
        )
        assert isinstance(plan_before, BuildPlanOutput)
        assert plan_before.plan

        top_task_id = plan_before.plan[0].task_id
        await complete_task({"workspace": workspace, "task_id": top_task_id})

        plan_after = await build_plan(
            {
                "workspace": workspace,
                "today": "2026-03-01",
                "max_items": 2,
            }
        )
        assert all(item.task_id != top_task_id for item in plan_after.plan)
    finally:
        configure_logic_caller(None)
