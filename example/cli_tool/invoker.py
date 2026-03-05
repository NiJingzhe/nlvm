from __future__ import annotations

from typing import Any

from nlvm import LogicExecutionRequest


class TaskCliInvoker:
    """Deterministic invoker for the example task-planning CLI."""

    async def __call__(
        self,
        request: LogicExecutionRequest,
        toolkit: list[Any],
    ) -> dict[str, Any]:
        del toolkit
        name = request.logic.name

        if name == "create_task":
            return await self._create_task(request)
        if name == "list_tasks":
            return await self._list_tasks(request)
        if name == "complete_task":
            return await self._complete_task(request)
        if name == "score_tasks":
            return await self._score_tasks(request)
        if name == "build_day_plan":
            return await self._build_day_plan(request)

        raise RuntimeError(f"Unsupported logic in TaskCliInvoker: {name}")

    async def _create_task(self, request: LogicExecutionRequest) -> dict[str, Any]:
        helpers = request.helper_functions
        payload = request.logic_args["input"]

        existing = await helpers["list_task_records"](payload["workspace"], True)
        duplicates = [
            item
            for item in existing
            if str(item.get("title", "")).strip().lower() == payload["title"].strip().lower()
        ]

        created = await helpers["create_task_record"](payload["workspace"], payload)
        dedupe_hint = None
        if duplicates:
            dedupe_hint = f"Found {len(duplicates)} existing task(s) with similar title; review duplicates later."

        return {
            "task": created,
            "dedupe_hint": dedupe_hint,
        }

    async def _list_tasks(self, request: LogicExecutionRequest) -> dict[str, Any]:
        helpers = request.helper_functions
        payload = request.logic_args["input"]

        tasks = await helpers["list_task_records"](
            payload["workspace"],
            payload.get("include_done", False),
        )
        pending = [item for item in tasks if not bool(item.get("done", False))]
        return {
            "tasks": tasks,
            "total": len(tasks),
            "pending": len(pending),
        }

    async def _complete_task(self, request: LogicExecutionRequest) -> dict[str, Any]:
        helpers = request.helper_functions
        payload = request.logic_args["input"]

        updated = await helpers["mark_task_done"](payload["workspace"], payload["task_id"])
        if updated is None:
            raise RuntimeError(f"Task not found: {payload['task_id']}")

        return {
            "task": updated,
        }

    async def _score_tasks(self, request: LogicExecutionRequest) -> dict[str, Any]:
        helpers = request.helper_functions
        payload = request.logic_args["input"]

        ranked = await helpers["score_task_records"](
            payload["tasks"],
            payload["today"],
        )
        return {
            "ranked": ranked,
        }

    async def _build_day_plan(self, request: LogicExecutionRequest) -> dict[str, Any]:
        helpers = request.helper_functions
        payload = request.logic_args["input"]

        pending = await helpers["list_task_records"](payload["workspace"], False)
        pending = [item for item in pending if not bool(item.get("done", False))]

        scored = await helpers["invoke_logic"](
            "tasks.score#score_tasks",
            {
                "tasks": pending,
                "today": payload["today"],
            },
        )
        ranked = list(scored.get("ranked", []))
        max_items = int(payload["max_items"])
        chosen = ranked[:max_items]

        plan_items = [
            {
                "task_id": item["task_id"],
                "title": item["title"],
                "score": int(item["score"]),
                "reason": ", ".join(item.get("reasons", [])[:2]) or "score_based",
            }
            for item in chosen
        ]

        narrative = (
            f"Plan generated from {len(pending)} pending task(s); "
            f"selected top {len(plan_items)} by priority score."
        )
        return {
            "plan": plan_items,
            "total_pending": len(pending),
            "narrative": narrative,
        }
