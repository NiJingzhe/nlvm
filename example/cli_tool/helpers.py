from __future__ import annotations

import json
from datetime import date, datetime, timezone
import os
from pathlib import Path
from typing import Any
import uuid

from example.cli_tool.types import CreateTaskInput, Task, TaskScore


_LOGIC_CALLER: Any = None
_FALLBACK_LOGIC_CALLER: Any = None


def configure_logic_caller(caller: Any) -> None:
    """Register a callable used by invoke_logic helper."""

    global _LOGIC_CALLER
    _LOGIC_CALLER = caller


def _clean_env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned


def _build_fallback_logic_caller() -> Any:
    modules_root_text = _clean_env_value("NLVM_MODULES_ROOT")
    if modules_root_text is None:
        raise RuntimeError("Logic caller is not configured")

    modules_root = Path(modules_root_text).expanduser().resolve()
    executor = (_clean_env_value("NLVM_EXECUTOR") or "deterministic").lower()
    is_daemon_process = False
    try:
        from multiprocessing import current_process

        is_daemon_process = bool(current_process().daemon)
    except Exception:
        is_daemon_process = False

    if executor == "llm":
        if is_daemon_process:
            from nlvm import NLRuntime

            from example.cli_tool.invoker import TaskCliInvoker

            runtime = NLRuntime(
                modules_root=modules_root,
                invoker=TaskCliInvoker(),
                include_pyrepl_tools=False,
            )

            async def _call_logic(logic_ref: str, payload: dict[str, Any]) -> Any:
                logic_func = runtime.use(logic_ref)
                return await logic_func(payload)

            return _call_logic

        providers_path_text = _clean_env_value("NLVM_PROVIDERS_PATH")
        if providers_path_text is None:
            raise RuntimeError("Logic caller is not configured")

        provider_id = _clean_env_value("NLVM_DEFAULT_PROVIDER")
        model_name = _clean_env_value("NLVM_DEFAULT_MODEL")
        if (provider_id is None) != (model_name is None):
            raise RuntimeError("NLVM_DEFAULT_PROVIDER and NLVM_DEFAULT_MODEL must be set together")

        from nlvm import build_runtime_from_provider_config

        runtime = build_runtime_from_provider_config(
            modules_root=modules_root,
            providers_path=Path(providers_path_text).expanduser().resolve(),
            provider_id=provider_id,
            model_name=model_name,
            include_pyrepl_tools=True,
        )
    else:
        from nlvm import NLRuntime

        from example.cli_tool.invoker import TaskCliInvoker

        runtime = NLRuntime(
            modules_root=modules_root,
            invoker=TaskCliInvoker(),
            include_pyrepl_tools=False,
        )

    async def _call_logic(logic_ref: str, payload: dict[str, Any]) -> Any:
        logic_func = runtime.use(logic_ref)
        return await logic_func(payload)

    return _call_logic


def _resolve_logic_caller() -> Any:
    if _LOGIC_CALLER is not None:
        return _LOGIC_CALLER

    global _FALLBACK_LOGIC_CALLER
    if _FALLBACK_LOGIC_CALLER is None:
        _FALLBACK_LOGIC_CALLER = _build_fallback_logic_caller()
    return _FALLBACK_LOGIC_CALLER


def _workspace_path(workspace: str) -> Path:
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / "tasks.json"


def _load_tasks(workspace: str) -> list[Task]:
    data_path = _workspace_path(workspace)
    if not data_path.exists():
        return []

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Task storage file must contain a list")
    return [Task.model_validate(item) for item in raw]


def _save_tasks(workspace: str, tasks: list[Task]) -> None:
    data_path = _workspace_path(workspace)
    payload = [item.model_dump(mode="json") for item in tasks]
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def create_task_record(
    workspace: str,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    normalized_payload: dict[str, Any]
    if payload is None:
        normalized_payload = {"workspace": workspace}
        normalized_payload.update(kwargs)
    else:
        normalized_payload = dict(payload)
        normalized_payload.setdefault("workspace", workspace)
        if kwargs:
            normalized_payload.update(kwargs)

    request = CreateTaskInput.model_validate(normalized_payload)
    now = datetime.now(timezone.utc)

    tasks = _load_tasks(workspace)
    task = Task(
        id=f"tsk_{uuid.uuid4().hex[:8]}",
        title=request.title,
        details=request.details,
        effort=request.effort,
        tags=request.tags,
        due_date=request.due_date,
        done=False,
        created_at=now,
        updated_at=now,
    )
    tasks.append(task)
    _save_tasks(workspace, tasks)
    return task.model_dump(mode="json")


async def list_task_records(
    workspace: str | None = None,
    include_done: bool = False,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    options: dict[str, Any] = {}
    if payload is not None:
        options.update(payload)
    if kwargs:
        options.update(kwargs)

    if workspace is None:
        candidate_workspace = options.pop("workspace", None)
        workspace = str(candidate_workspace) if candidate_workspace is not None else None
    if workspace is None:
        raise ValueError("list_task_records requires workspace")

    if "include_done" in options:
        include_done = bool(options.pop("include_done"))

    status_filter_value = options.pop("status_filter", None)
    status_filter = (
        str(status_filter_value).strip().lower() if status_filter_value is not None else None
    )
    if status_filter in {"all", "any"}:
        include_done = True

    due_date_filter: date | None = None
    date_filter_value = options.pop("date_filter", None)
    if isinstance(date_filter_value, date):
        due_date_filter = date_filter_value
    elif isinstance(date_filter_value, str):
        cleaned_date = date_filter_value.strip()
        if cleaned_date:
            due_date_filter = date.fromisoformat(cleaned_date)

    tasks = _load_tasks(workspace)
    visible = [item for item in tasks if include_done or not item.done]

    if status_filter in {"pending", "open", "todo"}:
        visible = [item for item in visible if not item.done]
    elif status_filter in {"done", "completed", "closed"}:
        visible = [item for item in tasks if item.done]

    if due_date_filter is not None:
        visible = [item for item in visible if item.due_date == due_date_filter]

    def _sort_key(task: Task) -> tuple[int, str, int]:
        due_sort = task.due_date.isoformat() if task.due_date is not None else "9999-12-31"
        return (1 if task.done else 0, due_sort, task.effort)

    visible.sort(key=_sort_key)
    return [item.model_dump(mode="json") for item in visible]


async def mark_task_done(
    workspace: str | None = None,
    task_id: str | None = None,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    options: dict[str, Any] = {}
    if payload is not None:
        options.update(payload)
    if kwargs:
        options.update(kwargs)

    if workspace is None:
        candidate_workspace = options.pop("workspace", None)
        workspace = str(candidate_workspace) if candidate_workspace is not None else None
    if task_id is None:
        task_id_value = (
            options.pop("task_id", None) or options.pop("taskId", None) or options.pop("id", None)
        )
        task_id = str(task_id_value) if task_id_value is not None else None

    if workspace is None or task_id is None:
        raise ValueError("mark_task_done requires workspace and task_id")

    tasks = _load_tasks(workspace)
    now = datetime.now(timezone.utc)
    updated: Task | None = None
    for task in tasks:
        if task.id != task_id:
            continue
        task.done = True
        task.updated_at = now
        updated = task
        break

    if updated is None:
        return None

    _save_tasks(workspace, tasks)
    return updated.model_dump(mode="json")


async def score_task_records(
    tasks: list[dict[str, Any]] | None = None,
    today: str | None = None,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    merged_options: dict[str, Any] = {}
    if payload is not None:
        merged_options.update(payload)
    if kwargs:
        merged_options.update(kwargs)

    embedded_input = merged_options.get("input")
    if isinstance(embedded_input, dict):
        for key, value in embedded_input.items():
            merged_options.setdefault(key, value)

    if payload is not None:
        if tasks is None:
            tasks = payload.get("tasks")
        if today is None:
            today = payload.get("today")

    if tasks is None:
        tasks = merged_options.get("tasks")
    if today is None:
        today = merged_options.get("today")
    if today is None:
        today = merged_options.get("date")
    if tasks is None or today is None:
        raise ValueError("score_task_records requires tasks and today")

    today_date = date.fromisoformat(today)
    parsed_tasks = [Task.model_validate(item) for item in tasks]
    ranked: list[TaskScore] = []

    for task in parsed_tasks:
        score = 0
        reasons: list[str] = []

        if task.due_date is not None:
            days_left = (task.due_date - today_date).days
            if days_left < 0:
                score += 120
                reasons.append("overdue")
            elif days_left == 0:
                score += 95
                reasons.append("due_today")
            elif days_left <= 2:
                score += 75
                reasons.append("due_soon")
            elif days_left <= 7:
                score += 35
                reasons.append("due_this_week")
            else:
                score += 10
                reasons.append("future_due")
        else:
            score += 15
            reasons.append("no_due_date")

        effort_bonus = max(0, 6 - int(task.effort)) * 8
        score += effort_bonus
        reasons.append(f"effort_{task.effort}")

        normalized_tags = {tag.lower() for tag in task.tags}
        if "urgent" in normalized_tags:
            score += 40
            reasons.append("tag_urgent")
        if "quick" in normalized_tags:
            score += 20
            reasons.append("tag_quick")
        if "blocked" in normalized_tags:
            score -= 30
            reasons.append("tag_blocked")

        if task.done:
            score -= 500
            reasons.append("already_done")

        ranked.append(
            TaskScore(
                task_id=task.id,
                title=task.title,
                score=score,
                reasons=reasons,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return [item.model_dump(mode="json") for item in ranked]


async def invoke_logic(
    logic_ref: str | None = None,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    normalized_payload: dict[str, Any] = {}
    if payload is not None:
        normalized_payload.update(payload)
    if kwargs:
        normalized_payload.update(kwargs)

    if logic_ref is None:
        for key in ("logic_ref", "logic_path", "logic", "logic_id"):
            candidate = normalized_payload.pop(key, None)
            if isinstance(candidate, str) and candidate.strip():
                logic_ref = candidate.strip()
                break

    if logic_ref is None:
        raise ValueError("invoke_logic requires logic_ref")

    embedded_payload = normalized_payload.pop("payload", None)
    if isinstance(embedded_payload, dict):
        normalized_payload = {**embedded_payload, **normalized_payload}

    embedded_input = normalized_payload.pop("input", None)
    if isinstance(embedded_input, dict):
        normalized_payload = {**embedded_input, **normalized_payload}

    logic_caller = _resolve_logic_caller()
    result = await logic_caller(logic_ref, normalized_payload)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result
