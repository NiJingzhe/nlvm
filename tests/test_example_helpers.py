from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import example.cli_tool.helpers as helpers


def _modules_root() -> Path:
    return Path(__file__).resolve().parents[1] / "example" / "nl_modules"


@pytest.fixture(autouse=True)
def _reset_logic_callers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "_LOGIC_CALLER", None)
    monkeypatch.setattr(helpers, "_FALLBACK_LOGIC_CALLER", None)


@pytest.mark.asyncio
async def test_list_task_records_supports_status_and_date_filters(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")

    due_today = await helpers.create_task_record(
        workspace,
        {
            "title": "Due today",
            "details": "high priority",
            "effort": 2,
            "tags": ["urgent"],
            "due_date": "2026-03-01",
        },
    )
    due_tomorrow = await helpers.create_task_record(
        workspace,
        {
            "title": "Due tomorrow",
            "details": "later",
            "effort": 3,
            "tags": [],
            "due_date": "2026-03-02",
        },
    )
    await helpers.mark_task_done(payload={"workspace": workspace, "taskId": due_tomorrow["id"]})

    filtered = await helpers.list_task_records(
        payload={
            "workspace": workspace,
            "include_done": True,
            "status_filter": "pending",
            "date_filter": "2026-03-01",
        }
    )

    assert [item["id"] for item in filtered] == [due_today["id"]]


@pytest.mark.asyncio
async def test_mark_task_done_accepts_taskid_alias(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")

    created = await helpers.create_task_record(
        workspace,
        {
            "title": "Document helper aliases",
            "details": None,
            "effort": 2,
            "tags": [],
            "due_date": None,
        },
    )

    updated = await helpers.mark_task_done(
        payload={"workspace": workspace, "taskId": created["id"]}
    )
    listed = await helpers.list_task_records(workspace=workspace, include_done=True)

    assert updated is not None
    assert updated["done"] is True
    assert any(item["id"] == created["id"] and item["done"] is True for item in listed)


@pytest.mark.asyncio
async def test_score_task_records_accepts_embedded_input_and_date_alias(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")

    urgent = await helpers.create_task_record(
        workspace,
        {
            "title": "Ship patch",
            "details": None,
            "effort": 1,
            "tags": ["urgent"],
            "due_date": "2026-03-01",
        },
    )
    await helpers.create_task_record(
        workspace,
        {
            "title": "Cleanup notes",
            "details": None,
            "effort": 4,
            "tags": [],
            "due_date": "2026-03-10",
        },
    )
    tasks = await helpers.list_task_records(workspace=workspace, include_done=True)

    ranked = await helpers.score_task_records(
        payload={
            "input": {
                "tasks": tasks,
                "date": "2026-03-01",
            }
        }
    )

    assert ranked
    assert ranked[0]["task_id"] == urgent["id"]
    assert ranked[0]["score"] >= ranked[-1]["score"]


@pytest.mark.asyncio
async def test_invoke_logic_normalizes_logic_path_and_embedded_payload() -> None:
    captured: dict[str, Any] = {}

    async def fake_caller(logic_ref: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["logic_ref"] = logic_ref
        captured["payload"] = payload
        return {"ok": True}

    helpers.configure_logic_caller(fake_caller)
    result = await helpers.invoke_logic(
        payload={
            "logic_path": "tasks.score#score_tasks",
            "payload": {
                "input": {
                    "tasks": [],
                    "today": "2026-03-01",
                }
            },
            "source": "test",
        }
    )

    assert result == {"ok": True}
    assert captured == {
        "logic_ref": "tasks.score#score_tasks",
        "payload": {
            "tasks": [],
            "today": "2026-03-01",
            "source": "test",
        },
    }


@pytest.mark.asyncio
async def test_invoke_logic_uses_env_fallback_in_deterministic_mode(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    created = await helpers.create_task_record(
        workspace,
        {
            "title": "Fallback score",
            "details": None,
            "effort": 2,
            "tags": ["quick"],
            "due_date": "2026-03-01",
        },
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NLVM_MODULES_ROOT", str(_modules_root()))
        patch.setenv("NLVM_EXECUTOR", "deterministic")
        patch.delenv("NLVM_PROVIDERS_PATH", raising=False)

        result = await helpers.invoke_logic(
            "tasks.score#score_tasks",
            payload={
                "tasks": [created],
                "today": "2026-03-01",
            },
        )

    assert "ranked" in result
    assert result["ranked"][0]["task_id"] == created["id"]


@pytest.mark.asyncio
async def test_fallback_logic_caller_uses_deterministic_runtime_for_daemon_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = str(tmp_path / "workspace")
    created = await helpers.create_task_record(
        workspace,
        {
            "title": "Daemon fallback",
            "details": None,
            "effort": 3,
            "tags": [],
            "due_date": None,
        },
    )

    monkeypatch.setenv("NLVM_MODULES_ROOT", str(_modules_root()))
    monkeypatch.setenv("NLVM_EXECUTOR", "llm")
    monkeypatch.delenv("NLVM_PROVIDERS_PATH", raising=False)
    monkeypatch.setattr("multiprocessing.current_process", lambda: SimpleNamespace(daemon=True))

    caller = helpers._build_fallback_logic_caller()
    result = await caller(
        "tasks.score#score_tasks",
        {
            "tasks": [created],
            "today": "2026-03-01",
        },
    )

    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    assert payload["ranked"][0]["task_id"] == created["id"]


def test_fallback_logic_caller_requires_providers_in_non_daemon_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NLVM_MODULES_ROOT", str(_modules_root()))
    monkeypatch.setenv("NLVM_EXECUTOR", "llm")
    monkeypatch.delenv("NLVM_PROVIDERS_PATH", raising=False)
    monkeypatch.setattr("multiprocessing.current_process", lambda: SimpleNamespace(daemon=False))

    with pytest.raises(RuntimeError, match="Logic caller is not configured"):
        helpers._build_fallback_logic_caller()
