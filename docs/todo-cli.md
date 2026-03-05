# Runtime-Intelligent TODO CLI

This repository keeps one runnable example under `example/`: a TODO tool that demonstrates runtime intelligence with strict typed contracts.

## What It Demonstrates

- User only provides natural language intent, and assistant schedules against existing todo records.
- Assistant can run clarification rounds (interactive Q&A) when key details are ambiguous.
- `Python -> NL`: CLI calls one orchestrator logic (`tasks.assistant#smart_schedule`).
- `NL -> Python`: orchestrator calls storage helpers (`list_task_records`, `create_task_record`).
- `NL -> Python -> NL`: orchestrator invokes planning logic via `invoke_logic("tasks.plan#build_day_plan", ...)`.
- Typed input/output validation using Pydantic models for every logic boundary.

## Key Files

- `example/cli_tool/types.py`: input/output models.
- `example/cli_tool/helpers.py`: storage helpers, scoring, and nested `invoke_logic`.
- `example/cli_tool/main.py`: simplified interactive CLI entrypoint.
- `example/nl_modules/tasks/smart_schedule.nls`: NL orchestrator for interactive scheduling.
- `example/nl_modules/tasks/build_plan.nls`: ranking-based daily plan assembly.

## Run (Interactive)

```bash
poetry run python -m example.cli_tool.main --providers ./providers.json
```

Then type your plan in natural language when prompted.

## Run (One Shot)

```bash
poetry run python -m example.cli_tool.main \
  "明天上午完成周报，下午准备发布说明，并把测试问题排查一下" \
  --providers ./providers.json
```

## Optional Arguments

- `--workspace`: todo data directory (default `./.nlvm_cli_workspace`)
- `--today`: planning date in `YYYY-MM-DD`
- `--max-items`: max recommended tasks in final schedule

## Provider Config

Prerequisites:

- A valid provider config file.
- Provider/model entries must match logic-level `llm` declarations in `example/nl_modules/tasks/*.nls`.

Template config:

- `example/providers.example.json`

Run:

```bash
poetry run python -m example.cli_tool.main \
  --providers ./providers.json
```
