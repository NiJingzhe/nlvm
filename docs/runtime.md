# Runtime Guide

`nlvm` executes NL logic fragments as typed Python async callables.

## Core Concepts

- `.nls` modules declare `module`, typed `import type`, helper `import func`, and `logic` blocks.
- Python calls logic by reference: `runtime.use("module#logic")`.
- Inputs and outputs are validated with Pydantic-backed type adapters.
- Each logic can declare its own model engine using `llm <provider> <model>`.

## Install

```bash
poetry install
```

Project requirement is Python `>=3.12,<4.0`.

## Runtime Modes

### 1) Custom Invoker (deterministic or your own executor)

```python
from pathlib import Path

from nlvm import NLRuntime


async def my_invoker(request, toolkit):
    del toolkit
    return request.logic_args["input"]


runtime = NLRuntime(
    modules_root=Path("./nl_modules"),
    invoker=my_invoker,
    include_pyrepl_tools=False,
)

logic_fn = runtime.use("tasks.list#list_tasks")
result = await logic_fn({"workspace": "./workspace", "include_done": False})
```

### 2) Provider Config + Built-in LLM Invoker

```python
from pathlib import Path

from nlvm import build_runtime_from_provider_config

runtime = build_runtime_from_provider_config(
    modules_root=Path("./nl_modules"),
    providers_path=Path("./providers.json"),
    include_pyrepl_tools=True,
)

plan = runtime.use("tasks.plan#build_day_plan")
result = await plan({"workspace": "./workspace", "today": "2026-03-01", "max_items": 3})
```

Optional fallback model for logics without a logic-level `llm` clause:

```python
runtime = build_runtime_from_provider_config(
    modules_root=Path("./nl_modules"),
    providers_path=Path("./providers.json"),
    provider_id="openai-compatible",
    model_name="gpt-4o-mini",
    include_pyrepl_tools=True,
)
```

## REPL Result Channel

In LLM + CodeAct mode, REPL includes a built-in helper:

```python
result_done(payload)
```

Use it when Python code has produced the final structured output.

REPL is persistent across tool calls (notebook-style session). You can inspect helper results,
keep intermediate variables, and then continue computation in later `execute_code` steps.

Recommended agent pattern for better pass@1:

- First grounding step: print concise type/keys/sample when helper return shape is uncertain.
- Then adapt parsing/logic based on observed data.
- Finish by calling `result_done(payload)` with schema-aligned data.

- `payload` should match the logic return schema.
- Runtime reads this channel directly from REPL and validates it.
- This avoids relying on a second LLM re-serialization step for final output.

## DSL Example

```text
syntax nls/1

module tasks.list

import type myapp.types.ListTasksInput as ListTasksInput
import type myapp.types.ListTasksOutput as ListTasksOutput
import func myapp.helpers.list_task_records as list_task_records

logic list_tasks(input: ListTasksInput) -> ListTasksOutput llm openai-compatible gpt-4o-mini {
List tasks by calling @list_task_records and return a typed payload.
}
```
