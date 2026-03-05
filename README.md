# nlvm

`nlvm` is a natural-language runtime with typed Python bindings.

- Declare logic in `.nls` modules and call it from Python with `runtime.use("module#logic")`.
- Keep strict Pydantic input/output contracts even when execution is LLM-driven.
- Mix orchestration flows like `Python -> NL`, `NL -> Python`, and nested `NL -> Python -> NL`.

## Requirements

- Python `>=3.12,<4.0`
- Install dependencies with Poetry:

```bash
poetry install
```

## Documentation

- Runtime usage and engine configuration: `docs/runtime.md`
- Single example (runtime-intelligent TODO tool): `docs/todo-cli.md`

## Example

The repository now keeps one runnable example under `example/`: a runtime-intelligent TODO CLI.

Quick run:

```bash
poetry run python -m example.cli_tool.main \
  "明天上午完成周报，下午整理发布说明并排查 CI 超时" \
  --providers ./providers.json
```
