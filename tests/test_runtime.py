from pathlib import Path
import json
from typing import Any

import pytest

import nlvm.runtime as runtime_module
from nlvm.models import NLLMConfig, NLLogic
from nlvm.runtime import LogicExecutionRequest, NLRuntime, NLRuntimeError
from tests.fixtures.sample_types import CreateOutput


class _FakeExecuteCodeTool:
    def __init__(self) -> None:
        self.name = "execute_code"


class _FakeRepl:
    def __init__(self, *, submitted: bool, payload: Any) -> None:
        self.submitted = submitted
        self.payload = payload
        self.closed = False
        self.executed: list[str] = []
        self.toolset = [_FakeExecuteCodeTool()]

    async def execute(self, code: str) -> dict[str, Any]:
        self.executed.append(code)
        if "__nlvm_result_channel_probe__" in code:
            channel_payload = {
                "submitted": self.submitted,
                "payload": self.payload,
            }
            return {
                "success": True,
                "stdout": json.dumps(channel_payload) + "\n",
            }
        return {
            "success": True,
            "stdout": "",
        }

    def close(self) -> None:
        self.closed = True


def _write_module(tmp_path: Path, relative_path: str, content: str) -> Path:
    module_path = tmp_path / relative_path
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(content.strip() + "\n", encoding="utf-8")
    return module_path


@pytest.mark.asyncio
async def test_runtime_use_returns_typed_callable_for_logic_ref(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "users/create_user.nls",
        """
        syntax nls/1

        module users.create

        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput
        import func tests.fixtures.sample_helpers.upsert_user as upsert_user

        logic create_user(input: CreateInput) -> CreateOutput {
        Create user from input and use @upsert_user in main flow.
        }
        """,
    )

    captured: dict[str, Any] = {}

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> dict[str, Any]:
        captured["request"] = request
        captured["toolkit"] = toolkit
        return {
            "name": request.logic_args["input"]["name"],
            "email": request.logic_args["input"].get("email"),
        }

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        invoker=fake_invoker,
        include_pyrepl_tools=False,
    )
    create_user = runtime.use("users.create#create_user")

    result = await create_user({"name": "Alice"})

    assert isinstance(result, CreateOutput)
    assert result.name == "Alice"
    assert captured["request"].logic.name == "create_user"
    assert captured["request"].helper_aliases == ["upsert_user"]
    assert callable(captured["request"].helper_functions["upsert_user"])
    assert captured["toolkit"] == []


def test_runtime_requires_simplellmfunc_when_pyrepl_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "PyRepl", None)

    with pytest.raises(NLRuntimeError, match="include_pyrepl_tools=True requires"):
        NLRuntime(
            llm_interface=object(),
            modules_root=tmp_path,
            include_pyrepl_tools=True,
        )


@pytest.mark.asyncio
async def test_runtime_use_rejects_invalid_logic_ref_format(tmp_path: Path) -> None:
    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        include_pyrepl_tools=False,
    )

    with pytest.raises(NLRuntimeError, match="Invalid logic reference"):
        runtime.use("users.create")


@pytest.mark.asyncio
async def test_runtime_callable_rejects_invalid_output_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "users/create_user.nls",
        """
        syntax nls/1

        module users.create

        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput

        logic create_user(input: CreateInput) -> CreateOutput {
        Create user from input.
        }
        """,
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> dict[str, Any]:
        del request, toolkit
        return {"id": "u_1"}

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        invoker=fake_invoker,
        include_pyrepl_tools=False,
    )
    create_user = runtime.use("users.create#create_user")

    with pytest.raises(NLRuntimeError, match="output validation failed"):
        await create_user({"name": "Alice"})


@pytest.mark.asyncio
async def test_runtime_callable_supports_multi_param_positional_and_keyword_calls(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "math/sum.nls",
        """
        syntax nls/1

        module math.sum

        logic add(a: int, b: int) -> int {
        Return the sum of a and b.
        }
        """,
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> int:
        del toolkit
        return int(request.logic_args["a"]) + int(request.logic_args["b"])

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        invoker=fake_invoker,
        include_pyrepl_tools=False,
    )
    add = runtime.use("math.sum#add")

    positional_result = await add(1, 2)
    keyword_result = await add(a=3, b=4)

    assert positional_result == 3
    assert keyword_result == 7


@pytest.mark.asyncio
async def test_runtime_callable_rejects_unknown_parameter_name(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "math/sum.nls",
        """
        syntax nls/1

        module math.sum

        logic add(a: int, b: int) -> int {
        Return the sum of a and b.
        }
        """,
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> int:
        del request, toolkit
        return 0

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        invoker=fake_invoker,
        include_pyrepl_tools=False,
    )
    add = runtime.use("math.sum#add")

    with pytest.raises(NLRuntimeError, match="Unknown logic argument"):
        await add(a=1, c=2)


@pytest.mark.asyncio
async def test_runtime_reports_missing_default_invoker_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "llm_function", None)

    _write_module(
        tmp_path,
        "users/create_user.nls",
        """
        syntax nls/1

        module users.create

        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput

        logic create_user(input: CreateInput) -> CreateOutput {
        Create user from input.
        }
        """,
    )

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        include_pyrepl_tools=False,
    )
    create_user = runtime.use("users.create#create_user")

    with pytest.raises(NLRuntimeError, match="Default invoker requires SimpleLLMFunc"):
        await create_user({"name": "Alice"})


def test_runtime_resolves_logic_level_llm_with_runtime_resolver(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_logic_llm_resolver(provider_id: str, model_name: str) -> str:
        captured["provider_id"] = provider_id
        captured["model_name"] = model_name
        return f"{provider_id}:{model_name}"

    runtime = NLRuntime(
        llm_interface="default-llm",
        modules_root=tmp_path,
        logic_llm_resolver=fake_logic_llm_resolver,
        include_pyrepl_tools=False,
    )
    logic = NLLogic(
        name="create_user",
        params=[],
        return_type_ref="int",
        body="",
        mentions=[],
        line=1,
        llm=NLLMConfig(provider_id="demo-provider", model_name="demo-model"),
    )

    selected = runtime._resolve_llm_interface_for_logic(logic)

    assert selected == "demo-provider:demo-model"
    assert captured == {
        "provider_id": "demo-provider",
        "model_name": "demo-model",
    }


def test_runtime_rejects_logic_level_llm_without_resolver(tmp_path: Path) -> None:
    runtime = NLRuntime(
        llm_interface="default-llm",
        modules_root=tmp_path,
        include_pyrepl_tools=False,
    )
    logic = NLLogic(
        name="create_user",
        params=[],
        return_type_ref="int",
        body="",
        mentions=[],
        line=1,
        llm=NLLMConfig(provider_id="demo-provider", model_name="demo-model"),
    )

    with pytest.raises(NLRuntimeError, match="declares llm"):
        runtime._resolve_llm_interface_for_logic(logic)


@pytest.mark.asyncio
async def test_runtime_prefers_repl_result_channel_payload_when_submitted(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "users/create_user.nls",
        """
        syntax nls/1

        module users.create

        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput

        logic create_user(input: CreateInput) -> CreateOutput {
        Create user from input.
        }
        """,
    )

    fake_repl = _FakeRepl(
        submitted=True,
        payload={"name": "FromRepl", "email": "repl@example.com"},
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> str:
        del request
        assert toolkit == fake_repl.toolset
        return "invoker did not return json"

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        invoker=fake_invoker,
        include_pyrepl_tools=True,
        repl_factory=lambda: fake_repl,
    )

    create_user = runtime.use("users.create#create_user")
    result = await create_user({"name": "Ignored"})

    assert isinstance(result, CreateOutput)
    assert result.name == "FromRepl"
    assert result.email == "repl@example.com"
    assert fake_repl.closed is True
    assert any("__nlvm_result_channel_probe__" in item for item in fake_repl.executed)


@pytest.mark.asyncio
async def test_runtime_falls_back_to_invoker_output_when_repl_result_not_submitted(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "users/create_user.nls",
        """
        syntax nls/1

        module users.create

        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput

        logic create_user(input: CreateInput) -> CreateOutput {
        Create user from input.
        }
        """,
    )

    fake_repl = _FakeRepl(
        submitted=False,
        payload={"name": "IgnoredRepl", "email": None},
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> dict[str, Any]:
        del request
        assert toolkit == fake_repl.toolset
        return {"name": "FromInvoker", "email": None}

    runtime = NLRuntime(
        llm_interface=object(),
        modules_root=tmp_path,
        invoker=fake_invoker,
        include_pyrepl_tools=True,
        repl_factory=lambda: fake_repl,
    )

    create_user = runtime.use("users.create#create_user")
    result = await create_user({"name": "Ignored"})

    assert isinstance(result, CreateOutput)
    assert result.name == "FromInvoker"
    assert fake_repl.closed is True
