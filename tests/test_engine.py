from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import nlvm.engine as engine_module
from nlvm.engine import (
    EngineConfigError,
    build_logic_llm_resolver_from_provider_config,
    build_runtime_from_provider_config,
    load_llm_interface_from_provider_config,
)
from nlvm.runtime import LogicExecutionRequest


def _write_providers_file(tmp_path: Path) -> Path:
    providers_path = tmp_path / "providers.json"
    providers_path.write_text("{}\n", encoding="utf-8")
    return providers_path


def _install_fake_simplellmfunc(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_table: dict[str, dict[str, Any]],
) -> type:
    class FakeOpenAICompatible:
        last_path: str | None = None

        @classmethod
        def load_from_json_file(cls, path: str) -> dict[str, dict[str, Any]]:
            cls.last_path = path
            return model_table

    fake_module = ModuleType("SimpleLLMFunc")
    setattr(fake_module, "OpenAICompatible", FakeOpenAICompatible)

    original_import_module = importlib.import_module

    def _fake_import_module(name: str) -> Any:
        if name == "SimpleLLMFunc":
            return fake_module
        return original_import_module(name)

    monkeypatch.setattr(engine_module.importlib, "import_module", _fake_import_module)
    return FakeOpenAICompatible


def test_load_provider_config_requires_existing_file(tmp_path: Path) -> None:
    providers_path = tmp_path / "missing.json"

    with pytest.raises(EngineConfigError, match="not found"):
        load_llm_interface_from_provider_config(
            providers_path,
            provider_id="demo",
            model_name="mini",
        )


def test_load_provider_config_requires_simplellmfunc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_path = _write_providers_file(tmp_path)

    def _raise_import(name: str) -> Any:
        if name == "SimpleLLMFunc":
            raise ModuleNotFoundError("SimpleLLMFunc")
        return importlib.import_module(name)

    monkeypatch.setattr(engine_module.importlib, "import_module", _raise_import)

    with pytest.raises(EngineConfigError, match="SimpleLLMFunc is required"):
        load_llm_interface_from_provider_config(
            providers_path,
            provider_id="demo",
            model_name="mini",
        )


def test_load_provider_config_validates_provider_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_path = _write_providers_file(tmp_path)
    _install_fake_simplellmfunc(
        monkeypatch,
        model_table={"demo": {"mini": object()}},
    )

    with pytest.raises(EngineConfigError, match="Missing provider/model"):
        load_llm_interface_from_provider_config(
            providers_path,
            provider_id="demo",
            model_name="missing",
        )


def test_build_logic_llm_resolver_caches_loaded_interfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_path = _write_providers_file(tmp_path)
    fake_interface = object()
    fake_openai_compatible = _install_fake_simplellmfunc(
        monkeypatch,
        model_table={"demo": {"mini": fake_interface}},
    )

    resolver = build_logic_llm_resolver_from_provider_config(providers_path)
    first = resolver("demo", "mini")
    second = resolver("demo", "mini")

    assert first is fake_interface
    assert second is fake_interface
    assert fake_openai_compatible.last_path == str(providers_path)


@pytest.mark.asyncio
async def test_build_runtime_from_provider_config_loads_llm_and_runs_logic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_path = _write_providers_file(tmp_path)
    fake_interface = object()
    fake_openai_compatible = _install_fake_simplellmfunc(
        monkeypatch,
        model_table={"demo": {"mini": fake_interface}},
    )

    module_path = tmp_path / "math" / "sum.nls"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "syntax nls/1",
                "",
                "module math.sum",
                "",
                "logic add(a: int, b: int) -> int {",
                "Return the sum of a and b.",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> int:
        del toolkit
        return int(request.logic_args["a"]) + int(request.logic_args["b"])

    runtime = build_runtime_from_provider_config(
        modules_root=tmp_path,
        providers_path=providers_path,
        provider_id="demo",
        model_name="mini",
        invoker=fake_invoker,
        include_pyrepl_tools=False,
    )
    add = runtime.use("math.sum#add")

    result = await add(2, 3)

    assert result == 5
    assert runtime._llm_interface is fake_interface
    assert fake_openai_compatible.last_path == str(providers_path)


@pytest.mark.asyncio
async def test_build_runtime_from_provider_config_supports_logic_level_llm_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_path = _write_providers_file(tmp_path)
    _install_fake_simplellmfunc(
        monkeypatch,
        model_table={"demo": {"mini": object()}},
    )

    module_path = tmp_path / "math" / "sum.nls"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "syntax nls/1",
                "",
                "module math.sum",
                "",
                "logic add(a: int, b: int) -> int llm demo mini {",
                "Return the sum of a and b.",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    async def fake_invoker(request: LogicExecutionRequest, toolkit: list[Any]) -> int:
        del toolkit
        assert request.logic.llm is not None
        assert request.logic.llm.provider_id == "demo"
        assert request.logic.llm.model_name == "mini"
        return int(request.logic_args["a"]) + int(request.logic_args["b"])

    runtime = build_runtime_from_provider_config(
        modules_root=tmp_path,
        providers_path=providers_path,
        invoker=fake_invoker,
        include_pyrepl_tools=False,
    )
    add = runtime.use("math.sum#add")

    result = await add(2, 3)

    assert result == 5
    assert runtime._llm_interface is None


def test_build_runtime_from_provider_config_requires_default_provider_pair(
    tmp_path: Path,
) -> None:
    providers_path = _write_providers_file(tmp_path)

    with pytest.raises(EngineConfigError, match="both provider_id and model_name"):
        build_runtime_from_provider_config(
            modules_root=tmp_path,
            providers_path=providers_path,
            provider_id="demo",
        )
