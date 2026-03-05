"""Execution engine configuration helpers for NLRuntime."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import logging
from pathlib import Path
from typing import Any

from nlvm.resolver import NLImportResolver
from nlvm.runtime import LogicInvoker, NLRuntime


class EngineConfigError(RuntimeError):
    """Raised when an execution engine cannot be configured."""


def load_llm_interface_from_provider_config(
    providers_path: Path,
    *,
    provider_id: str,
    model_name: str,
) -> Any:
    """Load one SimpleLLMFunc LLM interface from providers JSON config."""

    if not providers_path.exists():
        raise EngineConfigError(f"Provider config file not found: {providers_path}")
    if not providers_path.is_file():
        raise EngineConfigError(f"Provider config path is not a file: {providers_path}")

    try:
        simplellmfunc_module = importlib.import_module("SimpleLLMFunc")
    except Exception as exc:
        raise EngineConfigError(
            "SimpleLLMFunc is required to load provider config; install SimpleLLMFunc first"
        ) from exc

    openai_compatible = getattr(simplellmfunc_module, "OpenAICompatible", None)
    if openai_compatible is None:
        raise EngineConfigError("SimpleLLMFunc.OpenAICompatible is not available")

    all_models = openai_compatible.load_from_json_file(str(providers_path))
    try:
        return all_models[provider_id][model_name]
    except KeyError as exc:
        raise EngineConfigError(
            f"Missing provider/model in {providers_path}: {provider_id}/{model_name}"
        ) from exc


def build_logic_llm_resolver_from_provider_config(
    providers_path: Path,
) -> Callable[[str, str], Any]:
    """Build cached resolver for provider/model -> llm_interface."""

    cache: dict[tuple[str, str], Any] = {}

    def _resolve(provider_id: str, model_name: str) -> Any:
        key = (provider_id, model_name)
        cached = cache.get(key)
        if cached is not None:
            return cached

        llm_interface = load_llm_interface_from_provider_config(
            providers_path,
            provider_id=provider_id,
            model_name=model_name,
        )
        cache[key] = llm_interface
        return llm_interface

    return _resolve


def build_runtime_from_provider_config(
    *,
    modules_root: Path,
    providers_path: Path,
    provider_id: str | None = None,
    model_name: str | None = None,
    resolver: NLImportResolver | None = None,
    invoker: LogicInvoker | None = None,
    include_pyrepl_tools: bool = True,
    max_tool_calls: int = 8,
    repl_factory: Callable[[], Any] | None = None,
    logger: logging.Logger | None = None,
) -> NLRuntime:
    """Build NLRuntime with llm_interface loaded from providers config."""

    if (provider_id is None) != (model_name is None):
        raise EngineConfigError(
            "When setting default engine, both provider_id and model_name must be provided"
        )

    llm_interface = None
    if provider_id is not None and model_name is not None:
        llm_interface = load_llm_interface_from_provider_config(
            providers_path,
            provider_id=provider_id,
            model_name=model_name,
        )

    logic_llm_resolver = build_logic_llm_resolver_from_provider_config(providers_path)
    return NLRuntime(
        llm_interface=llm_interface,
        modules_root=modules_root,
        resolver=resolver,
        invoker=invoker,
        logic_llm_resolver=logic_llm_resolver,
        include_pyrepl_tools=include_pyrepl_tools,
        max_tool_calls=max_tool_calls,
        repl_factory=repl_factory,
        logger=logger,
    )


__all__ = [
    "EngineConfigError",
    "build_logic_llm_resolver_from_provider_config",
    "build_runtime_from_provider_config",
    "load_llm_interface_from_provider_config",
]
