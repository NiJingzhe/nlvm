"""NLVM runtime and parsing APIs."""

from nlvm.engine import (
    EngineConfigError,
    build_logic_llm_resolver_from_provider_config,
    build_runtime_from_provider_config,
    load_llm_interface_from_provider_config,
)
from nlvm.models import NLLMConfig, NLImport, NLLogic, NLLogicParam, NLModule
from nlvm.parser import NLModuleParseError, load_nl_module, parse_nl_module_text
from nlvm.registry import NLModuleRegistry, NLModuleRegistryError
from nlvm.resolver import (
    NLImportResolver,
    NLResolutionError,
    ResolvedLogic,
    ResolvedModule,
)
from nlvm.runtime import LogicExecutionRequest, LogicInvoker, NLRuntime, NLRuntimeError

__all__ = [
    "EngineConfigError",
    "LogicExecutionRequest",
    "LogicInvoker",
    "NLLMConfig",
    "NLImport",
    "NLImportResolver",
    "NLLogic",
    "NLLogicParam",
    "NLModule",
    "NLModuleParseError",
    "NLModuleRegistry",
    "NLModuleRegistryError",
    "NLResolutionError",
    "NLRuntime",
    "NLRuntimeError",
    "ResolvedLogic",
    "ResolvedModule",
    "build_logic_llm_resolver_from_provider_config",
    "build_runtime_from_provider_config",
    "load_llm_interface_from_provider_config",
    "load_nl_module",
    "parse_nl_module_text",
]
