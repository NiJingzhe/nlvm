"""Reference resolution for NLVM imports and logic type signatures."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any, Callable

from nlvm.models import NLLogic, NLModule


_BUILTIN_TYPE_MAP: dict[str, Any] = {
    "Any": Any,
    "bool": bool,
    "bytes": bytes,
    "dict": dict,
    "float": float,
    "int": int,
    "list": list,
    "object": object,
    "str": str,
}


class NLResolutionError(ValueError):
    """Raised when an NLVM import or type reference cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ResolvedLogic:
    """Resolved runtime contract for one logic fragment."""

    logic: NLLogic
    param_types: dict[str, Any]
    return_type: Any
    helpers: dict[str, Callable[..., Any]]
    helper_paths: dict[str, str]


@dataclass(frozen=True, slots=True)
class ResolvedModule:
    """Resolved runtime artifacts for a parsed NLVM module."""

    module: NLModule
    logics: dict[str, ResolvedLogic]


def _resolve_dotted_symbol(path: str) -> Any:
    """Resolve one dotted Python symbol path into the final object."""

    if not path or path.strip() != path:
        raise NLResolutionError(f"Invalid dotted path '{path}'")

    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        pass

    parts = path.split(".")
    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        attribute_path = parts[split_index:]
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        for attribute in attribute_path:
            if not hasattr(obj, attribute):
                raise NLResolutionError(
                    f"Cannot resolve dotted path '{path}'; missing attribute '{attribute}'"
                )
            obj = getattr(obj, attribute)
        return obj

    raise NLResolutionError(f"Cannot resolve dotted path '{path}'")


def _resolve_type_ref(
    type_ref: str,
    *,
    type_aliases: dict[str, Any],
    module_aliases: dict[str, ModuleType],
) -> Any:
    """Resolve one type reference from aliases, modules, builtins, or direct paths."""

    if type_ref in _BUILTIN_TYPE_MAP:
        return _BUILTIN_TYPE_MAP[type_ref]

    if type_ref in type_aliases:
        return type_aliases[type_ref]

    if "." in type_ref:
        first, remainder = type_ref.split(".", 1)
        if first in module_aliases:
            value: Any = module_aliases[first]
            for attribute in remainder.split("."):
                if not hasattr(value, attribute):
                    raise NLResolutionError(
                        f"Cannot resolve type reference '{type_ref}'; missing attribute '{attribute}'"
                    )
                value = getattr(value, attribute)
            return value

        return _resolve_dotted_symbol(type_ref)

    raise NLResolutionError(f"Unknown type reference '{type_ref}'")


class NLImportResolver:
    """Resolve NLVM module imports into Python objects."""

    def resolve(self, module: NLModule) -> ResolvedModule:
        """Resolve one parsed module into executable contracts."""

        module_aliases: dict[str, ModuleType] = {}
        type_aliases: dict[str, Any] = {}
        helper_aliases: dict[str, Callable[..., Any]] = {}
        helper_paths: dict[str, str] = {}

        for nl_import in module.imports:
            if nl_import.kind == "module":
                try:
                    imported_module = importlib.import_module(nl_import.target)
                except ModuleNotFoundError as exc:
                    raise NLResolutionError(
                        f"Cannot resolve module import '{nl_import.target}'"
                    ) from exc
                module_aliases[nl_import.alias] = imported_module
                continue

            resolved = _resolve_dotted_symbol(nl_import.target)
            if nl_import.kind == "type":
                type_aliases[nl_import.alias] = resolved
                continue

            if nl_import.kind == "func":
                if not callable(resolved):
                    raise NLResolutionError(
                        f"Function import '{nl_import.target}' must resolve to callable"
                    )
                helper_aliases[nl_import.alias] = resolved
                helper_paths[nl_import.alias] = nl_import.target
                continue

            raise NLResolutionError(f"Unsupported import kind '{nl_import.kind}'")

        resolved_logics: dict[str, ResolvedLogic] = {}
        for logic in module.logics:
            param_types: dict[str, Any] = {}
            for param in logic.params:
                param_types[param.name] = _resolve_type_ref(
                    param.type_ref,
                    type_aliases=type_aliases,
                    module_aliases=module_aliases,
                )

            return_type = _resolve_type_ref(
                logic.return_type_ref,
                type_aliases=type_aliases,
                module_aliases=module_aliases,
            )

            resolved_logics[logic.name] = ResolvedLogic(
                logic=logic,
                param_types=param_types,
                return_type=return_type,
                helpers=dict(helper_aliases),
                helper_paths=dict(helper_paths),
            )

        return ResolvedModule(module=module, logics=resolved_logics)


__all__ = [
    "NLImportResolver",
    "NLResolutionError",
    "ResolvedLogic",
    "ResolvedModule",
]
