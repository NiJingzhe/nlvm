"""Filesystem registry for NLVM modules."""

from __future__ import annotations

from pathlib import Path

from nlvm.models import NLModule
from nlvm.parser import NLModuleParseError, load_nl_module


class NLModuleRegistryError(ValueError):
    """Raised when NLVM module discovery fails."""


class NLModuleRegistry:
    """Load and index NLVM modules from a root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._modules: dict[str, NLModule] = {}

    def load(self) -> list[NLModule]:
        """Load all `.nls` files from root and index by module name."""

        if not self._root.exists():
            self._modules = {}
            return []

        if not self._root.is_dir():
            raise NLModuleRegistryError(f"NLVM root is not a directory: {self._root}")

        loaded: dict[str, NLModule] = {}
        for file_path in sorted(self._root.rglob("*.nls"), key=lambda item: str(item)):
            try:
                module = load_nl_module(file_path)
            except NLModuleParseError as exc:
                raise NLModuleRegistryError(str(exc)) from exc

            duplicate = loaded.get(module.module_name)
            if duplicate is not None:
                raise NLModuleRegistryError(
                    f"Duplicate module '{module.module_name}' in {duplicate.source_path} and {module.source_path}"
                )

            loaded[module.module_name] = module

        self._modules = loaded
        return list(loaded.values())

    def get(self, module_name: str) -> NLModule | None:
        """Get one loaded module by module name."""

        return self._modules.get(module_name)


__all__ = [
    "NLModuleRegistry",
    "NLModuleRegistryError",
]
