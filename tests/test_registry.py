from pathlib import Path

import pytest

from nlvm.registry import NLModuleRegistry, NLModuleRegistryError


def _write_nls(path: Path, module_name: str, logic_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "syntax nls/1",
                "",
                f"module {module_name}",
                "",
                f"logic {logic_name}() -> int {{",
                "Return zero.",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_registry_loads_modules_and_supports_lookup(tmp_path: Path) -> None:
    _write_nls(tmp_path / "users" / "create_user.nls", "users.create", "create_user")
    _write_nls(tmp_path / "users" / "read_user.nls", "users.read", "read_user")

    registry = NLModuleRegistry(tmp_path)
    modules = registry.load()

    assert len(modules) == 2
    assert registry.get("users.create") is not None
    assert registry.get("users.read") is not None


def test_registry_rejects_duplicate_module_names(tmp_path: Path) -> None:
    _write_nls(tmp_path / "a" / "create_user.nls", "users.create", "logic_a")
    _write_nls(tmp_path / "b" / "create_user.nls", "users.create", "logic_b")

    registry = NLModuleRegistry(tmp_path)

    with pytest.raises(NLModuleRegistryError, match="Duplicate module"):
        registry.load()
