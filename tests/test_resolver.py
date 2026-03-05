from pathlib import Path

import pytest

from nlvm.parser import parse_nl_module_text
from nlvm.resolver import NLImportResolver, NLResolutionError


def test_resolver_resolves_module_alias_type_alias_and_helpers() -> None:
    module = parse_nl_module_text(
        """
        syntax nls/1

        module users.read

        import module tests.fixtures.sample_types as sample_types
        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import func tests.fixtures.sample_helpers.upsert_user as upsert_user

        logic read_user(input: sample_types.CreateInput) -> CreateInput {
        Read user and call @upsert_user.
        }
        """,
        source=Path("nlvm/users/read_user.nls"),
    )

    resolved = NLImportResolver().resolve(module)
    logic = resolved.logics["read_user"]

    assert logic.param_types["input"].__name__ == "CreateInput"
    assert logic.return_type.__name__ == "CreateInput"
    assert "upsert_user" in logic.helpers
    assert callable(logic.helpers["upsert_user"])
    assert logic.helper_paths["upsert_user"] == "tests.fixtures.sample_helpers.upsert_user"


def test_resolver_rejects_non_callable_func_import() -> None:
    module = parse_nl_module_text(
        """
        syntax nls/1
        module users.read
        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import func tests.fixtures.sample_types as sample_module

        logic read_user(input: CreateInput) -> CreateInput {
        Call @sample_module.
        }
        """,
        source=Path("nlvm/users/read_user.nls"),
    )

    with pytest.raises(NLResolutionError, match="callable"):
        NLImportResolver().resolve(module)
