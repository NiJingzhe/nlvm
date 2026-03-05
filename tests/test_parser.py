from pathlib import Path

import pytest

from nlvm.parser import NLModuleParseError, load_nl_module


def _write_module(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "users" / "create_user.nls"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_parser_loads_logic_module_with_structured_header(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        """
        syntax nls/1

        module users.create

        import module tests.fixtures.sample_types as sample_types
        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput
        import func tests.fixtures.sample_helpers.upsert_user as upsert_user

        logic create_user(input: CreateInput) -> CreateOutput llm openai-compatible gpt-4o-mini {
        Create user and call @upsert_user.
        }
        """,
    )

    module = load_nl_module(module_path)
    logic = module.logics[0]

    assert module.syntax_version == "nls/1"
    assert module.module_name == "users.create"
    assert module.source_path == module_path
    assert [item.kind for item in module.imports] == ["module", "type", "type", "func"]
    assert logic.name == "create_user"
    assert logic.params[0].name == "input"
    assert logic.params[0].type_ref == "CreateInput"
    assert logic.return_type_ref == "CreateOutput"
    assert logic.llm is not None
    assert logic.llm.provider_id == "openai-compatible"
    assert logic.llm.model_name == "gpt-4o-mini"
    assert logic.mentions == ["upsert_user"]


def test_parser_rejects_unknown_helper_mentions(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        """
        syntax nls/1
        module users.create
        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput
        import func tests.fixtures.sample_helpers.upsert_user as upsert_user

        logic create_user(input: CreateInput) -> CreateOutput {
        Main flow uses @upsert_user.
        On conflict, call @unknown_helper.
        }
        """,
    )

    with pytest.raises(NLModuleParseError, match="Unknown helper mention"):
        load_nl_module(module_path)


def test_parser_accepts_logic_without_llm_clause(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        """
        syntax nls/1

        module users.create

        import type tests.fixtures.sample_types.CreateInput as CreateInput
        import type tests.fixtures.sample_types.CreateOutput as CreateOutput

        logic create_user(input: CreateInput) -> CreateOutput {
        Create user.
        }
        """,
    )

    module = load_nl_module(module_path)
    assert module.logics[0].llm is None


def test_parser_requires_at_least_one_logic_block(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        """
        syntax nls/1
        module users.create
        import type tests.fixtures.sample_types.CreateInput as CreateInput
        """,
    )

    with pytest.raises(NLModuleParseError, match="At least one logic block"):
        load_nl_module(module_path)
