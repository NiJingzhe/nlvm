"""Parser for the strong-format NLVM DSL."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from nlvm.models import NLLMConfig, NLImport, NLLogic, NLLogicParam, NLModule


_SYNTAX_PATTERN = re.compile(r"^syntax\s+(nls/\d+)\s*$")
_MODULE_PATTERN = re.compile(r"^module\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*$")
_IMPORT_PATTERN = re.compile(
    r"^import\s+(module|type|func)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
_LOGIC_HEADER_PATTERN = re.compile(
    r"^logic\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*->\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\s+llm\s+([A-Za-z0-9_.:-]+)\s+([A-Za-z0-9_.:/-]+))?\s*\{\s*$"
)
_PARAM_PATTERN = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)$"
)
_MENTION_PATTERN = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")


class NLModuleParseError(ValueError):
    """Raised when NLVM source cannot be parsed or validated."""


def _parse_error(message: str, *, source: Path, line: int | None = None) -> NLModuleParseError:
    if line is None:
        return NLModuleParseError(f"{message} in {source}")
    return NLModuleParseError(f"{message} in {source} at line {line}")


def _is_ignorable(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _skip_ignorable(lines: list[str], start: int) -> int:
    index = start
    while index < len(lines) and _is_ignorable(lines[index]):
        index += 1
    return index


def _parse_params(raw_params: str, *, source: Path, line: int) -> list[NLLogicParam]:
    stripped = raw_params.strip()
    if not stripped:
        return []

    params: list[NLLogicParam] = []
    seen_names: set[str] = set()
    for chunk in stripped.split(","):
        part = chunk.strip()
        match = _PARAM_PATTERN.fullmatch(part)
        if match is None:
            raise _parse_error(
                f"Invalid logic parameter declaration '{part}'", source=source, line=line
            )

        param_name, type_ref = match.group(1), match.group(2)
        if param_name in seen_names:
            raise _parse_error(
                f"Duplicate parameter name '{param_name}'",
                source=source,
                line=line,
            )

        seen_names.add(param_name)
        params.append(NLLogicParam(name=param_name, type_ref=type_ref))

    return params


def _collect_unknown_mentions(mentions: Iterable[str], known_helpers: set[str]) -> list[str]:
    unknown = sorted({name for name in mentions if name not in known_helpers})
    return unknown


def parse_nl_module_text(content: str, *, source: Path) -> NLModule:
    """Parse NLVM module source text into structured objects."""

    lines = content.splitlines()
    if not lines:
        raise _parse_error("Empty NLVM source", source=source)

    index = _skip_ignorable(lines, 0)
    if index >= len(lines):
        raise _parse_error("Empty NLVM source", source=source)

    syntax_match = _SYNTAX_PATTERN.fullmatch(lines[index].strip())
    if syntax_match is None:
        raise _parse_error("Missing or invalid syntax declaration", source=source, line=index + 1)
    syntax_version = syntax_match.group(1)
    index += 1

    index = _skip_ignorable(lines, index)
    if index >= len(lines):
        raise _parse_error("Missing module declaration", source=source)

    module_match = _MODULE_PATTERN.fullmatch(lines[index].strip())
    if module_match is None:
        raise _parse_error("Missing or invalid module declaration", source=source, line=index + 1)
    module_name = module_match.group(1)
    index += 1

    imports: list[NLImport] = []
    import_aliases: set[str] = set()
    helper_aliases: set[str] = set()

    while True:
        index = _skip_ignorable(lines, index)
        if index >= len(lines):
            break

        current = lines[index].strip()
        if current.startswith("logic "):
            break

        match = _IMPORT_PATTERN.fullmatch(current)
        if match is None:
            raise _parse_error("Invalid import declaration", source=source, line=index + 1)

        kind, target, alias = match.group(1), match.group(2), match.group(3)
        if alias in import_aliases:
            raise _parse_error(
                f"Duplicate import alias '{alias}'",
                source=source,
                line=index + 1,
            )

        import_aliases.add(alias)
        if kind == "func":
            helper_aliases.add(alias)

        imports.append(
            NLImport(
                kind=kind,
                target=target,
                alias=alias,
                line=index + 1,
            )
        )
        index += 1

    logics: list[NLLogic] = []
    logic_names: set[str] = set()
    while True:
        index = _skip_ignorable(lines, index)
        if index >= len(lines):
            break

        header_line = lines[index].strip()
        header_match = _LOGIC_HEADER_PATTERN.fullmatch(header_line)
        if header_match is None:
            raise _parse_error("Invalid logic declaration", source=source, line=index + 1)

        logic_name = header_match.group(1)
        if logic_name in logic_names:
            raise _parse_error(
                f"Duplicate logic name '{logic_name}'",
                source=source,
                line=index + 1,
            )
        logic_names.add(logic_name)

        raw_params = header_match.group(2)
        return_type_ref = header_match.group(3)
        llm_provider_id = header_match.group(4)
        llm_model_name = header_match.group(5)
        logic_line = index + 1
        params = _parse_params(raw_params, source=source, line=logic_line)
        llm_config = None
        if llm_provider_id is not None and llm_model_name is not None:
            llm_config = NLLMConfig(
                provider_id=llm_provider_id,
                model_name=llm_model_name,
            )

        index += 1
        body_lines: list[str] = []
        found_closing = False
        while index < len(lines):
            current = lines[index]
            if current.strip() == "}":
                found_closing = True
                break
            body_lines.append(current)
            index += 1

        if not found_closing:
            raise _parse_error(
                f"Missing closing '}}' for logic '{logic_name}'",
                source=source,
                line=logic_line,
            )

        body = "\n".join(body_lines).strip()
        mentions = _MENTION_PATTERN.findall(body)
        unknown_mentions = _collect_unknown_mentions(mentions, helper_aliases)
        if unknown_mentions:
            missing = ", ".join(unknown_mentions)
            raise _parse_error(
                f"Unknown helper mention(s): {missing}",
                source=source,
                line=logic_line,
            )

        logics.append(
            NLLogic(
                name=logic_name,
                params=params,
                return_type_ref=return_type_ref,
                body=body,
                mentions=mentions,
                line=logic_line,
                llm=llm_config,
            )
        )

        index += 1

    if not logics:
        raise _parse_error("At least one logic block is required", source=source)

    return NLModule(
        syntax_version=syntax_version,
        module_name=module_name,
        imports=imports,
        logics=logics,
        source_path=source,
    )


def load_nl_module(source_path: Path) -> NLModule:
    """Load one NLVM module from disk and parse it."""

    if not source_path.exists():
        raise NLModuleParseError(f"NLVM source does not exist: {source_path}")
    if not source_path.is_file():
        raise NLModuleParseError(f"NLVM source path is not a file: {source_path}")

    content = source_path.read_text(encoding="utf-8")
    return parse_nl_module_text(content, source=source_path)


__all__ = [
    "NLModuleParseError",
    "load_nl_module",
    "parse_nl_module_text",
]
