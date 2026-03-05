"""Core data models for the NLVM DSL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NLImport:
    """One import declaration in an NLVM module."""

    kind: str
    target: str
    alias: str
    line: int


@dataclass(frozen=True, slots=True)
class NLLogicParam:
    """One parameter in a logic signature."""

    name: str
    type_ref: str


@dataclass(frozen=True, slots=True)
class NLLMConfig:
    """Optional LLM selection attached to one logic fragment."""

    provider_id: str
    model_name: str


@dataclass(frozen=True, slots=True)
class NLLogic:
    """One executable logic fragment declared in an NLVM module."""

    name: str
    params: list[NLLogicParam]
    return_type_ref: str
    body: str
    mentions: list[str]
    line: int
    llm: NLLMConfig | None = None


@dataclass(frozen=True, slots=True)
class NLModule:
    """Parsed representation of one NLVM source module."""

    syntax_version: str
    module_name: str
    imports: list[NLImport]
    logics: list[NLLogic]
    source_path: Path

    def get_logic(self, name: str) -> NLLogic | None:
        """Return one logic definition by name."""

        for logic in self.logics:
            if logic.name == name:
                return logic
        return None


__all__ = [
    "NLLMConfig",
    "NLImport",
    "NLLogic",
    "NLLogicParam",
    "NLModule",
]
