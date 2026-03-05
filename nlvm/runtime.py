"""Generic NLVM runtime for calling logic fragments as Python functions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

from pydantic import TypeAdapter, ValidationError

try:  # pragma: no cover - optional dependency path
    from SimpleLLMFunc import llm_function  # type: ignore[import-not-found]
    from SimpleLLMFunc.builtin import PyRepl  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency path
    llm_function = None
    PyRepl = None

from nlvm.models import NLLMConfig, NLLogic
from nlvm.registry import NLModuleRegistry
from nlvm.resolver import NLImportResolver, NLResolutionError, ResolvedLogic


_LOGIC_REF_PATTERN = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)#([A-Za-z_][A-Za-z0-9_]*)$"
)


class NLRuntimeError(RuntimeError):
    """Raised when NLVM execution fails."""


@dataclass(frozen=True, slots=True)
class LogicExecutionRequest:
    """Normalized execution request passed to invokers."""

    module_name: str
    logic: NLLogic
    logic_args: dict[str, Any]
    runtime_context: dict[str, Any]
    helper_aliases: list[str]
    helper_functions: dict[str, Callable[..., Any]]
    helper_mentions: list[str]
    helper_instructions: list[str]
    helper_signatures: dict[str, str]
    output_schema: dict[str, Any]
    output_required_fields: list[str]


class LogicInvoker(Protocol):
    """Protocol for pluggable NLVM invokers."""

    async def __call__(
        self,
        request: LogicExecutionRequest,
        toolkit: list[Any],
    ) -> Any:
        """Execute one logic request and return raw output payload."""

        ...


class NLRuntime:
    """Execute NL logic by reference and expose it as call-like Python async functions."""

    def __init__(
        self,
        *,
        llm_interface: Any | None = None,
        modules_root: Path,
        resolver: NLImportResolver | None = None,
        invoker: LogicInvoker | None = None,
        logic_llm_resolver: Callable[[str, str], Any] | None = None,
        include_pyrepl_tools: bool = False,
        max_tool_calls: int = 8,
        repl_factory: Callable[[], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm_interface = llm_interface
        self._module_registry = NLModuleRegistry(modules_root)
        self._resolver = resolver or NLImportResolver()
        self._invoker = invoker
        self._logic_llm_resolver = logic_llm_resolver
        self._include_pyrepl_tools = include_pyrepl_tools
        self._max_tool_calls = max_tool_calls
        self._repl_factory = repl_factory
        self._logger = logger or logging.getLogger(__name__)

        if self._include_pyrepl_tools and self._repl_factory is None:
            if PyRepl is None:
                raise NLRuntimeError(
                    "include_pyrepl_tools=True requires SimpleLLMFunc to be installed"
                )
            self._repl_factory = PyRepl

    def use(self, logic_ref: str) -> Callable[..., Awaitable[Any]]:
        """Return one async Python callable for `module#logic` reference."""

        module_name, logic_name = self._parse_logic_ref(logic_ref)
        resolved_logic = self._resolve_logic(module_name=module_name, logic_name=logic_name)

        async def _logic_callable(*args: Any, **kwargs: Any) -> Any:
            raw_logic_args = self._bind_call_arguments(resolved_logic, args=args, kwargs=kwargs)
            return await self._execute_logic(
                module_name=module_name,
                logic_name=logic_name,
                resolved_logic=resolved_logic,
                raw_logic_args=raw_logic_args,
            )

        _logic_callable.__name__ = logic_name
        _logic_callable.__qualname__ = logic_name
        _logic_callable.__doc__ = resolved_logic.logic.body
        setattr(_logic_callable, "__nl_logic_ref__", logic_ref)
        return _logic_callable

    @staticmethod
    def _parse_logic_ref(logic_ref: str) -> tuple[str, str]:
        """Parse one module#logic reference."""

        match = _LOGIC_REF_PATTERN.fullmatch(logic_ref.strip())
        if match is None:
            raise NLRuntimeError("Invalid logic reference; expected '<module_name>#<logic_name>'")
        return match.group(1), match.group(2)

    def _resolve_logic(self, *, module_name: str, logic_name: str) -> ResolvedLogic:
        """Load and resolve one logic definition from module registry."""

        self._module_registry.load()
        module = self._module_registry.get(module_name)
        if module is None:
            raise NLRuntimeError(f"Module '{module_name}' not found under NLVM root")

        try:
            resolved_module = self._resolver.resolve(module)
        except NLResolutionError as exc:
            raise NLRuntimeError(f"Failed to resolve NL module '{module_name}': {exc}") from exc

        resolved_logic = resolved_module.logics.get(logic_name)
        if resolved_logic is None:
            raise NLRuntimeError(f"Logic '{logic_name}' not found in module '{module_name}'")
        return resolved_logic

    @staticmethod
    def _bind_call_arguments(
        resolved_logic: ResolvedLogic,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind Python call arguments to logic parameter names."""

        param_names = [item.name for item in resolved_logic.logic.params]
        if len(args) > len(param_names):
            raise NLRuntimeError(
                f"Logic accepts {len(param_names)} positional argument(s), got {len(args)}"
            )

        bound: dict[str, Any] = {}
        for index, value in enumerate(args):
            bound[param_names[index]] = value

        for key, value in kwargs.items():
            if key not in param_names:
                raise NLRuntimeError(f"Unknown logic argument '{key}'")
            if key in bound:
                raise NLRuntimeError(f"Logic argument '{key}' received multiple values")
            bound[key] = value

        missing = [name for name in param_names if name not in bound]
        if missing:
            missing_text = ", ".join(missing)
            raise NLRuntimeError(f"Missing required logic argument(s): {missing_text}")

        return bound

    async def _execute_logic(
        self,
        *,
        module_name: str,
        logic_name: str,
        resolved_logic: ResolvedLogic,
        raw_logic_args: Mapping[str, Any],
    ) -> Any:
        """Execute one resolved logic with validated input and output contracts."""

        dumped_logic_args = self._validate_logic_args(resolved_logic, raw_logic_args)
        runtime_context: dict[str, Any] = {}

        helper_aliases = list(resolved_logic.helpers.keys())
        helper_mentions = list(dict.fromkeys(resolved_logic.logic.mentions))
        helper_signatures = self._build_helper_signatures(resolved_logic.helpers)
        helper_instructions = self._build_helper_instructions(helper_signatures)
        output_schema = self._build_output_schema(resolved_logic.return_type)
        output_required_fields = self._extract_required_output_fields(output_schema)

        toolkit: list[Any] = []
        repl: Any | None = None
        if self._include_pyrepl_tools:
            toolkit, repl = await self._prepare_codeact_runtime(
                logic_args=dumped_logic_args,
                runtime_context=runtime_context,
                helper_paths=resolved_logic.helper_paths,
            )

        request = LogicExecutionRequest(
            module_name=module_name,
            logic=resolved_logic.logic,
            logic_args=dumped_logic_args,
            runtime_context=runtime_context,
            helper_aliases=helper_aliases,
            helper_functions=dict(resolved_logic.helpers),
            helper_mentions=helper_mentions,
            helper_instructions=helper_instructions,
            helper_signatures=helper_signatures,
            output_schema=output_schema,
            output_required_fields=output_required_fields,
        )

        invoker = self._invoker or self._default_invoker
        raw_output: Any = None
        result_submitted = False
        submitted_payload: Any = None
        try:
            raw_output = await invoker(request, toolkit)
            if repl is not None:
                result_submitted, submitted_payload = await self._read_repl_result_channel(repl)
        except NLRuntimeError:
            raise
        except Exception as exc:
            raise NLRuntimeError(
                f"NL logic execution failed for '{module_name}#{logic_name}': {exc}"
            ) from exc
        finally:
            if repl is not None:
                repl.close()

        if result_submitted:
            normalized_output = self._normalize_invoker_output(submitted_payload)
        else:
            normalized_output = self._normalize_invoker_output(raw_output)
        try:
            adapter = TypeAdapter(resolved_logic.return_type)
            return adapter.validate_python(normalized_output)
        except ValidationError as exc:
            raise NLRuntimeError(
                f"NL logic output validation failed for '{module_name}#{logic_name}'"
            ) from exc

    def _validate_logic_args(
        self,
        resolved_logic: ResolvedLogic,
        raw_logic_args: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate logic input arguments with declared parameter types."""

        dumped_values: dict[str, Any] = {}
        for param_name, param_type in resolved_logic.param_types.items():
            adapter = TypeAdapter(param_type)
            try:
                typed_value = adapter.validate_python(raw_logic_args[param_name])
            except ValidationError as exc:
                raise NLRuntimeError(
                    f"NL logic input validation failed for parameter '{param_name}'"
                ) from exc
            dumped_values[param_name] = adapter.dump_python(typed_value, mode="json")

        return dumped_values

    @staticmethod
    def _select_repl_tools(repl: Any) -> list[Any]:
        """Select only execute_code to enforce CodeAct behavior."""

        selected_tools: list[Any] = []
        for tool_obj in repl.toolset:
            name = getattr(getattr(tool_obj, "_tool", None), "name", None)
            if name == "execute_code":
                selected_tools.append(tool_obj)
                continue
            if getattr(tool_obj, "name", None) == "execute_code":
                selected_tools.append(tool_obj)

        if selected_tools:
            return selected_tools
        return list(repl.toolset)

    async def _prepare_codeact_runtime(
        self,
        *,
        logic_args: Mapping[str, Any],
        runtime_context: Mapping[str, Any],
        helper_paths: Mapping[str, str],
    ) -> tuple[list[Any], Any]:
        """Prepare PyRepl with preloaded logic arguments and helper wrappers."""

        if self._repl_factory is None:
            raise NLRuntimeError("PyRepl factory is not configured")
        repl = self._repl_factory()
        toolkit = self._select_repl_tools(repl)

        payload_code = self._build_payload_bootstrap_code(logic_args, runtime_context)
        payload_result = await repl.execute(payload_code)
        if isinstance(payload_result, Mapping) and payload_result.get("success") is False:
            raise NLRuntimeError(
                f"Failed to inject logic runtime payload: {payload_result.get('error')}"
            )

        helper_code = self._build_helper_bootstrap_code(helper_paths)
        if helper_code:
            helper_result = await repl.execute(helper_code)
            if isinstance(helper_result, Mapping) and helper_result.get("success") is False:
                raise NLRuntimeError(
                    f"Failed to inject helper wrappers: {helper_result.get('error')}"
                )

        return toolkit, repl

    @staticmethod
    def _build_payload_bootstrap_code(
        logic_args: Mapping[str, Any],
        runtime_context: Mapping[str, Any],
    ) -> str:
        """Build REPL bootstrap code for logic arguments and context."""

        serialized_args = json.dumps(dict(logic_args), ensure_ascii=False, default=str)
        serialized_context = json.dumps(dict(runtime_context), ensure_ascii=False, default=str)
        args_literal = json.dumps(serialized_args, ensure_ascii=False)
        context_literal = json.dumps(serialized_context, ensure_ascii=False)

        return "\n".join(
            [
                "import json",
                f"logic_args = json.loads({args_literal})",
                f"runtime_context = json.loads({context_literal})",
                "__nlvm_result_channel = {'submitted': False, 'payload': None}",
                "def result_done(payload):",
                "    __nlvm_result_channel['submitted'] = True",
                "    __nlvm_result_channel['payload'] = payload",
                "    return {'submitted': True}",
            ]
        )

    @staticmethod
    def _build_helper_bootstrap_code(helper_paths: Mapping[str, str]) -> str:
        """Build REPL wrappers that expose imported Python functions."""

        if not helper_paths:
            return ""

        lines = [
            "import asyncio",
            "import inspect",
            "import importlib",
            "",
            "def _resolve_dotted_callable(_dotted):",
            "    _parts = _dotted.split('.')",
            "    for _split_index in range(len(_parts), 0, -1):",
            "        _module_name = '.'.join(_parts[:_split_index])",
            "        try:",
            "            _module = importlib.import_module(_module_name)",
            "            _obj = _module",
            "            for _attr in _parts[_split_index:]:",
            "                _obj = getattr(_obj, _attr)",
            "            return _obj",
            "        except ModuleNotFoundError:",
            "            continue",
            "    raise RuntimeError(f'Cannot resolve helper path: {_dotted}')",
            "",
            "def _run_async_helper(_coroutine):",
            "    try:",
            "        asyncio.get_running_loop()",
            "    except RuntimeError:",
            "        return asyncio.run(_coroutine)",
            "    raise RuntimeError('REPL helper requires a non-running event loop context')",
            "",
        ]

        for alias, path in helper_paths.items():
            helper_ref = f"_helper_ref_{alias}"
            helper_param_names = f"_helper_param_names_{alias}"
            path_literal = json.dumps(path, ensure_ascii=False)
            lines.extend(
                [
                    f"{helper_ref} = _resolve_dotted_callable({path_literal})",
                    f"{helper_param_names} = list(inspect.signature({helper_ref}).parameters.keys())",
                    f"def {alias}(*args, **kwargs):",
                    "    _kwargs = dict(kwargs)",
                    "    _logic_input = logic_args.get('input') if isinstance(logic_args, dict) else None",
                    "    if isinstance(_logic_input, dict):",
                    f"        if 'workspace' in {helper_param_names} and 'workspace' not in _kwargs:",
                    f"            _workspace_position = {helper_param_names}.index('workspace')",
                    "            if len(args) <= _workspace_position and 'workspace' in _logic_input:",
                    "                _kwargs['workspace'] = _logic_input['workspace']",
                    f"        if 'payload' in {helper_param_names} and 'payload' not in _kwargs:",
                    f"            _payload_position = {helper_param_names}.index('payload')",
                    "            if len(args) <= _payload_position:",
                    "                _kwargs['payload'] = _logic_input",
                    f"        if 'input' in {helper_param_names} and 'input' not in _kwargs:",
                    f"            _input_position = {helper_param_names}.index('input')",
                    "            if len(args) <= _input_position:",
                    "                _kwargs['input'] = _logic_input",
                    f"    _result = {helper_ref}(*args, **_kwargs)",
                    "    if asyncio.iscoroutine(_result):",
                    "        return _run_async_helper(_result)",
                    "    return _result",
                    "",
                ]
            )

        return "\n".join(lines).strip()

    @staticmethod
    def _build_helper_signatures(
        helper_functions: Mapping[str, Callable[..., Any]],
    ) -> dict[str, str]:
        """Build stable helper signatures for prompt guidance."""

        signatures: dict[str, str] = {}
        for alias, helper_func in helper_functions.items():
            try:
                signature = str(inspect.signature(helper_func))
            except (TypeError, ValueError):
                signature = "(...)"
            signatures[alias] = f"{alias}{signature}"
        return signatures

    @staticmethod
    def _build_helper_instructions(helper_signatures: Mapping[str, str]) -> list[str]:
        """Build helper usage hints for the invoker prompt."""

        helper_aliases = list(helper_signatures.keys())
        if not helper_aliases:
            return []

        lines = [
            "# Runtime CodeAct Mode",
            "Use execute_code as your action tool.",
            f"Available helper functions in REPL: {', '.join(helper_aliases)}",
            "Use those helpers inside Python code blocks executed by execute_code.",
            "REPL session is persistent across execute_code calls (notebook-style state is preserved).",
            "Start with input_payload = logic_args['input'] and pass required helper arguments explicitly.",
            "Call helpers with exactly their declared parameters; do not invent keyword names.",
            "When helper/API behavior is uncertain, do one quick grounding step: print type/keys/sample output before continuing.",
            "Prefer iterative flow: inspect -> adapt -> compute -> result_done(payload).",
            "When final payload is ready in Python, call result_done(payload).",
            "After result_done(payload), return any short acknowledgement text; runtime will use payload directly.",
            "Do not regenerate final JSON in natural language after result_done(payload).",
        ]
        if helper_signatures:
            lines.append("Helper signatures:")
            for helper_name in helper_aliases:
                lines.append(f"- {helper_signatures[helper_name]}")

        if "invoke_logic" in helper_signatures:
            lines.append(
                "For invoke_logic use invoke_logic('module#logic', payload={...}); "
                "do not use logic_path keyword."
            )

        return ["\n".join(lines)]

    @staticmethod
    def _build_output_schema(return_type: Any) -> dict[str, Any]:
        """Build JSON schema for logic output contract."""

        adapter = TypeAdapter(return_type)
        schema = adapter.json_schema()
        if isinstance(schema, dict):
            return schema
        return {"type": "object"}

    @staticmethod
    def _extract_required_output_fields(output_schema: Mapping[str, Any]) -> list[str]:
        """Extract required top-level output fields from one JSON schema."""

        required = output_schema.get("required")
        if not isinstance(required, list):
            return []
        return [item for item in required if isinstance(item, str)]

    @staticmethod
    def _normalize_invoker_output(raw_output: Any) -> Any:
        """Normalize invoker output into JSON-like Python values."""

        if not isinstance(raw_output, str):
            return raw_output

        text = raw_output.strip()
        if not text:
            raise NLRuntimeError("LLM returned empty output text")

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return parsed

        raise NLRuntimeError("LLM returned non-JSON output text")

    async def _read_repl_result_channel(self, repl: Any) -> tuple[bool, Any]:
        """Read result channel populated by result_done(payload) in REPL."""

        probe_code = "\n".join(
            [
                "import json",
                "# __nlvm_result_channel_probe__",
                "print(json.dumps(__nlvm_result_channel, ensure_ascii=False, default=str))",
            ]
        )

        try:
            probe_result = await repl.execute(probe_code)
        except Exception as exc:
            self._logger.debug("Failed to probe REPL result channel: %s", exc)
            return False, None

        if isinstance(probe_result, Mapping) and probe_result.get("success") is False:
            self._logger.debug("REPL result channel probe failed: %s", probe_result.get("error"))
            return False, None

        if not isinstance(probe_result, Mapping):
            return False, None

        stdout = probe_result.get("stdout")
        if not isinstance(stdout, str):
            return False, None

        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, Mapping):
                continue

            submitted = bool(parsed.get("submitted", False))
            if not submitted:
                return False, None
            return True, parsed.get("payload")

        return False, None

    def _resolve_llm_interface_for_logic(self, logic: NLLogic) -> Any:
        """Resolve llm_interface for one logic from logic-level or runtime defaults."""

        llm_config: NLLMConfig | None = logic.llm
        if llm_config is not None:
            if self._logic_llm_resolver is None:
                raise NLRuntimeError(
                    "Logic declares llm but runtime has no logic_llm_resolver configured"
                )
            return self._logic_llm_resolver(llm_config.provider_id, llm_config.model_name)

        if self._llm_interface is None:
            raise NLRuntimeError(
                "No default llm_interface configured; declare llm in logic or set runtime llm_interface"
            )
        return self._llm_interface

    async def _default_invoker(self, request: LogicExecutionRequest, toolkit: list[Any]) -> Any:
        """Default SimpleLLMFunc invoker for one logic execution."""

        if llm_function is None:
            raise NLRuntimeError(
                "Default invoker requires SimpleLLMFunc; pass custom invoker or install codeact extra"
            )

        selected_llm_interface = self._resolve_llm_interface_for_logic(request.logic)

        @llm_function(
            llm_interface=selected_llm_interface,
            toolkit=toolkit,
            max_tool_calls=self._max_tool_calls,
        )
        async def run_logic_fragment(
            module_name: str,
            logic_name: str,
            logic_signature: str,
            logic_instruction: str,
            logic_args: dict[str, Any],
            runtime_context: dict[str, Any],
            helper_aliases: list[str],
            helper_mentions: list[str],
            helper_instructions: list[str],
            helper_signatures: dict[str, str],
            output_schema: dict[str, Any],
            output_required_fields: list[str],
        ) -> str:
            """You execute one NL logic fragment as backend runtime logic.

            Follow logic_instruction strictly as the business source of truth.
            Respect the logic_signature input/output contract strictly.
            Use execute_code as your only action tool.
            REPL already has:
            - logic_args: dict of validated logic input arguments
            - runtime_context: dict runtime metadata/context
            - helper functions listed in helper_aliases
            REPL state is persistent across execute_code calls (like one notebook kernel).
            You may inspect intermediate values and reuse variables in later steps.
            If helper_mentions is not empty, prioritize those helpers when relevant.
            helper_signatures gives exact helper call signatures.
            When a helper/API return shape is uncertain, ground quickly: print concise type/keys/sample output first.
            Use iterative workflow: inspect -> adapt -> compute -> submit.
            In REPL there is a result_done(payload) helper; call it when you have final structured result.
            After result_done(payload), do not regenerate the final JSON with free-form text.
            output_schema is the JSON schema your final output must satisfy.
            output_required_fields are required top-level keys and must be present.
            Do not rename or omit schema keys.
            Return plain JSON text only (no markdown, no XML tags).
            """

            raise NotImplementedError

        logic_signature = self._format_logic_signature(request.logic)
        raw_output = await run_logic_fragment(
            module_name=request.module_name,
            logic_name=request.logic.name,
            logic_signature=logic_signature,
            logic_instruction=request.logic.body,
            logic_args=request.logic_args,
            runtime_context=request.runtime_context,
            helper_aliases=request.helper_aliases,
            helper_mentions=request.helper_mentions,
            helper_instructions=request.helper_instructions,
            helper_signatures=request.helper_signatures,
            output_schema=request.output_schema,
            output_required_fields=request.output_required_fields,
        )
        return self._normalize_invoker_output(raw_output)

    @staticmethod
    def _format_logic_signature(logic: NLLogic) -> str:
        """Format logic signature for prompt context."""

        params = ", ".join(f"{item.name}: {item.type_ref}" for item in logic.params)
        return f"logic {logic.name}({params}) -> {logic.return_type_ref}"


__all__ = [
    "LogicExecutionRequest",
    "LogicInvoker",
    "NLRuntime",
    "NLRuntimeError",
]
