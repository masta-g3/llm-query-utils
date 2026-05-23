import logging
import os
import asyncio
import json
from copy import deepcopy
import subprocess
import tempfile
import time
import traceback
import warnings
from types import SimpleNamespace
from typing import Type, Optional, Union

from pydantic import BaseModel
from litellm import completion, cost_per_token, get_model_info
from litellm.exceptions import BadRequestError
import instructor
from claude_agent_sdk import (
    query as agent_sdk_query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

from .config import DEFAULT_MODEL
from .usage import UsageData, fire_usage_callback

warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

logger = logging.getLogger(__name__)

CODEX_MODEL_PREFIXES = ("gpt-5",)
PI_OPTION_KEYS = {
    "model",
    "provider",
    "thinking",
    "tools",
    "timeout",
    "cwd",
    "isolate_resources",
    "no_session",
}


def _calculate_usage_costs(
    usage,
    llm_model: str,
) -> tuple[float, float, Optional[float], Optional[float]]:
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    prompt_cost, completion_cost = cost_per_token(
        model=llm_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
    )

    model_info = get_model_info(model=llm_model)
    input_rate = model_info.get("input_cost_per_token")
    cache_read_rate = model_info.get("cache_read_input_token_cost")
    cache_creation_rate = model_info.get("cache_creation_input_token_cost")

    cache_read_cost = (
        cache_read_tokens * cache_read_rate
        if cache_read_tokens and cache_read_rate is not None
        else None
    )
    cache_creation_effective_rate = (
        cache_creation_rate if cache_creation_rate is not None else (
            input_rate * 2 if input_rate is not None else None
        )
    )
    cache_creation_cost = (
        cache_creation_tokens * cache_creation_effective_rate
        if cache_creation_tokens and cache_creation_effective_rate is not None
        else None
    )

    return prompt_cost, completion_cost, cache_creation_cost, cache_read_cost


def _log_usage(
    usage, llm_model: str, process_id: str = None, verbose: bool = False
) -> None:
    prompt_cost, completion_cost, cache_creation_cost, cache_read_cost = (
        _calculate_usage_costs(usage, llm_model)
    )

    if verbose:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        total_cost = (prompt_cost or 0) + (completion_cost or 0)
        cache_info = ""
        cache_creation_tokens_val = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens_val = getattr(usage, "cache_read_input_tokens", 0) or 0
        if cache_creation_tokens_val:
            cache_info += f", cache_create={cache_creation_tokens_val:,}"
        if cache_read_tokens_val:
            cache_info += f", cache_read={cache_read_tokens_val:,}"
        logger.info(
            f"[{llm_model}] {usage.prompt_tokens:,}+{usage.completion_tokens:,}="
            f"{total_tokens:,} tokens, ${total_cost:.4f}{cache_info}"
        )

    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    fire_usage_callback(UsageData(
        model=llm_model,
        process_id=process_id,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
        cache_creation_cost=cache_creation_cost,
        cache_read_cost=cache_read_cost,
    ))


def _fire_agent_usage(
    llm_model: str,
    process_id: Optional[str],
    usage_data,
    total_cost_usd: Optional[float],
) -> None:
    usage = usage_data if isinstance(usage_data, dict) else {}
    if usage_data is None:
        logger.debug("Agent SDK usage missing; emitting zero-token usage callback.")

    fire_usage_callback(UsageData(
        model=llm_model,
        process_id=process_id,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        prompt_cost=float(total_cost_usd or 0),
        completion_cost=0,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
    ))


async def _agent_sdk_query(
    system_message: Optional[str],
    user_message: str,
    llm_model: str,
    allowed_tools: Optional[list[str]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    process_id: str = None,
) -> Union[str, BaseModel]:
    """Execute single-turn query via Agent SDK with logging."""
    model = llm_model.split("/")[-1]

    logger.debug(f"Agent SDK query starting (model={model}, prompt_len={len(user_message)})")

    output_format = None
    if output_schema:
        output_format = {
            "type": "json_schema",
            "schema": output_schema.model_json_schema(),
        }

    options = ClaudeAgentOptions(
        system_prompt=system_message or "",
        model=model,
        allowed_tools=allowed_tools or [],
        output_format=output_format,
    )

    text_parts = []
    message_count = 0
    block_count = 0
    tool_use_count = 0
    structured_result = None
    error_result = None
    total_cost_usd = None
    usage_data = None
    has_result_message = False

    async for message in agent_sdk_query(prompt=user_message, options=options):
        message_count += 1
        if isinstance(message, ResultMessage):
            has_result_message = True
            total_cost_usd = message.total_cost_usd
            usage_data = message.usage
            if message.is_error:
                error_result = message.result
            elif output_schema and message.structured_output:
                structured_result = output_schema.model_validate(message.structured_output)
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                block_count += 1
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_use_count += 1
                    logger.info(f"Agent tool call #{tool_use_count}: {block.name}")

    if has_result_message:
        _fire_agent_usage(
            llm_model=llm_model,
            process_id=process_id,
            usage_data=usage_data,
            total_cost_usd=total_cost_usd,
        )

    if error_result:
        raise RuntimeError(f"Agent SDK error: {error_result}")

    if structured_result is not None:
        logger.info(
            f"Agent SDK query complete: {message_count} messages, "
            f"{tool_use_count} tool calls, structured output returned"
        )
        return structured_result

    result = "".join(text_parts)
    logger.info(
        f"Agent SDK query complete: {message_count} messages, "
        f"{block_count} blocks, {tool_use_count} tool calls, {len(result)} chars"
    )

    if not result.strip():
        raise RuntimeError("Agent SDK returned empty response")
    return result


def _is_codex_model(llm_model: str) -> bool:
    model = llm_model.split("/")[-1]
    return model.startswith(CODEX_MODEL_PREFIXES)


def _build_codex_prompt(system_message: Optional[str], user_message: str) -> str:
    if system_message is None:
        return user_message
    return f"System:\n{system_message}\n\nUser:\n{user_message}"


def _parse_codex_event_stream(event_stream: str) -> tuple[Optional[dict], Optional[str]]:
    usage = None
    error_message = None
    for line in event_stream.splitlines():
        raw = line.strip()
        if not raw or not raw.startswith("{"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        event_type = payload.get("type")
        if event_type == "turn.completed":
            usage = payload.get("usage")
        elif event_type == "turn.failed":
            error = payload.get("error") or {}
            error_message = error.get("message") or payload.get("message")
        elif event_type == "error":
            message = payload.get("message")
            if message:
                error_message = message
    return usage, error_message


def _to_codex_strict_schema(schema: dict) -> dict:
    strict_schema = deepcopy(schema)

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)

            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
                for value in properties.values():
                    _walk(value)

            items = node.get("items")
            if items is not None:
                _walk(items)

            for key in ("allOf", "anyOf", "oneOf"):
                variants = node.get(key)
                if isinstance(variants, list):
                    for variant in variants:
                        _walk(variant)

            for defs_key in ("$defs", "definitions"):
                defs = node.get(defs_key)
                if isinstance(defs, dict):
                    for value in defs.values():
                        _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(strict_schema)
    return strict_schema


def _codex_usage_to_namespace(usage: dict) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get(
            "cached_input_tokens",
            usage.get("cache_read_input_tokens", 0),
        ),
    )


def _codex_pricing_model(llm_model: str) -> str:
    return (llm_model or "").split("/")[-1]


def _is_pricing_lookup_error(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, BadRequestError) or "isn't mapped yet" in message


def _calculate_codex_usage_costs(
    usage: SimpleNamespace,
    llm_model: str,
) -> tuple[float, float, Optional[float], Optional[float]]:
    try:
        return _calculate_usage_costs(usage, _codex_pricing_model(llm_model))
    except Exception as exc:
        if not _is_pricing_lookup_error(exc):
            raise
        logger.warning(
            "Codex API-equivalent pricing unavailable for model=%s; logging zero cost",
            llm_model,
        )
        return 0, 0, None, None


def _fire_codex_usage(
    usage: Optional[dict],
    llm_model: str,
    process_id: Optional[str],
    pricing_model: Optional[str] = None,
    logged_model: Optional[str] = None,
) -> None:
    if not usage:
        return

    codex_usage = _codex_usage_to_namespace(usage)
    prompt_cost, completion_cost, cache_creation_cost, cache_read_cost = (
        _calculate_codex_usage_costs(codex_usage, pricing_model or llm_model)
    )

    fire_usage_callback(UsageData(
        model=logged_model or llm_model,
        process_id=process_id,
        prompt_tokens=codex_usage.prompt_tokens,
        completion_tokens=codex_usage.completion_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        cache_creation_input_tokens=codex_usage.cache_creation_input_tokens,
        cache_read_input_tokens=codex_usage.cache_read_input_tokens,
        cache_creation_cost=cache_creation_cost,
        cache_read_cost=cache_read_cost,
    ))


def _run_codex_login_preflight() -> None:
    try:
        result = subprocess.run(
            ["codex", "login", "status"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Codex CLI not found. Install Codex CLI before using use_codex_sdk=True."
        ) from exc
    combined_output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "Logged in using ChatGPT" not in combined_output:
        raise RuntimeError(
            "Codex CLI is not authenticated. Run `codex login` with ChatGPT before using use_codex_sdk=True."
        )


def _run_codex_exec(
    model: str,
    prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    extra_args: Optional[list[str]] = None,
) -> tuple[str, Optional[dict], str]:
    schema_path = None
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name

    cmd = [
        "codex",
        "exec",
        "-m",
        model,
        prompt,
        "--json",
        "--output-last-message",
        output_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    if output_schema is not None:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as schema_file:
            json.dump(_to_codex_strict_schema(output_schema.model_json_schema()), schema_file)
            schema_path = schema_file.name
        cmd.extend(["--output-schema", schema_path])

    try:
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Codex CLI not found. Install Codex CLI before using use_codex_sdk=True."
            ) from exc

        with open(output_path, "r") as f:
            output_text = f.read().strip()
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
        if schema_path is not None and os.path.exists(schema_path):
            os.remove(schema_path)

    event_stream = "\n".join(
        part for part in (result.stdout, result.stderr) if part and part.strip()
    )
    usage, error_message = _parse_codex_event_stream(event_stream)

    if result.returncode != 0:
        if error_message:
            raise RuntimeError(f"Codex CLI error: {error_message}")
        raise RuntimeError(f"Codex CLI error: {event_stream.strip() or 'unknown error'}")

    if not output_text:
        raise RuntimeError("Codex CLI returned empty response")

    return output_text, usage, event_stream


async def _codex_sdk_query(
    system_message: Optional[str],
    user_message: str,
    llm_model: str,
    output_schema: Optional[Type[BaseModel]] = None,
    process_id: str = None,
    codex_options: Optional[dict] = None,
) -> Union[str, BaseModel]:
    if user_message is None:
        raise ValueError("user_message is required when use_codex_sdk=True")

    options = codex_options or {}
    if not isinstance(options, dict):
        raise ValueError("codex_options must be a dictionary when provided")

    model = (options.get("model") or llm_model).split("/")[-1]
    if not _is_codex_model(model):
        supported_prefixes = ", ".join(f"{prefix}*" for prefix in CODEX_MODEL_PREFIXES)
        raise ValueError(
            f"Unsupported Codex model '{model}'. Supported model prefixes: {supported_prefixes}"
        )
    extra_args = options.get("extra_args")
    if extra_args is not None and not isinstance(extra_args, list):
        raise ValueError("codex_options.extra_args must be a list when provided")

    prompt = _build_codex_prompt(system_message, user_message)

    await asyncio.to_thread(_run_codex_login_preflight)
    output_text, usage, _ = await asyncio.to_thread(
        _run_codex_exec,
        model,
        prompt,
        output_schema,
        extra_args,
    )

    if output_schema is not None:
        result = output_schema.model_validate_json(output_text)
    else:
        result = output_text

    _fire_codex_usage(
        usage,
        llm_model,
        process_id,
        pricing_model=model,
        logged_model=model if options.get("model") else None,
    )
    logger.info(f"Codex query complete: {len(output_text)} chars")

    return result


def _validate_pi_options(pi_options: Optional[dict]) -> dict:
    options = {} if pi_options is None else pi_options
    if not isinstance(options, dict):
        raise ValueError("pi_options must be a dictionary when provided")

    unknown_keys = sorted(set(options) - PI_OPTION_KEYS)
    if unknown_keys:
        raise ValueError(f"Unsupported pi_options: {', '.join(unknown_keys)}")

    for key in ("model", "provider", "thinking", "cwd"):
        if options.get(key) is not None and not isinstance(options[key], str):
            raise ValueError(f"pi_options.{key} must be a string when provided")

    tools = options.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
            raise ValueError("pi_options.tools must be a list of strings when provided")

    timeout = options.get("timeout")
    if timeout is not None and (not isinstance(timeout, int) or isinstance(timeout, bool)):
        raise ValueError("pi_options.timeout must be an integer when provided")

    for key in ("isolate_resources", "no_session"):
        if options.get(key) is not None and not isinstance(options[key], bool):
            raise ValueError(f"pi_options.{key} must be a boolean when provided")

    return options


def _build_pi_structured_system_message(
    system_message: Optional[str],
    output_schema: Optional[Type[BaseModel]],
) -> Optional[str]:
    if output_schema is None:
        return system_message

    instruction = (
        "Return only valid JSON matching this JSON Schema. "
        "Do not include Markdown fences or explanatory text.\n"
        f"{json.dumps(output_schema.model_json_schema(), separators=(',', ':'))}"
    )
    if system_message:
        return f"{system_message}\n\n{instruction}"
    return instruction


def _build_pi_command(
    llm_model: str,
    system_message: Optional[str],
    user_message: str,
    pi_options: Optional[dict] = None,
) -> tuple[list[str], Optional[int], Optional[str]]:
    options = _validate_pi_options(pi_options)
    cmd = ["pi", "--mode", "json"]

    if options.get("no_session", True):
        cmd.append("--no-session")

    if options.get("isolate_resources", True):
        cmd.extend([
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
        ])

    tools = options.get("tools")
    if tools:
        cmd.extend(["--tools", ",".join(tools)])
    else:
        cmd.append("--no-tools")

    provider = options.get("provider")
    if provider:
        cmd.extend(["--provider", provider])

    model_name = options.get("model") or llm_model
    if model_name:
        cmd.extend(["--model", model_name])

    thinking = options.get("thinking")
    if thinking:
        cmd.extend(["--thinking", thinking])

    if system_message:
        cmd.extend(["--system-prompt", system_message])

    cmd.append(user_message)
    return cmd, options.get("timeout"), options.get("cwd")


def _extract_pi_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _parse_pi_event_stream(event_stream: str) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    text = None
    usage = None
    error_message = None
    saw_message_end = False

    def capture_message(message: dict) -> bool:
        nonlocal text, usage
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return False
        message_text = _extract_pi_text(message)
        if message_text:
            text = message_text
        message_usage = message.get("usage")
        if isinstance(message_usage, dict):
            usage = message_usage
        return bool(message_text)

    for line in event_stream.splitlines():
        raw = line.strip()
        if not raw or not raw.startswith("{"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(payload.get("error"), str):
            error_message = payload["error"]
        elif isinstance(payload.get("error"), dict):
            error_message = payload["error"].get("message") or error_message
        elif isinstance(payload.get("message"), str) and payload.get("type") == "error":
            error_message = payload["message"]

        event_type = payload.get("type")
        if event_type == "message_end":
            saw_message_end = capture_message(payload.get("message") or {}) or saw_message_end
        elif event_type == "agent_end" and not saw_message_end:
            messages = payload.get("messages") or []
            if isinstance(messages, list):
                for message in messages:
                    if capture_message(message):
                        break
        elif event_type == "compaction_end" and payload.get("errorMessage"):
            error_message = payload["errorMessage"]
        elif event_type == "message_update":
            assistant_event = payload.get("assistantMessageEvent") or {}
            if isinstance(assistant_event, dict) and assistant_event.get("type") == "error":
                error_message = assistant_event.get("message") or error_message

    return text, usage, error_message


def _fire_pi_usage(
    usage: Optional[dict],
    llm_model: str,
    process_id: Optional[str],
) -> None:
    if not usage:
        return

    cost = usage.get("cost") or {}
    fire_usage_callback(UsageData(
        model=llm_model,
        process_id=process_id,
        prompt_tokens=usage.get("input", 0),
        completion_tokens=usage.get("output", 0),
        prompt_cost=cost.get("input", 0),
        completion_cost=cost.get("output", 0),
        cache_creation_input_tokens=usage.get("cacheWrite", 0),
        cache_read_input_tokens=usage.get("cacheRead", 0),
        cache_creation_cost=cost.get("cacheWrite"),
        cache_read_cost=cost.get("cacheRead"),
    ))


def _run_pi_cli(
    llm_model: str,
    system_message: Optional[str],
    user_message: str,
    pi_options: Optional[dict] = None,
) -> tuple[str, Optional[dict], str]:
    cmd, timeout, cwd = _build_pi_command(llm_model, system_message, user_message, pi_options)

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Pi CLI timed out after {timeout} seconds") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Pi CLI not found. Install pi before using use_pi_sdk=True."
        ) from exc

    event_stream = "\n".join(
        part for part in (result.stdout, result.stderr) if part and part.strip()
    )
    output_text, usage, error_message = _parse_pi_event_stream(event_stream)

    if result.returncode != 0:
        if error_message:
            raise RuntimeError(f"Pi CLI error: {error_message}")
        details = event_stream.strip() or "unknown error"
        if len(details) > 1000:
            details = f"...{details[-1000:]}"
        raise RuntimeError(f"Pi CLI error: {details}")

    if not output_text or not output_text.strip():
        raise RuntimeError("Pi CLI returned empty response")

    return output_text.strip(), usage, event_stream


async def _pi_sdk_query(
    system_message: Optional[str],
    user_message: str,
    llm_model: str,
    output_schema: Optional[Type[BaseModel]] = None,
    process_id: str = None,
    pi_options: Optional[dict] = None,
) -> Union[str, BaseModel]:
    if user_message is None:
        raise ValueError("user_message is required when use_pi_sdk=True")

    system_message = _build_pi_structured_system_message(system_message, output_schema)
    output_text, usage, _ = await asyncio.to_thread(
        _run_pi_cli,
        llm_model,
        system_message,
        user_message,
        pi_options,
    )

    _fire_pi_usage(usage, llm_model, process_id)
    logger.info(f"Pi query complete: {len(output_text)} chars")

    if output_schema is not None:
        return output_schema.model_validate_json(output_text)
    return output_text


def run_query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: Optional[Type[BaseModel]] = None,
    llm_model: str = DEFAULT_MODEL,
    temperature: float = 1,
    process_id: str = None,
    messages: Optional[list[dict]] = None,
    verbose: bool = False,
    enable_extended_thinking: bool = False,
    extended_thinking_budget_tokens: Optional[int] = None,
    thinking_options: Optional[dict] = None,
    extended_thinking_beta: Optional[str] = None,
    use_agent_sdk: bool = True,
    use_codex_sdk: bool = False,
    use_pi_sdk: bool = False,
    sdk_allowed_tools: Optional[list[str]] = None,
    codex_options: Optional[dict] = None,
    pi_options: Optional[dict] = None,
    **kwargs,
) -> Union[BaseModel, str]:
    """Run an LLM query, routing through Agent SDK (Claude) or LiteLLM/Instructor (others).

    Args:
        use_agent_sdk: Route query through Agent SDK instead of direct API (default: True).
            Automatically disabled for: custom messages, extended thinking, non-Claude models.
        use_codex_sdk: Route query through Codex CLI transport (default: False).
            Supported for gpt-5* models and single-turn user_message input.
        use_pi_sdk: Route query through Pi CLI JSON mode (default: False).
        sdk_allowed_tools: Tools the agent can use (e.g., ["Read"] for image analysis).
    """
    thinking_requested = (
        enable_extended_thinking
        or thinking_options is not None
        or extended_thinking_budget_tokens is not None
    )

    if use_pi_sdk:
        if messages is not None:
            raise ValueError("messages is not supported when use_pi_sdk=True")
        if sdk_allowed_tools:
            raise ValueError(
                "sdk_allowed_tools is not supported when use_pi_sdk=True; use pi_options.tools"
            )
        if thinking_requested or extended_thinking_beta is not None:
            raise ValueError(
                "extended thinking options are not supported when use_pi_sdk=True; use pi_options.thinking"
            )

    use_codex_route = (
        use_codex_sdk
        and _is_codex_model(llm_model)
        and messages is None
        and not thinking_requested
    )
    if use_codex_route and sdk_allowed_tools:
        raise ValueError("sdk_allowed_tools is not supported when use_codex_sdk=True")

    if use_agent_sdk:
        if messages is not None:
            use_agent_sdk = False
        elif enable_extended_thinking or thinking_options:
            use_agent_sdk = False
        elif "claude" not in llm_model.lower():
            use_agent_sdk = False

    if messages is None and user_message is None:
        raise ValueError(
            "Either 'messages' parameter or 'user_message' parameter must be provided"
        )

    if use_pi_sdk:
        return asyncio.run(_pi_sdk_query(
            system_message,
            user_message,
            llm_model,
            output_schema=model,
            process_id=process_id,
            pi_options=pi_options,
        ))

    if use_codex_route:
        return asyncio.run(_codex_sdk_query(
            system_message,
            user_message,
            llm_model,
            output_schema=model,
            process_id=process_id,
            codex_options=codex_options,
        ))

    if use_agent_sdk:
        return asyncio.run(_agent_sdk_query(
            system_message, user_message, llm_model, sdk_allowed_tools,
            output_schema=model,
            process_id=process_id,
        ))

    temperature = (
        None if any(x in llm_model for x in ["o1", "r1", "o3", "gpt-5"]) else temperature
    )

    if thinking_requested and any(
        provider in llm_model.lower() for provider in ["claude", "anthropic"]
    ):
        thinking_payload = kwargs.get("thinking")
        if thinking_payload is None:
            thinking_payload = (thinking_options or {}).copy()
        else:
            thinking_payload = thinking_payload.copy()

        thinking_payload.setdefault("type", "enabled")
        budget_tokens = extended_thinking_budget_tokens
        if budget_tokens is None:
            budget_tokens = 4096
        if budget_tokens:
            thinking_payload.setdefault("budget_tokens", budget_tokens)

        kwargs["thinking"] = thinking_payload

        beta_flag = extended_thinking_beta
        if beta_flag is None:
            beta_flag = os.environ.get("ANTHROPIC_EXTENDED_THINKING_BETA")

        if beta_flag:
            extra_headers = kwargs.get("extra_headers")
            extra_headers = (extra_headers or {}).copy()
            if "anthropic-beta" in extra_headers:
                existing = [
                    h.strip() for h in extra_headers["anthropic-beta"].split(",")
                ]
                if beta_flag not in existing:
                    existing.append(beta_flag)
                    extra_headers["anthropic-beta"] = ",".join(existing)
            else:
                extra_headers["anthropic-beta"] = beta_flag
            kwargs["extra_headers"] = extra_headers

    if messages is None:
        messages = [{"role": "user", "content": user_message}]
        if system_message is not None:
            messages.insert(0, {"role": "system", "content": system_message})

    retry_delays = [0, 30, 60, 120]
    max_retries = len(retry_delays)

    for attempt in range(max_retries):
        try:
            if model is None:
                response = completion(
                    model=llm_model,
                    temperature=temperature,
                    messages=messages,
                    **kwargs,
                )
                answer = response.choices[0].message.content.strip()
                usage = response.usage
            else:
                client = instructor.from_litellm(
                    completion, mode=instructor.Mode.TOOLS_STRICT
                )
                response, completion_obj = (
                    client.chat.completions.create_with_completion(
                        model=llm_model,
                        temperature=temperature,
                        messages=messages,
                        response_model=model,
                        **kwargs,
                    )
                )
                answer = response
                usage = completion_obj.usage
            break
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < max_retries - 1:
                delay = retry_delays[attempt + 1]
                if delay > 0:
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed:\n{traceback.format_exc()}")
                raise e

    _log_usage(usage, llm_model, process_id, verbose)

    return answer
