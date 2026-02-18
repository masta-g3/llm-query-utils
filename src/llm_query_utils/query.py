import logging
import os
import asyncio
import json
import subprocess
import tempfile
import time
import traceback
import warnings
from typing import Type, Optional, Union

from pydantic import BaseModel
from tokencost import calculate_cost_by_tokens
from litellm import completion
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

CODEX_SUPPORTED_MODELS = {"gpt-5-codex"}


def _log_usage(
    usage, llm_model: str, process_id: str = None, verbose: bool = False
) -> None:
    prompt_cost = calculate_cost_by_tokens(usage.prompt_tokens, llm_model, "input")
    completion_cost = calculate_cost_by_tokens(
        usage.completion_tokens, llm_model, "output"
    )

    cache_creation_cost = None
    cache_read_cost = None
    if (
        hasattr(usage, "cache_creation_input_tokens")
        and usage.cache_creation_input_tokens
    ):
        cache_creation_cost = (
            calculate_cost_by_tokens(
                usage.cache_creation_input_tokens, llm_model, "input"
            )
            * 2
        )
    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
        cache_read_cost = calculate_cost_by_tokens(
            usage.cache_read_input_tokens, llm_model, "cached"
        )

    if verbose:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        total_cost = (
            (prompt_cost or 0)
            + (completion_cost or 0)
            + (cache_creation_cost or 0)
            + (cache_read_cost or 0)
        )
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

    async for message in agent_sdk_query(prompt=user_message, options=options):
        message_count += 1
        if isinstance(message, ResultMessage):
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

    if usage_data:
        fire_usage_callback(UsageData(
            model=llm_model,
            process_id=process_id,
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            prompt_cost=total_cost_usd or 0,
            completion_cost=0,
            cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
        ))

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
    return model in CODEX_SUPPORTED_MODELS


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


def _fire_codex_usage(
    usage: Optional[dict],
    llm_model: str,
    process_id: Optional[str],
) -> None:
    if not usage:
        return

    fire_usage_callback(UsageData(
        model=llm_model,
        process_id=process_id,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        prompt_cost=0,
        completion_cost=0,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cached_input_tokens", 0),
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
            json.dump(output_schema.model_json_schema(), schema_file)
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

    model = options.get("model") or llm_model.split("/")[-1]
    if model not in CODEX_SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported Codex model '{model}'. Supported models: {sorted(CODEX_SUPPORTED_MODELS)}"
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

    _fire_codex_usage(usage, llm_model, process_id)
    logger.info(f"Codex query complete: {len(output_text)} chars")

    return result


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
    sdk_allowed_tools: Optional[list[str]] = None,
    codex_options: Optional[dict] = None,
    **kwargs,
) -> Union[BaseModel, str]:
    """Run an LLM query, routing through Agent SDK (Claude) or LiteLLM/Instructor (others).

    Args:
        use_agent_sdk: Route query through Agent SDK instead of direct API (default: True).
            Automatically disabled for: custom messages, extended thinking, non-Claude models.
        use_codex_sdk: Route query through Codex CLI transport (default: False).
            Supported only for explicitly allowed Codex models and single-turn user_message input.
        sdk_allowed_tools: Tools the agent can use (e.g., ["Read"] for image analysis).
    """
    thinking_requested = (
        enable_extended_thinking
        or thinking_options is not None
        or extended_thinking_budget_tokens is not None
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
