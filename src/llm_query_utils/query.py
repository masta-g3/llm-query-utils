import logging
import os
import asyncio
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
    sdk_allowed_tools: Optional[list[str]] = None,
    **kwargs,
) -> Union[BaseModel, str]:
    """Run an LLM query, routing through Agent SDK (Claude) or LiteLLM/Instructor (others).

    Args:
        use_agent_sdk: Route query through Agent SDK instead of direct API (default: True).
            Automatically disabled for: custom messages, extended thinking, non-Claude models.
        sdk_allowed_tools: Tools the agent can use (e.g., ["Read"] for image analysis).
    """
    if use_agent_sdk:
        if messages is not None:
            use_agent_sdk = False
        elif enable_extended_thinking or thinking_options:
            use_agent_sdk = False
        elif "claude" not in llm_model.lower():
            use_agent_sdk = False

    if use_agent_sdk:
        return asyncio.run(_agent_sdk_query(
            system_message, user_message, llm_model, sdk_allowed_tools,
            output_schema=model,
            process_id=process_id,
        ))

    if messages is None and user_message is None:
        raise ValueError(
            "Either 'messages' parameter or 'user_message' parameter must be provided"
        )

    temperature = (
        None if any(x in llm_model for x in ["o1", "r1", "o3", "gpt-5"]) else temperature
    )

    thinking_requested = (
        enable_extended_thinking
        or thinking_options is not None
        or extended_thinking_budget_tokens is not None
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
