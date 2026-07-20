import asyncio
from typing import Optional
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import llm_query_utils.query as query_module
from llm_query_utils.cache import add_cache_control
from llm_query_utils.vision import format_vision_messages
from llm_query_utils.usage import set_usage_callback


class OutputSchema(BaseModel):
    status: str
    count: int


class OptionalOutputSchema(BaseModel):
    required_value: str
    optional_value: Optional[str] = None


def test_codex_strict_schema_requires_all_object_properties():
    schema = query_module._to_codex_strict_schema(OptionalOutputSchema.model_json_schema())

    assert schema["required"] == ["required_value", "optional_value"]
    assert schema["additionalProperties"] is False


def test_agent_sdk_query_raises_on_error_and_empty_response(monkeypatch):
    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def error_stream(*args, **kwargs):
        yield FakeResultMessage(is_error=True, result="bad request")

    monkeypatch.setattr(query_module, "agent_sdk_query", error_stream)

    with pytest.raises(RuntimeError, match="Agent SDK error"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )

    async def empty_stream(*args, **kwargs):
        yield FakeResultMessage(is_error=False, result=None)

    monkeypatch.setattr(query_module, "agent_sdk_query", empty_stream)

    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )


def test_agent_sdk_query_emits_zero_usage_on_terminal_success_without_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def success_stream(*args, **kwargs):
        yield FakeAssistantMessage([FakeTextBlock("agent-ok")])
        yield FakeResultMessage(usage=None, total_cost_usd=0.42, is_error=False)

    monkeypatch.setattr(query_module, "agent_sdk_query", success_stream)

    result = asyncio.run(
        query_module._agent_sdk_query(
            system_message="system",
            user_message="hello",
            llm_model="claude-3-5-sonnet",
            process_id="p-1",
        )
    )

    assert result == "agent-ok"
    assert len(usage_events) == 1
    assert usage_events[0].process_id == "p-1"
    assert usage_events[0].prompt_tokens == 0
    assert usage_events[0].completion_tokens == 0
    assert usage_events[0].prompt_cost == 0.42
    assert usage_events[0].completion_cost == 0


def test_agent_sdk_query_emits_usage_with_present_payload(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def success_stream(*args, **kwargs):
        yield FakeAssistantMessage([FakeTextBlock("agent-ok")])
        yield FakeResultMessage(
            usage={
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 1,
            },
            total_cost_usd=0.77,
            is_error=False,
        )

    monkeypatch.setattr(query_module, "agent_sdk_query", success_stream)

    result = asyncio.run(
        query_module._agent_sdk_query(
            system_message="system",
            user_message="hello",
            llm_model="claude-3-5-sonnet",
            process_id="p-usage",
        )
    )

    assert result == "agent-ok"
    assert len(usage_events) == 1
    assert usage_events[0].process_id == "p-usage"
    assert usage_events[0].prompt_tokens == 12
    assert usage_events[0].completion_tokens == 5
    assert usage_events[0].cache_creation_input_tokens == 3
    assert usage_events[0].cache_read_input_tokens == 1
    assert usage_events[0].prompt_cost == 0.77
    assert usage_events[0].completion_cost == 0


def test_agent_sdk_query_emits_usage_before_terminal_error(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def error_stream(*args, **kwargs):
        yield FakeResultMessage(usage=None, total_cost_usd=0.0, is_error=True, result="bad request")

    monkeypatch.setattr(query_module, "agent_sdk_query", error_stream)

    with pytest.raises(RuntimeError, match="Agent SDK error"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )

    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 0
    assert usage_events[0].completion_tokens == 0


def test_agent_sdk_query_transport_error_before_terminal_emits_no_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def broken_stream(*args, **kwargs):
        raise RuntimeError("stream failure")
        yield  # pragma: no cover

    monkeypatch.setattr(query_module, "agent_sdk_query", broken_stream)

    with pytest.raises(RuntimeError, match="stream failure"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )

    assert usage_events == []


def test_fire_agent_usage_handles_malformed_payload():
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    query_module._fire_agent_usage(
        llm_model="claude-3-5-sonnet",
        process_id="p-malformed",
        usage_data="not-a-dict",
        total_cost_usd=None,
    )

    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 0
    assert usage_events[0].completion_tokens == 0
    assert usage_events[0].cache_creation_input_tokens == 0
    assert usage_events[0].cache_read_input_tokens == 0
    assert usage_events[0].prompt_cost == 0


def test_parse_codex_event_stream_extracts_usage_and_error():
    event_stream = "\n".join(
        [
            "noise",
            '{"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":3}}',
            '{"type":"turn.failed","error":{"message":"turn failed"}}',
            '{"type":"error","message":"runtime failure"}',
        ]
    )

    usage, error = query_module._parse_codex_event_stream(event_stream)

    assert usage == {"input_tokens": 11, "output_tokens": 3}
    assert error == "runtime failure"


def test_fire_codex_usage_accepts_both_cache_read_token_keys(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))
    monkeypatch.setattr(
        query_module,
        "_calculate_codex_usage_costs",
        lambda usage, llm_model: (0.1, 0.2, None, 0.3),
    )

    query_module._fire_codex_usage(
        {"input_tokens": 2, "output_tokens": 1, "cache_read_input_tokens": 9},
        llm_model="gpt-5-codex",
        process_id="p-1",
    )
    query_module._fire_codex_usage(
        {"input_tokens": 2, "output_tokens": 1, "cached_input_tokens": 7},
        llm_model="gpt-5-codex",
        process_id="p-2",
    )

    assert usage_events[0].cache_read_input_tokens == 9
    assert usage_events[1].cache_read_input_tokens == 7
    assert usage_events[0].prompt_cost == 0.1
    assert usage_events[0].completion_cost == 0.2
    assert usage_events[0].cache_read_cost == 0.3


def test_fire_codex_usage_emits_api_equivalent_costs(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))
    monkeypatch.setattr(
        query_module,
        "_calculate_usage_costs",
        lambda usage, llm_model: (1.25, 2.5, None, 0.75),
    )

    query_module._fire_codex_usage(
        {"input_tokens": 1000, "output_tokens": 200, "cached_input_tokens": 300},
        llm_model="gpt-5.5",
        process_id="p-1",
    )

    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 1000
    assert usage_events[0].completion_tokens == 200
    assert usage_events[0].cache_read_input_tokens == 300
    assert usage_events[0].prompt_cost == 1.25
    assert usage_events[0].completion_cost == 2.5
    assert usage_events[0].cache_read_cost == 0.75


def test_fire_codex_usage_uses_model_basename_for_pricing(monkeypatch):
    priced_models = []
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    def fake_calculate_usage_costs(usage, llm_model):
        priced_models.append(llm_model)
        return 1.0, 2.0, None, None

    monkeypatch.setattr(query_module, "_calculate_usage_costs", fake_calculate_usage_costs)

    query_module._fire_codex_usage(
        {"input_tokens": 10, "output_tokens": 5},
        llm_model="chatgpt/gpt-5.5",
        process_id="p-1",
    )

    assert priced_models == ["gpt-5.5"]
    assert usage_events[0].model == "chatgpt/gpt-5.5"


def test_fire_codex_usage_keeps_tokens_when_pricing_missing(monkeypatch, caplog):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    def raise_unknown_model(*args, **kwargs):
        raise Exception("This model isn't mapped yet")

    monkeypatch.setattr(query_module, "_calculate_usage_costs", raise_unknown_model)

    with caplog.at_level("WARNING"):
        query_module._fire_codex_usage(
            {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 3},
            llm_model="gpt-5-new",
            process_id="p-1",
        )

    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 10
    assert usage_events[0].completion_tokens == 5
    assert usage_events[0].cache_read_input_tokens == 3
    assert usage_events[0].prompt_cost == 0
    assert usage_events[0].completion_cost == 0
    assert "pricing unavailable" in caplog.text


def test_fire_codex_usage_raises_unexpected_pricing_errors(monkeypatch):
    set_usage_callback(lambda data: None)

    def raise_bug(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(query_module, "_calculate_usage_costs", raise_bug)

    with pytest.raises(RuntimeError, match="bug"):
        query_module._fire_codex_usage(
            {"input_tokens": 10, "output_tokens": 5},
            llm_model="gpt-5.5",
            process_id="p-1",
        )


def test_add_cache_control_only_for_supported_models_and_valid_index():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "prompt"},
    ]

    updated = add_cache_control(messages, cache_message_index=1, llm_model="claude-3-5-sonnet")
    assert updated[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[1]

    invalid_index = add_cache_control(messages, cache_message_index=9, llm_model="claude-3-5-sonnet")
    assert invalid_index == messages

    unsupported_model = add_cache_control(messages, cache_message_index=0, llm_model="gpt-5-mini")
    assert unsupported_model == messages


def test_format_vision_messages_applies_provider_order_and_url_formatting():
    images = ["BASE64_IMAGE", "https://example.com/image.png"]

    claude_messages = format_vision_messages(
        images=images,
        text="describe this",
        model="claude-3-5-sonnet",
        system_message="vision rules",
    )
    claude_content = claude_messages[1]["content"]

    assert claude_messages[0]["role"] == "system"
    assert claude_content[0]["type"] == "image_url"
    assert claude_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert claude_content[1]["image_url"]["url"] == "https://example.com/image.png"
    assert claude_content[-1] == {"type": "text", "text": "describe this"}

    other_messages = format_vision_messages(
        images=images,
        text="describe this",
        model="gpt-5-mini",
    )
    other_content = other_messages[0]["content"]

    assert other_content[0] == {"type": "text", "text": "describe this"}
    assert other_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert other_content[2]["image_url"]["url"] == "https://example.com/image.png"


def test_log_usage_emits_costs_for_gpt5_codex(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))
    monkeypatch.setattr(query_module, "cost_per_token", lambda **kwargs: (1.25, 2.5))
    monkeypatch.setattr(
        query_module,
        "get_model_info",
        lambda **kwargs: {
            "input_cost_per_token": 0.5,
            "cache_creation_input_token_cost": None,
            "cache_read_input_token_cost": 0.1,
        },
    )

    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    query_module._log_usage(usage, llm_model="gpt-5-codex")

    assert len(usage_events) == 1
    assert usage_events[0].prompt_cost > 0
    assert usage_events[0].completion_cost > 0


def test_calculate_usage_costs_falls_back_for_cache_creation_rate(monkeypatch):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        cache_creation_input_tokens=4,
        cache_read_input_tokens=3,
    )

    monkeypatch.setattr(query_module, "cost_per_token", lambda **kwargs: (1.25, 2.5))
    monkeypatch.setattr(
        query_module,
        "get_model_info",
        lambda **kwargs: {
            "input_cost_per_token": 0.5,
            "cache_creation_input_token_cost": None,
            "cache_read_input_token_cost": 0.1,
        },
    )

    prompt_cost, completion_cost, cache_creation_cost, cache_read_cost = (
        query_module._calculate_usage_costs(usage, "gpt-5-codex")
    )

    assert prompt_cost == 1.25
    assert completion_cost == 2.5
    assert cache_creation_cost == 4.0
    assert cache_read_cost == 0.30000000000000004


def test_calculate_usage_costs_matches_litellm_totals_for_supported_model():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    expected_prompt_cost, expected_completion_cost = query_module.cost_per_token(
        model="gpt-4o-mini",
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
    )

    prompt_cost, completion_cost, _, _ = query_module._calculate_usage_costs(
        usage, "gpt-4o-mini"
    )

    assert prompt_cost == expected_prompt_cost
    assert completion_cost == expected_completion_cost


def test_codex_query_parses_structured_output_and_fires_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    monkeypatch.setattr(query_module, "_run_codex_login_preflight", lambda: None)
    monkeypatch.setattr(
        query_module,
        "_run_codex_exec",
        lambda model, prompt, output_schema, extra_args, timeout=None: (
            '{"status":"ok","count":7}',
            {"input_tokens": 12, "output_tokens": 4, "cached_input_tokens": 2},
            "",
        ),
    )

    result = asyncio.run(
        query_module._codex_sdk_query(
            system_message="be precise",
            user_message="return json",
            llm_model="gpt-5-codex",
            output_schema=OutputSchema,
            process_id="p-1",
        )
    )

    assert isinstance(result, OutputSchema)
    assert result.count == 7
    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 12
    assert usage_events[0].completion_tokens == 4
    assert usage_events[0].cache_read_input_tokens == 2


def test_codex_query_returns_text(monkeypatch):
    monkeypatch.setattr(query_module, "_run_codex_login_preflight", lambda: None)
    monkeypatch.setattr(
        query_module,
        "_run_codex_exec",
        lambda model, prompt, output_schema, extra_args, timeout=None: (
            "plain-text",
            {"input_tokens": 3, "output_tokens": 1},
            "",
        ),
    )

    result = asyncio.run(
        query_module._codex_sdk_query(
            system_message=None,
            user_message="hello",
            llm_model="gpt-5-codex",
            output_schema=None,
        )
    )

    assert result == "plain-text"


def test_codex_options_extra_args_must_be_list():
    with pytest.raises(ValueError, match="codex_options.extra_args must be a list"):
        asyncio.run(
            query_module._codex_sdk_query(
                system_message=None,
                user_message="hello",
                llm_model="gpt-5-codex",
                codex_options={"extra_args": "--bad"},
            )
        )


def test_codex_options_accepts_any_gpt5_model(monkeypatch):
    monkeypatch.setattr(query_module, "_run_codex_login_preflight", lambda: None)
    captured = {}
    priced_models = []
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    def fake_exec(model, prompt, output_schema, extra_args, timeout=None):
        captured["model"] = model
        return "ok", {"input_tokens": 10, "output_tokens": 5}, ""

    def fake_calculate_usage_costs(usage, llm_model):
        priced_models.append(llm_model)
        return 1.0, 2.0, None, None

    monkeypatch.setattr(query_module, "_run_codex_exec", fake_exec)
    monkeypatch.setattr(query_module, "_calculate_usage_costs", fake_calculate_usage_costs)

    result = asyncio.run(
        query_module._codex_sdk_query(
            system_message=None,
            user_message="hello",
            llm_model="gpt-5-codex",
            codex_options={"model": "gpt-5.4"},
        )
    )

    assert result == "ok"
    assert captured["model"] == "gpt-5.4"
    assert priced_models == ["gpt-5.4"]
    assert usage_events[0].model == "gpt-5.4"


def test_codex_options_model_must_be_supported():
    with pytest.raises(ValueError, match="Unsupported Codex model"):
        asyncio.run(
            query_module._codex_sdk_query(
                system_message=None,
                user_message="hello",
                llm_model="gpt-5-codex",
                codex_options={"model": "codex-mini-latest"},
            )
        )


def test_parse_pi_event_stream_extracts_text_and_usage():
    event_stream = "\n".join([
        '{"type":"session","version":3,"id":"s1"}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"PI_OK"}],"usage":{"input":10,"output":3,"cacheRead":2,"cacheWrite":1,"cost":{"input":0.01,"output":0.02,"cacheRead":0.001,"cacheWrite":0.002,"total":0.033}}}}',
        '{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"WRONG"}],"usage":{"input":99,"output":99}}]}',
    ])

    text, usage, error = query_module._parse_pi_event_stream(event_stream)

    assert text == "PI_OK"
    assert usage["input"] == 10
    assert usage["output"] == 3
    assert error is None


def test_parse_pi_event_stream_falls_back_to_agent_end():
    event_stream = "\n".join([
        '{"type":"session","version":3,"id":"s1"}',
        '{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"fallback"}],"usage":{"input":4,"output":1}}]}',
    ])

    text, usage, error = query_module._parse_pi_event_stream(event_stream)

    assert text == "fallback"
    assert usage == {"input": 4, "output": 1}
    assert error is None


def test_fire_pi_usage_maps_tokens_costs_and_cache():
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    query_module._fire_pi_usage(
        {
            "input": 10,
            "output": 3,
            "cacheRead": 2,
            "cacheWrite": 1,
            "cost": {
                "input": 0.01,
                "output": 0.02,
                "cacheRead": 0.001,
                "cacheWrite": 0.002,
            },
        },
        llm_model="anthropic/claude-sonnet-4-5",
        process_id="p-pi",
    )

    assert len(usage_events) == 1
    assert usage_events[0].model == "anthropic/claude-sonnet-4-5"
    assert usage_events[0].process_id == "p-pi"
    assert usage_events[0].prompt_tokens == 10
    assert usage_events[0].completion_tokens == 3
    assert usage_events[0].cache_read_input_tokens == 2
    assert usage_events[0].cache_creation_input_tokens == 1
    assert usage_events[0].prompt_cost == 0.01
    assert usage_events[0].completion_cost == 0.02
    assert usage_events[0].cache_read_cost == 0.001
    assert usage_events[0].cache_creation_cost == 0.002


def test_build_pi_command_defaults_to_safe_query_mode():
    cmd, timeout, cwd = query_module._build_pi_command(
        llm_model="anthropic/claude-sonnet-4-5",
        system_message="system rules",
        user_message="hello",
        pi_options={"timeout": 90, "cwd": "/tmp"},
    )

    assert cmd == [
        "pi",
        "--mode",
        "json",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "--no-tools",
        "--model",
        "anthropic/claude-sonnet-4-5",
        "--system-prompt",
        "system rules",
        "hello",
    ]
    assert timeout == 90
    assert cwd == "/tmp"


def test_pi_options_validation_rejects_bad_types():
    with pytest.raises(ValueError, match="pi_options must be a dictionary"):
        query_module._build_pi_command(
            llm_model="claude",
            system_message=None,
            user_message="hello",
            pi_options=[],
        )

    with pytest.raises(ValueError, match="pi_options.tools"):
        query_module._build_pi_command(
            llm_model="claude",
            system_message=None,
            user_message="hello",
            pi_options={"tools": "read"},
        )

    with pytest.raises(ValueError, match="pi_options.timeout"):
        query_module._build_pi_command(
            llm_model="claude",
            system_message=None,
            user_message="hello",
            pi_options={"timeout": "30"},
        )


def test_pi_options_validation_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unsupported pi_options"):
        query_module._build_pi_command(
            llm_model="claude",
            system_message=None,
            user_message="hello",
            pi_options={"extra_args": ["--verbose"]},
        )


def test_run_query_routes_structured_pi_public_path(monkeypatch):
    async def fake_pi(system_message, user_message, llm_model, output_schema=None, process_id=None, pi_options=None):
        assert system_message == "system"
        assert user_message == "user"
        assert llm_model == "gpt-5.4-mini"
        assert output_schema is OutputSchema
        assert process_id == "p-public"
        assert pi_options == {"provider": "openai-codex"}
        return OutputSchema(status="ok", count=7)

    monkeypatch.setattr(query_module, "_pi_sdk_query", fake_pi)

    result = query_module.run_query(
        "system",
        "user",
        model=OutputSchema,
        llm_model="gpt-5.4-mini",
        process_id="p-public",
        use_pi_sdk=True,
        use_agent_sdk=False,
        pi_options={"provider": "openai-codex"},
    )

    assert result == OutputSchema(status="ok", count=7)


def test_pi_query_parses_structured_output_and_fires_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))
    captured = {}

    def fake_run(model, system_message, user_message, pi_options):
        captured["system_message"] = system_message
        return (
            '{"status":"ok","count":7}',
            {"input": 6, "output": 2, "cost": {"input": 0.1, "output": 0.2}},
            "",
        )

    monkeypatch.setattr(query_module, "_run_pi_cli", fake_run)

    result = asyncio.run(
        query_module._pi_sdk_query(
            system_message="be precise",
            user_message="return json",
            llm_model="anthropic/claude-sonnet-4-5",
            output_schema=OutputSchema,
            process_id="p-pi",
        )
    )

    assert isinstance(result, OutputSchema)
    assert result.count == 7
    assert "Return only valid JSON" in captured["system_message"]
    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 6
    assert usage_events[0].completion_tokens == 2


def test_run_pi_cli_handles_subprocess_failures(monkeypatch):
    def missing_pi(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(query_module.subprocess, "run", missing_pi)
    with pytest.raises(RuntimeError, match="Pi CLI not found"):
        query_module._run_pi_cli("claude", None, "hello")

    def timeout(*args, **kwargs):
        raise query_module.subprocess.TimeoutExpired(cmd=["pi"], timeout=5)

    monkeypatch.setattr(query_module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 5 seconds"):
        query_module._run_pi_cli("claude", None, "hello", {"timeout": 5})

    def failed(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad pi")

    monkeypatch.setattr(query_module.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="Pi CLI error: bad pi"):
        query_module._run_pi_cli("claude", None, "hello")
