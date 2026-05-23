from types import SimpleNamespace

import llm_query_utils.query as query_module


def _fake_completion(*args, **kwargs):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="litellm-path"))],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def test_routes_to_codex_when_enabled(monkeypatch):
    async def fake_codex(*args, **kwargs):
        return "codex-path"

    monkeypatch.setattr(query_module, "_codex_sdk_query", fake_codex)

    result = query_module.run_query(
        user_message="hello",
        llm_model="gpt-5.3-codex",
        use_codex_sdk=True,
        use_agent_sdk=False,
    )

    assert result == "codex-path"


def test_routes_any_gpt5_model_to_codex_when_enabled(monkeypatch):
    async def fake_codex(*args, **kwargs):
        return "codex-path"

    monkeypatch.setattr(query_module, "_codex_sdk_query", fake_codex)

    result = query_module.run_query(
        user_message="hello",
        llm_model="gpt-5-mini",
        use_codex_sdk=True,
        use_agent_sdk=False,
    )

    assert result == "codex-path"


def test_routes_prefixed_gpt5_model_to_codex_when_enabled(monkeypatch):
    async def fake_codex(*args, **kwargs):
        return "codex-path"

    monkeypatch.setattr(query_module, "_codex_sdk_query", fake_codex)

    result = query_module.run_query(
        user_message="hello",
        llm_model="chatgpt/gpt-5.4",
        use_codex_sdk=True,
        use_agent_sdk=False,
    )

    assert result == "codex-path"


def test_routes_to_codex_with_legacy_model_alias(monkeypatch):
    async def fake_codex(*args, **kwargs):
        return "codex-path"

    monkeypatch.setattr(query_module, "_codex_sdk_query", fake_codex)

    result = query_module.run_query(
        user_message="hello",
        llm_model="gpt-5-codex",
        use_codex_sdk=True,
        use_agent_sdk=False,
    )

    assert result == "codex-path"


def test_routes_to_agent_sdk_for_claude(monkeypatch):
    async def fake_agent(*args, **kwargs):
        return "agent-path"

    monkeypatch.setattr(query_module, "_agent_sdk_query", fake_agent)

    result = query_module.run_query(
        user_message="hello",
        llm_model="claude-3-5-sonnet",
        use_codex_sdk=False,
        use_agent_sdk=True,
    )

    assert result == "agent-path"


def test_routes_to_pi_when_enabled(monkeypatch):
    async def fake_pi(*args, **kwargs):
        return "pi-path"

    monkeypatch.setattr(query_module, "_pi_sdk_query", fake_pi)

    result = query_module.run_query(
        user_message="hello",
        llm_model="anthropic/claude-sonnet-4-5",
        use_pi_sdk=True,
    )

    assert result == "pi-path"


def test_pi_route_wins_over_claude_agent_when_enabled(monkeypatch):
    async def fake_pi(*args, **kwargs):
        return "pi-path"

    async def fake_agent(*args, **kwargs):
        return "agent-path"

    monkeypatch.setattr(query_module, "_pi_sdk_query", fake_pi)
    monkeypatch.setattr(query_module, "_agent_sdk_query", fake_agent)

    result = query_module.run_query(
        user_message="hello",
        llm_model="claude-3-5-sonnet",
        use_pi_sdk=True,
        use_agent_sdk=True,
    )

    assert result == "pi-path"


def test_pi_route_rejects_messages():
    try:
        query_module.run_query(
            messages=[{"role": "user", "content": "hello"}],
            llm_model="claude-3-5-sonnet",
            use_pi_sdk=True,
        )
    except ValueError as exc:
        assert "messages is not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for messages + use_pi_sdk")


def test_pi_route_rejects_sdk_tools():
    try:
        query_module.run_query(
            user_message="hello",
            llm_model="claude-3-5-sonnet",
            use_pi_sdk=True,
            sdk_allowed_tools=["Read"],
        )
    except ValueError as exc:
        assert "sdk_allowed_tools is not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for sdk_allowed_tools + use_pi_sdk")


def test_pi_route_rejects_extended_thinking_options():
    try:
        query_module.run_query(
            user_message="hello",
            llm_model="claude-3-5-sonnet",
            use_pi_sdk=True,
            enable_extended_thinking=True,
            pi_options={"thinking": "high"},
        )
    except ValueError as exc:
        assert "extended thinking options are not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for extended thinking + use_pi_sdk")


def test_routes_to_litellm_when_codex_disabled(monkeypatch):
    monkeypatch.setattr(query_module, "completion", _fake_completion)
    monkeypatch.setattr(query_module, "_log_usage", lambda *args, **kwargs: None)

    result = query_module.run_query(
        user_message="hello",
        llm_model="gpt-5.3-codex",
        use_codex_sdk=False,
        use_agent_sdk=False,
    )

    assert result == "litellm-path"


def test_chatgpt_provider_routes_to_litellm(monkeypatch):
    captured = {}

    def fake_completion(*args, **kwargs):
        captured["model"] = kwargs["model"]
        return _fake_completion()

    monkeypatch.setattr(query_module, "completion", fake_completion)
    monkeypatch.setattr(query_module, "_log_usage", lambda *args, **kwargs: None)

    result = query_module.run_query(
        user_message="hello",
        llm_model="chatgpt/gpt-5.4",
        use_codex_sdk=False,
    )

    assert result == "litellm-path"
    assert captured["model"] == "chatgpt/gpt-5.4"


def test_codex_route_rejects_sdk_tools():
    try:
        query_module.run_query(
            user_message="hello",
            llm_model="gpt-5.3-codex",
            use_codex_sdk=True,
            use_agent_sdk=False,
            sdk_allowed_tools=["Read"],
        )
    except ValueError as exc:
        assert "sdk_allowed_tools is not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for sdk_allowed_tools + use_codex_sdk")


def test_retries_litellm_transient_failures_with_backoff(monkeypatch):
    call_count = {"value": 0}
    sleeps = []

    def flaky_completion(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] < 3:
            raise RuntimeError("transient failure")
        return _fake_completion()

    monkeypatch.setattr(query_module, "completion", flaky_completion)
    monkeypatch.setattr(query_module, "_log_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(query_module.time, "sleep", lambda delay: sleeps.append(delay))

    result = query_module.run_query(
        user_message="hello",
        llm_model="claude-3-5-sonnet",
        use_agent_sdk=False,
    )

    assert result == "litellm-path"
    assert call_count["value"] == 3
    assert sleeps == [30, 60]


def test_extended_thinking_injects_payload_and_merges_beta_header(monkeypatch):
    captured = {}

    def fake_completion(*args, **kwargs):
        captured["thinking"] = kwargs.get("thinking")
        captured["extra_headers"] = kwargs.get("extra_headers")
        return _fake_completion()

    monkeypatch.setattr(query_module, "completion", fake_completion)
    monkeypatch.setattr(query_module, "_log_usage", lambda *args, **kwargs: None)

    query_module.run_query(
        user_message="hello",
        llm_model="claude-3-5-sonnet",
        use_agent_sdk=False,
        enable_extended_thinking=True,
        extended_thinking_budget_tokens=2048,
        thinking_options={"focus": "high"},
        extended_thinking_beta="beta-1",
        extra_headers={"anthropic-beta": "beta-0"},
    )

    assert captured["thinking"]["type"] == "enabled"
    assert captured["thinking"]["budget_tokens"] == 2048
    assert captured["thinking"]["focus"] == "high"
    assert captured["extra_headers"]["anthropic-beta"] == "beta-0,beta-1"
