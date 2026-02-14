from .config import DEFAULT_MODEL


def add_cache_control(
    messages: list[dict], cache_message_index: int = 0, llm_model: str = DEFAULT_MODEL
) -> list[dict]:
    """Add cache control to messages for prompt caching support."""
    if not any(provider in llm_model.lower() for provider in ["claude", "anthropic"]):
        return messages

    if cache_message_index >= len(messages) or cache_message_index < 0:
        return messages

    cached_messages = [msg.copy() for msg in messages]
    cached_messages[cache_message_index]["cache_control"] = {"type": "ephemeral"}

    return cached_messages
