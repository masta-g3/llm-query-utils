from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class UsageData:
    model: str
    process_id: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    prompt_cost: float
    completion_cost: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_cost: Optional[float] = None
    cache_read_cost: Optional[float] = None


_usage_callback: Optional[Callable[[UsageData], None]] = None


def set_usage_callback(callback: Callable[[UsageData], None]) -> None:
    global _usage_callback
    _usage_callback = callback


def fire_usage_callback(data: UsageData) -> None:
    if _usage_callback is not None:
        _usage_callback(data)
