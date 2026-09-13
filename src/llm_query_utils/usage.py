import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class UsageData:
    model: str
    process_id: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    prompt_cost: Optional[float]
    completion_cost: Optional[float]
    cache_creation_input_tokens: Optional[int] = 0
    cache_read_input_tokens: Optional[int] = 0
    cache_creation_cost: Optional[float] = None
    cache_read_cost: Optional[float] = None
    reasoning_output_tokens: Optional[int] = None
    total_cost: Optional[float] = None


def valid_cost(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or not math.isfinite(value):
        return None
    return value


def pi_total_cost(usage: dict) -> Optional[float]:
    """Use the reported total, or derive one only from complete component data."""
    cost = usage.get("cost") or {}
    reported = valid_cost(cost.get("total"))
    if reported is not None:
        return reported
    total = 0.0
    for key in ("input", "output", "cacheWrite", "cacheRead"):
        component = valid_cost(cost.get(key))
        if component is not None:
            total += component
        elif usage.get(key) != 0:
            return None
    return valid_cost(total)


_usage_callback: Optional[Callable[[UsageData], None]] = None


def set_usage_callback(callback: Callable[[UsageData], None]) -> None:
    global _usage_callback
    _usage_callback = callback


def fire_usage_callback(data: UsageData) -> None:
    if _usage_callback is not None:
        _usage_callback(data)
