# codex-002 Agent Usage Logging Reliability

## Goal
Ensure Agent SDK queries always emit usage callback events at terminal completion so downstream DB token logging does not silently miss runs.

## Delivered
- Added `_fire_agent_usage(...)` in `query.py` to normalize Agent usage callback emission.
- Updated `_agent_sdk_query(...)` to emit one usage callback whenever a terminal `ResultMessage` is observed.
- Added zero-token fallback behavior when Agent SDK omits usage payload fields.
- Preserved existing behavior for pre-terminal transport errors (no synthetic usage event).

## Validation
- `uv run --with pytest pytest tests/test_query_helpers_and_utils.py -q` -> passed.
- Added coverage for:
  - terminal success with missing usage
  - terminal success with present usage mapping
  - terminal error with missing usage
  - pre-terminal transport failure (no usage event)
  - malformed usage payload fallback

## Files
- `src/llm_query_utils/query.py`
- `tests/test_query_helpers_and_utils.py`
- `docs/STRUCTURE.md`
