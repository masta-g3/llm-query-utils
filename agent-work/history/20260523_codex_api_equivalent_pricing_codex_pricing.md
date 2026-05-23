# Codex API-equivalent pricing

Implemented dynamic API-equivalent cost logging for Codex CLI / ChatGPT-subscription calls in `llm-query-utils`.

## What changed

- `src/llm_query_utils/query.py`
  - Converted Codex usage dicts into the same attribute shape used by existing LiteLLM cost helpers.
  - Reused `_calculate_usage_costs()` so Codex pricing follows LiteLLM's dynamic model pricing table.
  - Normalized provider-prefixed model names for pricing lookup while preserving logged model identity.
  - Used the actual executed Codex model for cost/logging when `codex_options["model"]` overrides `llm_model`.
  - Preserved token logging with zero costs only for expected LiteLLM missing-pricing failures; unexpected pricing errors surface.

- `tests/test_query_helpers_and_utils.py`
  - Added coverage for Codex API-equivalent costs, cache-read token key mapping, provider-prefix normalization, missing-pricing fallback, unexpected pricing errors, and override-model pricing/logging.

- `README.md` and `docs/STRUCTURE.md`
  - Documented that Codex cost fields are LiteLLM API-equivalent estimates, not actual ChatGPT subscription spend.

## Verification

- Baseline before implementation: `uv run pytest tests/test_query_helpers_and_utils.py -q` → 27 passed.
- Targeted after review fixes: `uv run pytest tests/test_query_helpers_and_utils.py -q` → 31 passed.
- Full suite after review fixes: `uv run pytest -q` → 46 passed, 6 skipped.
- Dynamic pricing smoke confirmed `chatgpt/gpt-5.5` normalizes to `gpt-5.5` and returns nonzero LiteLLM costs.

## Notes

This change does not alter routing behavior, DB schemas, consumer callback contracts, or historical rows. Existing usage callbacks continue to receive `UsageData`; Codex rows now include estimated API-equivalent costs when LiteLLM knows the model price.
