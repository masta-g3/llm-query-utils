**Feature:** structured-pi-routing → Centralize structured GPT/Pi routing on `llm-query-utils.run_query`

## Summary

Standardized structured GPT-5/Pydantic calls in `llmpedia_workflows` and `llmpedia_social` on the existing `llm-query-utils.run_query(..., model=..., use_pi_sdk=True)` Pi route. This preserves the bad-`OPENAI_API_KEY` fix by avoiding LiteLLM/OpenAI for active structured GPT paths, while removing duplicate repo-local JSON parsing helpers.

## Implemented

- Added a public-path regression test in `llm-query-utils` proving `run_query(..., model=..., use_pi_sdk=True)` passes the Pydantic schema to `_pi_sdk_query` and returns the validated model.
- Updated `llmpedia_workflows/utils/instruct.py` so GPT-5 calls without `messages`, including structured Pydantic calls, route through Pi with `use_pi_sdk=True`, `use_agent_sdk=False`, and repo `pi_options` policy.
- Removed `run_pi_json_query` and local JSON/fence parsing from `llmpedia_workflows`; active structured helpers now call `run_instructor_query(..., model=po.X, llm_model=...)` and rely on centralized `llm-query-utils` validation.
- Removed local JSON/Pydantic parsing from `llmpedia_social` tweet relevance; `assess_llm_relevance` now calls `run_query(..., model=TweetRelevanceInfo, **query_route_kwargs(...))` using `NANO_MODEL` through Pi/openai-codex.
- Updated routing tests in both app repos, including provider-env isolation in tests after review.
- Updated durable routing docs in `llmpedia_workflows` and `llmpedia_social` to describe structured GPT/Pydantic calls through Pi with `llm-query-utils` validation.

## Validation

- `llm-query-utils`: `uv run pytest tests/test_query_routing.py tests/test_query_helpers_and_utils.py -q` → 47 passed.
- `llmpedia_workflows`: `.venv/bin/python -m pytest tests/test_workflow_model_routing.py tests/test_paper_download_backoff.py tests/test_workflow_error_handling.py tests/test_thumbnail_codex_pi.py tests/test_thumbnail_api.py -q` → 39 passed, 1 skipped.
- `llmpedia_social`: `uv run pytest tests/test_tweet_relevance_direct_api.py tests/test_codex_model_routing.py tests/test_codex_model_config.py -q` → 14 passed.
- Bad-key smoke checks passed for `llmpedia_workflows.verify_llm_paper(...)` and `llmpedia_social.assess_llm_relevance(...)` with `OPENAI_API_KEY=bad-key`.
- Search confirmed no remaining `run_pi_json_query`, `parse_json_response`, or `_strip_json_fence` in active app repo paths.
- Restarted `workflow:llmpedia_workflows_workflow` and `workflow:llmpedia_social_tweet_collect`; both reported active.

## Discovered Work

None.
