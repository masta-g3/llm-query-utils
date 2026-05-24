import os

import pytest
from pydantic import BaseModel

from llm_query_utils import run_query


RUN_LIVE_CODEX_TESTS = os.environ.get("RUN_LIVE_CODEX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_LIVE_CODEX_TESTS,
    reason="Set RUN_LIVE_CODEX_TESTS=1 to run live Codex regression tests.",
)


class StructuredReply(BaseModel):
    status: str
    count: int


def test_live_codex_plain_text_regression():
    result = run_query(
        user_message="Return exactly this token and nothing else: CODEX_PLAIN_OK",
        llm_model="gpt-5.3-codex",
        use_codex_sdk=True,
        use_agent_sdk=False,
    )

    assert result.strip() == "CODEX_PLAIN_OK"


def test_live_codex_structured_output_regression():
    result = run_query(
        user_message="Return status='ok' and count=7.",
        model=StructuredReply,
        llm_model="gpt-5.3-codex",
        use_codex_sdk=True,
        use_agent_sdk=False,
    )

    assert isinstance(result, StructuredReply)
    assert result.status == "ok"
    assert result.count == 7
