import os

import pytest
from pydantic import BaseModel

from llm_query_utils import run_query
from llm_query_utils.config import DEFAULT_MODEL


RUN_LIVE_AGENT_TESTS = os.environ.get("RUN_LIVE_AGENT_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_LIVE_AGENT_TESTS,
    reason="Set RUN_LIVE_AGENT_TESTS=1 to run live Agent SDK regression tests.",
)


class StructuredReply(BaseModel):
    status: str
    count: int


def test_live_agent_plain_text_regression():
    result = run_query(
        user_message="Return exactly this token and nothing else: AGENT_PLAIN_OK",
        llm_model=DEFAULT_MODEL,
        use_agent_sdk=True,
        use_codex_sdk=False,
    )

    assert result.strip() == "AGENT_PLAIN_OK"


def test_live_agent_structured_output_regression():
    result = run_query(
        user_message="Return status='ok' and count=7.",
        model=StructuredReply,
        llm_model=DEFAULT_MODEL,
        use_agent_sdk=True,
        use_codex_sdk=False,
    )

    assert isinstance(result, StructuredReply)
    assert result.status == "ok"
    assert result.count == 7
