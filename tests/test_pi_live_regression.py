import os

import pytest
from pydantic import BaseModel

from llm_query_utils import run_query


RUN_LIVE_PI_TESTS = os.environ.get("RUN_LIVE_PI_TESTS") == "1"
PI_LIVE_MODEL = os.environ.get("PI_LIVE_MODEL", "anthropic/claude-sonnet-4-5")
pytestmark = pytest.mark.skipif(
    not RUN_LIVE_PI_TESTS,
    reason="Set RUN_LIVE_PI_TESTS=1 to run live Pi CLI regression tests.",
)


class StructuredReply(BaseModel):
    status: str
    count: int


def test_live_pi_plain_text_regression():
    result = run_query(
        user_message="Return exactly this token and nothing else: PI_PLAIN_OK",
        llm_model=PI_LIVE_MODEL,
        use_pi_sdk=True,
        use_agent_sdk=False,
        pi_options={"timeout": 120},
    )

    assert result.strip() == "PI_PLAIN_OK"


def test_live_pi_structured_output_regression():
    result = run_query(
        user_message="Return status='ok' and count=7.",
        model=StructuredReply,
        llm_model=PI_LIVE_MODEL,
        use_pi_sdk=True,
        use_agent_sdk=False,
        pi_options={"timeout": 120},
    )

    assert isinstance(result, StructuredReply)
    assert result.status == "ok"
    assert result.count == 7
