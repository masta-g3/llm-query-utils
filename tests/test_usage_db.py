from datetime import datetime
from uuid import UUID

import pytest

from llm_query_utils import UsageData, postgres_usage_callback
from llm_query_utils.usage_db import USAGE_COLUMNS, insert_usage_rows


class Cursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, values):
        self.connection.executions.append((sql, values))


class Connection:
    def __init__(self):
        self.executions = []
        self.commits = 0
        self.closed = False

    def cursor(self):
        return Cursor(self)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def test_usage_columns_are_the_shared_insert_contract():
    assert USAGE_COLUMNS == (
        "id", "tstp", "model_name", "process_id", "session_id",
        "prompt_tokens", "completion_tokens", "prompt_cost", "completion_cost",
        "cache_creation_input_tokens", "cache_read_input_tokens",
        "cache_creation_cost", "cache_read_cost", "project_id",
        "reasoning_output_tokens", "total_cost",
    )


def test_insert_usage_rows_batches_without_owning_transaction():
    connection = Connection()
    row = tuple(range(len(USAGE_COLUMNS)))

    insert_usage_rows(connection, [row, row])

    assert connection.commits == 0 and not connection.closed
    sql, values = connection.executions[0]
    assert "papers.token_usage_logs" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert sql.count("%s") == 2 * len(USAGE_COLUMNS)
    assert values == row + row


def test_postgres_callback_maps_event_and_owns_connection():
    connection = Connection()
    callback = postgres_usage_callback(project_id="sentiment", connect=lambda: connection)

    callback(UsageData(
        model="openai-codex/gpt-5.5",
        process_id="sentiment_analyze",
        prompt_tokens=10,
        completion_tokens=3,
        prompt_cost=0.1,
        completion_cost=0.2,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=4,
        cache_creation_cost=None,
        cache_read_cost=0.01,
        reasoning_output_tokens=2,
        total_cost=0.3,
    ))

    assert connection.commits == 1 and connection.closed
    row = connection.executions[0][1]
    assert isinstance(UUID(row[0]), UUID)
    assert isinstance(row[1], datetime) and row[1].tzinfo is None
    assert row[2:5] == ("openai-codex/gpt-5.5", "sentiment_analyze", None)
    assert row[13:] == ("sentiment", 2, 0.3)


def test_postgres_callback_skips_no_process_without_connecting():
    calls = []
    callback = postgres_usage_callback(project_id="sentiment", connect=lambda: calls.append(1))
    callback(UsageData("model", None, 1, 2, 0.1, 0.2))
    assert calls == []


@pytest.mark.parametrize("project_id", ["", "   ", None])
def test_postgres_callback_requires_explicit_project(project_id):
    with pytest.raises(ValueError, match="project_id"):
        postgres_usage_callback(project_id=project_id, connect=lambda: Connection())
