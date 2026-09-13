"""PostgreSQL persistence for shared LLM usage events."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .usage import UsageData

USAGE_COLUMNS = (
    "id",
    "tstp",
    "model_name",
    "process_id",
    "session_id",
    "prompt_tokens",
    "completion_tokens",
    "prompt_cost",
    "completion_cost",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_creation_cost",
    "cache_read_cost",
    "project_id",
    "reasoning_output_tokens",
    "total_cost",
)


def insert_usage_rows(connection: Any, rows: Iterable[tuple[Any, ...]]) -> None:
    """Insert rows in USAGE_COLUMNS order without committing the caller's connection."""
    values_rows = list(rows)
    if not values_rows:
        return
    if any(len(row) != len(USAGE_COLUMNS) for row in values_rows):
        raise ValueError(f"usage rows must contain {len(USAGE_COLUMNS)} values")

    row_placeholders = "(" + ", ".join(["%s"] * len(USAGE_COLUMNS)) + ")"
    sql = (
        f"INSERT INTO papers.token_usage_logs ({', '.join(USAGE_COLUMNS)}) VALUES "
        + ", ".join([row_placeholders] * len(values_rows))
        + " ON CONFLICT (id) DO NOTHING"
    )
    values = tuple(value for row in values_rows for value in row)
    with connection.cursor() as cursor:
        cursor.execute(sql, values)


def postgres_usage_callback(
    project_id: str,
    connect: Callable[[], Any],
) -> Callable[[UsageData], None]:
    """Create a callback that owns one fresh connection and transaction per event."""
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id must be a non-empty string")
    if not callable(connect):
        raise ValueError("connect must be callable")

    def callback(usage: UsageData) -> None:
        if not usage.process_id:
            return
        row = (
            str(uuid.uuid4()),
            datetime.now(timezone.utc).replace(tzinfo=None),
            usage.model,
            usage.process_id,
            None,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.prompt_cost,
            usage.completion_cost,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_cost,
            usage.cache_read_cost,
            project_id,
            usage.reasoning_output_tokens,
            usage.total_cost,
        )
        connection = connect()
        try:
            insert_usage_rows(connection, [row])
            connection.commit()
        finally:
            connection.close()

    return callback
