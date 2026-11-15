"""Materialize the cleaned data in SQLite and export named aggregate queries."""

from __future__ import annotations

import re
import sqlite3

import pandas as pd

from credit_risk.config import PROCESSED_DIR, SQL_FILE, SQL_REPORTS_DIR, ensure_directories


def parse_named_queries(sql_text: str) -> dict[str, str]:
    pattern = re.compile(
        r"^-- name:\s*([a-z0-9_]+)\s*$\n(.*?)(?=^-- name:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    return {name: query.strip().rstrip(";") for name, query in pattern.findall(sql_text)}


def run_sql_analysis(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ensure_directories()
    database_path = PROCESSED_DIR / "credit_risk.sqlite"
    queries = parse_named_queries(SQL_FILE.read_text(encoding="utf-8"))
    if not queries:
        raise ValueError(f"No named queries found in {SQL_FILE}")

    outputs: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(database_path) as connection:
        frame.to_sql("customer_credit", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_customer_id ON customer_credit(ID)"
        )
        for name, query in queries.items():
            output = pd.read_sql_query(query, connection)
            output.to_csv(SQL_REPORTS_DIR / f"{name}.csv", index=False)
            outputs[name] = output
    return outputs
