import pandas as pd

import credit_risk.sql_analysis as sql_analysis
from credit_risk.data import MONTHS
from credit_risk.sql_analysis import parse_named_queries


def test_parse_named_queries() -> None:
    queries = parse_named_queries("-- name: first\nSELECT 1;\n\n-- name: second\nSELECT 2;\n")
    assert queries == {"first": "SELECT 1", "second": "SELECT 2"}


def test_actual_sql_queries_run_on_sqlite_and_export_correct_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(sql_analysis, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(sql_analysis, "SQL_REPORTS_DIR", tmp_path)
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "DEFAULT": [0, 1, 0, 1],
            "LIMIT_BAL": [50_000, 150_000, 250_000, 600_000],
            "AGE": [25, 35, 45, 65],
            **{f"PAY_{month}": [0, 1, -1, 2] for month in MONTHS},
        }
    )
    outputs = sql_analysis.run_sql_analysis(frame)
    summary = outputs["portfolio_summary"].iloc[0]
    assert summary["customers"] == 4
    assert summary["defaults"] == 2
    assert summary["default_rate_pct"] == 50.0
    assert len(outputs) == 5
    for name, output in outputs.items():
        assert not output.empty
        assert (tmp_path / f"{name}.csv").exists()
