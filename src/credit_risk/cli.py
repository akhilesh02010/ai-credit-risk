"""Command-line entry points for the reproducible workflow."""

from __future__ import annotations

import argparse
import json

from credit_risk.config import PROJECT_ROOT
from credit_risk.data import load_dataset, prepare_dataset
from credit_risk.modeling import train_and_evaluate
from credit_risk.sql_analysis import run_sql_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Credit risk project workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and clean the UCI data")
    download.add_argument("--force", action="store_true", help="Download again")
    download.add_argument(
        "--allow-upstream-change",
        action="store_true",
        help="Skip the pinned archive checksum (inspect provenance before using)",
    )
    subparsers.add_parser("train", help="Train, select, evaluate, and explain models")
    subparsers.add_parser("sql", help="Build SQLite database and export SQL analyses")
    subparsers.add_parser("all", help="Run data, modeling, explanations, and SQL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        output = prepare_dataset(force=args.force, verify_checksum=not args.allow_upstream_change)
        print(f"Prepared data: {output}")
        return

    if args.command == "all":
        prepare_dataset()

    frame = load_dataset()
    if args.command in {"train", "all"}:
        results = train_and_evaluate(frame)
        print(json.dumps(results["selection"], indent=2))
    if args.command in {"sql", "all"}:
        outputs = run_sql_analysis(frame)
        print(f"Exported {len(outputs)} SQL analyses under {PROJECT_ROOT / 'reports' / 'sql'}")


if __name__ == "__main__":
    main()
