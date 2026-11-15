"""Download, validate, and clean the UCI credit default dataset."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import (
    PROCESSED_DIR,
    RAW_DIR,
    UCI_DATASET_DOI,
    UCI_DATASET_ID,
    UCI_DATASET_URL,
    UCI_ZIP_SHA256,
    ensure_directories,
)

ZIP_PATH = RAW_DIR / "default_credit_card_clients.zip"
XLS_PATH = RAW_DIR / "default of credit card clients.xls"
PROCESSED_PATH = PROCESSED_DIR / "credit_default.csv"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"

MONTHS = ["SEPT", "AUG", "JUL", "JUN", "MAY", "APR"]
RENAME_COLUMNS = {
    "PAY_0": "PAY_SEPT",
    "PAY_2": "PAY_AUG",
    "PAY_3": "PAY_JUL",
    "PAY_4": "PAY_JUN",
    "PAY_5": "PAY_MAY",
    "PAY_6": "PAY_APR",
    **{f"BILL_AMT{i}": f"BILL_{month}" for i, month in enumerate(MONTHS, start=1)},
    **{f"PAY_AMT{i}": f"PAID_{month}" for i, month in enumerate(MONTHS, start=1)},
    "default payment next month": "DEFAULT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(force: bool = False, verify_checksum: bool = True) -> Path:
    """Download the versioned UCI archive and verify its known SHA-256 digest."""
    ensure_directories()
    if force or not ZIP_PATH.exists():
        temporary_path = ZIP_PATH.with_suffix(".zip.part")
        request = urllib.request.Request(
            UCI_DATASET_URL,
            headers={"User-Agent": "ai-credit-risk/0.1 (educational reproducibility)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            temporary_path.write_bytes(response.read())
        temporary_path.replace(ZIP_PATH)

    actual_hash = sha256(ZIP_PATH)
    if verify_checksum and actual_hash != UCI_ZIP_SHA256:
        raise ValueError(
            "The UCI archive checksum changed. Refusing to continue because the upstream "
            f"data may have changed. Expected {UCI_ZIP_SHA256}, got {actual_hash}."
        )

    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = archive.namelist()
        if members != ["default of credit card clients.xls"]:
            raise ValueError(f"Unexpected archive contents: {members}")
        archive.extract(members[0], RAW_DIR)
    return XLS_PATH


def clean_raw_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and documented category anomalies without using the target."""
    frame = raw.rename(columns=RENAME_COLUMNS).copy()
    expected = {"ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "DEFAULT"}
    expected.update(f"PAY_{month}" for month in MONTHS)
    expected.update(f"BILL_{month}" for month in MONTHS)
    expected.update(f"PAID_{month}" for month in MONTHS)
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    unexpected = sorted(set(frame.columns).difference(expected))
    if unexpected or not frame.columns.is_unique:
        raise ValueError(f"Unexpected or duplicate columns: {unexpected}")

    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame[["ID", "DEFAULT"]].isna().any().any():
        raise ValueError("Customer IDs and DEFAULT must not be missing")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite values are not supported")
    frame["EDUCATION"] = frame["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    frame["MARRIAGE"] = frame["MARRIAGE"].replace({0: 3})

    if frame["ID"].duplicated().any():
        raise ValueError("Customer IDs must be unique")
    if not set(frame["DEFAULT"].unique()).issubset({0, 1}):
        raise ValueError("DEFAULT must be binary")
    if not set(frame["SEX"].unique()).issubset({1, 2}):
        raise ValueError("Unexpected SEX code")
    if not set(frame["EDUCATION"].unique()).issubset({1, 2, 3, 4}):
        raise ValueError("Unexpected EDUCATION code after normalization")
    if not set(frame["MARRIAGE"].unique()).issubset({1, 2, 3}):
        raise ValueError("Unexpected MARRIAGE code after normalization")
    return frame


def prepare_dataset(force: bool = False, verify_checksum: bool = True) -> Path:
    """Create a clean CSV and provenance metadata from the official workbook."""
    workbook_path = download_dataset(force=force, verify_checksum=verify_checksum)
    raw = pd.read_excel(workbook_path, header=1, engine="xlrd")
    clean = clean_raw_dataframe(raw)
    if clean.shape != (30_000, 25):
        raise ValueError(f"Expected 30,000 rows and 25 columns, got {clean.shape}")
    clean.to_csv(PROCESSED_PATH, index=False)

    metadata = {
        "dataset_id": UCI_DATASET_ID,
        "title": "Default of Credit Card Clients",
        "creator": "I-Cheng Yeh",
        "doi": UCI_DATASET_DOI,
        "source_url": UCI_DATASET_URL,
        "license": "CC BY 4.0",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256(ZIP_PATH),
        "processed_csv_sha256": sha256(PROCESSED_PATH),
        "missing_values": {name: int(count) for name, count in clean.isna().sum().items()},
        "rows": int(clean.shape[0]),
        "columns": int(clean.shape[1]),
        "target": "DEFAULT",
        "positive_class_rate": float(clean["DEFAULT"].mean()),
        "cleaning": [
            "Renamed monthly columns to explicit SEPT-to-APR names.",
            "Mapped undocumented EDUCATION codes 0, 5, and 6 to OTHER (4).",
            "Mapped undocumented MARRIAGE code 0 to OTHER (3).",
            "Preserved repayment codes -2 and 0 because they occur in the source data.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PROCESSED_PATH


def load_dataset() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_PATH}. Run `uv run credit-risk download`."
        )
    return pd.read_csv(PROCESSED_PATH)
