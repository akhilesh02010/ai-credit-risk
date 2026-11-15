import pandas as pd
import pytest

from credit_risk.data import MONTHS, clean_raw_dataframe


def raw_frame() -> pd.DataFrame:
    row = {
        "ID": 1,
        "LIMIT_BAL": 100_000,
        "SEX": 2,
        "EDUCATION": 5,
        "MARRIAGE": 0,
        "AGE": 32,
        "default payment next month": 1,
    }
    pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    row.update(dict.fromkeys(pay_columns, -1))
    row.update({f"BILL_AMT{i}": i * 1_000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": i * 100 for i in range(1, 7)})
    return pd.DataFrame([row])


def test_clean_raw_dataframe_normalizes_names_and_unknown_categories() -> None:
    clean = clean_raw_dataframe(raw_frame())
    assert clean.loc[0, "EDUCATION"] == 4
    assert clean.loc[0, "MARRIAGE"] == 3
    assert clean.loc[0, "DEFAULT"] == 1
    for month in MONTHS:
        assert f"PAY_{month}" in clean.columns
        assert f"BILL_{month}" in clean.columns
        assert f"PAID_{month}" in clean.columns


def test_clean_raw_dataframe_rejects_invalid_target() -> None:
    raw = raw_frame()
    raw.loc[0, "default payment next month"] = 2
    try:
        clean_raw_dataframe(raw)
    except ValueError as error:
        assert "binary" in str(error)
    else:
        raise AssertionError("Expected invalid target to be rejected")


@pytest.mark.parametrize("column", ["ID", "default payment next month"])
def test_clean_raw_dataframe_rejects_missing_identifiers_and_labels(column):
    raw = raw_frame().astype(float)
    raw.loc[0, column] = float("nan")
    with pytest.raises(ValueError, match="must not be missing"):
        clean_raw_dataframe(raw)


def test_clean_raw_dataframe_rejects_unexpected_columns():
    raw = raw_frame().assign(future_payment=100)
    with pytest.raises(ValueError, match="Unexpected"):
        clean_raw_dataframe(raw)
