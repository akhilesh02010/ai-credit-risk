import numpy as np
import pandas as pd
import pytest

from credit_risk.data import MONTHS
from credit_risk.modeling import select_threshold, split_dataset, threshold_curve


def synthetic_frame(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "ID": np.arange(1, rows + 1),
            "LIMIT_BAL": rng.integers(20_000, 800_000, rows),
            "SEX": rng.integers(1, 3, rows),
            "EDUCATION": rng.integers(1, 5, rows),
            "MARRIAGE": rng.integers(1, 4, rows),
            "AGE": rng.integers(21, 80, rows),
            "DEFAULT": np.tile([0, 0, 0, 1, 0], rows // 5),
        }
    )
    for month in MONTHS:
        frame[f"PAY_{month}"] = rng.integers(-2, 5, rows)
        frame[f"BILL_{month}"] = rng.integers(-2_000, 300_000, rows)
        frame[f"PAID_{month}"] = rng.integers(0, 80_000, rows)
    return frame


def test_split_is_disjoint_stratified_and_excludes_identifiers() -> None:
    split = split_dataset(synthetic_frame())
    train = set(split.train_indices)
    validation = set(split.validation_indices)
    test = set(split.test_indices)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert [len(train), len(validation), len(test)] == [300, 100, 100]
    assert "ID" not in split.X_train.columns
    assert "DEFAULT" not in split.X_train.columns
    assert abs(split.y_train.mean() - 0.2) < 0.03
    assert abs(split.y_validation.mean() - 0.2) < 0.03
    assert abs(split.y_test.mean() - 0.2) < 0.03


def test_threshold_is_selected_from_declared_cost_curve() -> None:
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.8, 0.2, 0.7, 0.9])
    curve = threshold_curve(y_true, probabilities)
    threshold = select_threshold(curve)
    assert 0.05 <= threshold <= 0.95
    selected_cost = curve.loc[curve["threshold"].eq(threshold), "cost_per_customer"].iloc[0]
    assert selected_cost == curve["cost_per_customer"].min()


def test_split_is_reproducible_and_covers_every_customer_once() -> None:
    frame = synthetic_frame()
    first = split_dataset(frame)
    second = split_dataset(frame)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    all_indices = np.concatenate(
        [first.train_indices, first.validation_indices, first.test_indices]
    )
    assert sorted(all_indices) == list(frame.index)


def test_split_rejects_repeated_customer_ids() -> None:
    frame = synthetic_frame()
    frame.loc[1, "ID"] = frame.loc[0, "ID"]
    with pytest.raises(ValueError, match="unique"):
        split_dataset(frame)


def test_identical_feature_profiles_never_cross_splits_even_with_different_labels():
    frame = synthetic_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["ID"] = 1_001
    duplicate["DEFAULT"] = 1 - duplicate["DEFAULT"]
    frame = pd.concat([frame, duplicate], ignore_index=True)
    split = split_dataset(frame)
    index_sets = [
        set(split.train_indices),
        set(split.validation_indices),
        set(split.test_indices),
    ]
    assert any({0, 500}.issubset(indices) for indices in index_sets)
