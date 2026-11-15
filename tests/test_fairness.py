import numpy as np
import pandas as pd

from credit_risk.fairness import group_metrics


def test_group_rates_preserve_nonconsecutive_test_indices_and_undefined_rates():
    indices = [17, 4, 99, 23]
    groups = pd.Series(["A", "A", "B", "B"], index=indices)
    target = pd.Series([0, 1, 0, 0], index=indices)
    result = group_metrics(groups, target, np.array([0.1, 0.9, 0.8, 0.2]), 0.5)
    result = result.set_index("group")
    assert result.loc["A", "true_positive_rate"] == 1.0
    assert result.loc["A", "false_positive_rate"] == 0.0
    assert result.loc["B", "false_positive_rate"] == 0.5
    assert np.isnan(result.loc["B", "true_positive_rate"])
    assert result["n"].sum() == len(target)
