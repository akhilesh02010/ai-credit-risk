"""Descriptive group metrics for model auditing (not a fairness certification)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def group_metrics(
    groups: pd.Series,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Compute selection and error rates by group on a held-out sample."""
    audit = pd.DataFrame(
        {
            "group": groups.astype(str).to_numpy(),
            "y_true": y_true.to_numpy(),
            "probability": probabilities,
        },
        index=y_true.index,
    )
    audit["prediction"] = (audit["probability"] >= threshold).astype(int)
    rows: list[dict[str, float | int | str]] = []
    for group_name, subset in audit.groupby("group", dropna=False):
        tn, fp, fn, tp = confusion_matrix(
            subset["y_true"], subset["prediction"], labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "group": group_name,
                "n": int(len(subset)),
                "observed_default_rate": float(subset["y_true"].mean()),
                "predicted_positive_rate": float(subset["prediction"].mean()),
                "mean_predicted_risk": float(subset["probability"].mean()),
                "true_positive_rate": _safe_rate(tp, tp + fn),
                "false_positive_rate": _safe_rate(fp, fp + tn),
                "precision": float(
                    precision_score(subset["y_true"], subset["prediction"], zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
