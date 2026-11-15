"""Train, select, evaluate, and explain credit default models."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_directories,
)
from credit_risk.data import PROCESSED_PATH, sha256
from credit_risk.fairness import group_metrics
from credit_risk.features import MODEL_FEATURES, make_preprocessor


@dataclass(frozen=True)
class DataSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(frame: pd.DataFrame) -> DataSplit:
    """Create approximately stratified 60/20/20 splits, grouping exact feature profiles."""
    if not frame.index.is_unique or frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("Split input must have unique row indices and unique, nonmissing IDs")
    # Group by features only: no target is used to define a profile. Different IDs with
    # identical observed inputs cannot span fitting and evaluation sets.
    groups = pd.factorize(pd.MultiIndex.from_frame(frame[MODEL_FEATURES]), sort=False)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_validation_pos, test_pos = next(
        outer.split(frame[MODEL_FEATURES], frame["DEFAULT"], groups)
    )
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_pos, validation_pos = next(
        inner.split(
            frame.iloc[train_validation_pos][MODEL_FEATURES],
            frame.iloc[train_validation_pos]["DEFAULT"],
            groups[train_validation_pos],
        )
    )
    train_idx = frame.index.to_numpy()[train_validation_pos[train_pos]]
    validation_idx = frame.index.to_numpy()[train_validation_pos[validation_pos]]
    test_idx = frame.index.to_numpy()[test_pos]

    def features(indices: np.ndarray) -> pd.DataFrame:
        return frame.loc[indices, MODEL_FEATURES].copy()

    def target(indices: np.ndarray) -> pd.Series:
        return frame.loc[indices, "DEFAULT"].copy()

    return DataSplit(
        train_indices=train_idx,
        validation_indices=validation_idx,
        test_indices=test_idx,
        X_train=features(train_idx),
        X_validation=features(validation_idx),
        X_test=features(test_idx),
        y_train=target(train_idx),
        y_validation=target(validation_idx),
        y_test=target(test_idx),
    )


def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    imbalance_ratio = negative_count / positive_count
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.035,
                        min_child_weight=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.05,
                        reg_lambda=1.5,
                        scale_pos_weight=imbalance_ratio,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
    }


def probability_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }


def classification_metrics(
    y_true: pd.Series, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_curve(
    y_true: pd.Series,
    probability: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metrics = classification_metrics(y_true, probability, float(threshold))
        cost = (
            metrics["false_positives"] * false_positive_cost
            + metrics["false_negatives"] * false_negative_cost
        ) / len(y_true)
        rows.append({**metrics, "cost_per_customer": float(cost)})
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame) -> float:
    """Select the validation threshold minimizing the declared 5:1 FN:FP cost."""
    best = curve.sort_values(
        ["cost_per_customer", "recall", "precision"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["threshold"])


def _save_comparison_plots(y_test: pd.Series, test_probabilities: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, probability in test_probabilities.items():
        label = model_name.replace("_", " ").title()
        PrecisionRecallDisplay.from_predictions(y_test, probability, name=label, ax=axes[0])
        RocCurveDisplay.from_predictions(y_test, probability, name=label, ax=axes[1])
    axes[0].set_title("Held-out test: precision-recall")
    axes[1].set_title("Held-out test: ROC")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_shap_artifacts(model: Pipeline, X_test: pd.DataFrame) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    sample = X_test.sample(n=min(2_000, len(X_test)), random_state=RANDOM_STATE)
    transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=sample.index)

    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed_frame)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)

    shap.plots.beeswarm(explanation, max_display=18, show=False)
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def train_and_evaluate(frame: pd.DataFrame) -> dict[str, object]:
    """Run model comparison, validation-only selection, and one-shot test evaluation."""
    ensure_directories()
    split = split_dataset(frame)
    manifest = pd.concat(
        [
            pd.DataFrame({"ID": frame.loc[indices, "ID"].to_numpy(), "split": name})
            for name, indices in (
                ("train", split.train_indices),
                ("validation", split.validation_indices),
                ("test", split.test_indices),
            )
        ],
        ignore_index=True,
    )
    manifest.to_csv(REPORTS_DIR / "split_manifest.csv", index=False)
    models = build_models(split.y_train)
    validation_probabilities: dict[str, np.ndarray] = {}
    validation_metrics: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        model.fit(split.X_train, split.y_train)
        validation_probability = model.predict_proba(split.X_validation)[:, 1]
        validation_probabilities[model_name] = validation_probability
        validation_metrics[model_name] = probability_metrics(
            split.y_validation, validation_probability
        )
        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

    selected_model_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["pr_auc"],
    )
    selected_model = models[selected_model_name]
    validation_curve = threshold_curve(
        split.y_validation, validation_probabilities[selected_model_name]
    )
    selected_threshold = select_threshold(validation_curve)
    validation_curve.to_csv(REPORTS_DIR / "validation_threshold_curve.csv", index=False)

    test_probabilities: dict[str, np.ndarray] = {}
    test_metrics = {}
    for model_name, model in models.items():
        probability = model.predict_proba(split.X_test)[:, 1]
        test_probabilities[model_name] = probability
        test_metrics[model_name] = {
            **probability_metrics(split.y_test, probability),
            "at_0_50": classification_metrics(split.y_test, probability, 0.50),
        }

    selected_test_probability = test_probabilities[selected_model_name]
    selected_test_metrics = classification_metrics(
        split.y_test, selected_test_probability, selected_threshold
    )
    test_predictions = pd.DataFrame(
        {
            "ID": frame.loc[split.test_indices, "ID"].to_numpy(),
            "actual_default": split.y_test.to_numpy(),
            "predicted_probability": selected_test_probability,
            "prediction": (selected_test_probability >= selected_threshold).astype(int),
        }
    )
    test_predictions.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    sex_labels = frame.loc[split.test_indices, "SEX"].map({1: "Male", 2: "Female"})
    age_groups = pd.cut(
        frame.loc[split.test_indices, "AGE"],
        bins=[0, 29, 39, 49, 59, np.inf],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
    )
    education_labels = frame.loc[split.test_indices, "EDUCATION"].map(
        {1: "Graduate school", 2: "University", 3: "High school", 4: "Other/unknown"}
    )
    for audit_name, groups in {
        "sex": sex_labels,
        "age": age_groups,
        "education": education_labels,
    }.items():
        group_metrics(
            groups=groups,
            y_true=split.y_test,
            probabilities=selected_test_probability,
            threshold=selected_threshold,
        ).to_csv(REPORTS_DIR / f"fairness_by_{audit_name}.csv", index=False)

    _save_comparison_plots(split.y_test, test_probabilities)
    _save_shap_artifacts(models["xgboost"], split.X_test)
    joblib.dump(selected_model, MODELS_DIR / "selected_model.joblib")

    results: dict[str, object] = {
        "split": {
            "strategy": (
                "approximately stratified 60/20/20 train/validation/test; "
                "exact feature profiles grouped within one split"
            ),
            "random_state": RANDOM_STATE,
            "train_rows": len(split.train_indices),
            "validation_rows": len(split.validation_indices),
            "test_rows": len(split.test_indices),
            "repeated_feature_profiles": int(frame[MODEL_FEATURES].duplicated().sum()),
            "train_default_rate": float(split.y_train.mean()),
            "validation_default_rate": float(split.y_validation.mean()),
            "test_default_rate": float(split.y_test.mean()),
        },
        "selection": {
            "model_metric": "validation PR-AUC",
            "selected_model": selected_model_name,
            "threshold_rule": "minimum validation cost with false-negative:false-positive = 5:1",
            "selected_threshold": selected_threshold,
        },
        "validation_probability_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_constant_baseline": probability_metrics(
            split.y_test, np.full(len(split.y_test), split.y_train.mean())
        ),
        "selected_model_test_at_selected_threshold": selected_test_metrics,
        "disclosure": (
            "The final test set was not used for preprocessing, fitting, model selection, "
            "or threshold selection. Test metrics compare locked candidates and should not "
            "be used for further tuning."
        ),
    }
    (REPORTS_DIR / "model_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "streamlit")
        },
        "dataset_sha256": sha256(PROCESSED_PATH) if PROCESSED_PATH.exists() else None,
        "random_state": RANDOM_STATE,
        "features": MODEL_FEATURES,
        "model_parameters": {
            name: model.named_steps["classifier"].get_params() for name, model in models.items()
        },
        "shap": {
            "model": "xgboost",
            "sample_rows": min(2_000, len(split.X_test)),
            "output_scale": "raw margin (log odds), not probability or causal effect",
        },
    }
    (REPORTS_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    metadata_path = PROCESSED_DIR / "dataset_metadata.json"
    if metadata_path.exists():
        (REPORTS_DIR / "dataset_metadata.json").write_text(
            metadata_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return results
