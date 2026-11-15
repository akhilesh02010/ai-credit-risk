from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.config import FIGURES_DIR, REPORTS_DIR, SQL_REPORTS_DIR  # noqa: E402
from credit_risk.data import PROCESSED_PATH  # noqa: E402

st.set_page_config(page_title="Credit Default Risk Lab", page_icon="📊", layout="wide")
st.title("Credit Default Risk Lab")
st.caption(
    "Portfolio analytics and held-out model evaluation on the UCI Taiwan credit-card dataset. "
    "Educational analysis only—not a lending decision system."
)

required = [
    PROCESSED_PATH,
    REPORTS_DIR / "model_metrics.json",
    REPORTS_DIR / "test_predictions.csv",
]
missing = [path for path in required if not path.exists()]
if missing:
    st.error("Generated artifacts are missing. Run `uv run credit-risk all` first.")
    st.code("uv run credit-risk all")
    st.stop()


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    data = pd.read_csv(PROCESSED_PATH)
    metrics = json.loads((REPORTS_DIR / "model_metrics.json").read_text())
    predictions = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    return data, metrics, predictions


data, metrics, predictions = load_artifacts()
selection = metrics["selection"]
selected_name = selection["selected_model"].replace("_", " ").title()

overview, drivers, threshold_tab, segment_tab, sql_tab, limitations = st.tabs(
    ["Overview", "Risk drivers", "Threshold lab", "Segment audit", "SQL analysis", "Limitations"]
)

with overview:
    test_metrics = metrics["test_metrics"][selection["selected_model"]]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(data):,}")
    col2.metric("Observed defaults", f"{data['DEFAULT'].mean():.1%}")
    col3.metric("Selected model", selected_name)
    col4.metric("Test PR-AUC", f"{test_metrics['pr_auc']:.3f}")
    st.info(metrics["disclosure"])
    st.caption(
        "PR-AUC is average precision. Scores come from class-weighted models and require "
        "calibration before they can be interpreted as default probabilities. See Limitations."
    )
    if (FIGURES_DIR / "model_comparison.png").exists():
        st.image(str(FIGURES_DIR / "model_comparison.png"), width="stretch")

with drivers:
    importance_path = REPORTS_DIR / "shap_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).sort_values("mean_abs_shap")
        figure = px.bar(
            importance,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
            title="Global XGBoost feature importance on a held-out sample",
        )
        st.plotly_chart(figure, width="stretch")
    if (FIGURES_DIR / "shap_summary.png").exists():
        st.image(str(FIGURES_DIR / "shap_summary.png"), width="stretch")
    st.caption(
        "SHAP describes this fitted model, not causal effects. A large attribution does not mean "
        "a feature causes default or is appropriate for a real lending decision."
    )

with threshold_tab:
    locked_threshold = float(selection["selected_threshold"])
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=locked_threshold,
        step=0.01,
        help="The marked default was selected using validation data only.",
    )
    y_true = predictions["actual_default"].to_numpy()
    y_pred = (predictions["predicted_probability"].to_numpy() >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cols = st.columns(5)
    cols[0].metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.1%}")
    cols[1].metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.1%}")
    cols[2].metric("False positives", f"{fp:,}")
    cols[3].metric("False negatives", f"{fn:,}")
    cols[4].metric("Locked threshold", f"{locked_threshold:.2f}")
    confusion = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual non-default", "Actual default"],
        columns=["Predicted non-default", "Predicted default"],
    )
    st.dataframe(confusion, width="stretch")
    st.warning(
        "Changing this slider inspects test-set trade-offs. It must not be used to retune the "
        "reported model; a new threshold needs new validation data and documented business costs."
    )

with segment_tab:
    audit_choice = st.selectbox("Audit grouping", ["sex", "age", "education"])
    audit = pd.read_csv(REPORTS_DIR / f"fairness_by_{audit_choice}.csv")
    st.dataframe(audit.style.format(precision=3), width="stretch")
    rate_columns = [
        "observed_default_rate",
        "predicted_positive_rate",
        "true_positive_rate",
        "false_positive_rate",
    ]
    melted = audit.melt(id_vars="group", value_vars=rate_columns, var_name="metric")
    st.plotly_chart(
        px.bar(melted, x="group", y="value", color="metric", barmode="group"),
        width="stretch",
    )
    st.caption(
        "These are descriptive held-out metrics. Group gaps can reflect sampling, measurement, "
        "historical inequity, or model behavior; they do not establish legal or ethical fairness."
    )

with sql_tab:
    sql_files = sorted(SQL_REPORTS_DIR.glob("*.csv"))
    if sql_files:
        choice = st.selectbox("Saved aggregate query", [path.stem for path in sql_files])
        selected_file = SQL_REPORTS_DIR / f"{choice}.csv"
        st.dataframe(pd.read_csv(selected_file), width="stretch")
    else:
        st.info("Run `uv run credit-risk sql` to generate the SQL analysis tables.")

with limitations:
    st.markdown(
        """
        - The data describe 30,000 Taiwanese credit-card clients in 2005, not current applicants.
        - The target is next-month credit-card payment default, not general loan default or loss.
        - The random split tests interpolation within one historical dataset; it does not establish
          temporal, geographic, economic-cycle, or product-level generalization.
        - SEX, EDUCATION, MARRIAGE, and AGE are sensitive or proxy attributes. Their inclusion here
          supports analysis and auditing, not a recommendation to use them in production decisions.
        - Category definitions are coarse and dated. Undocumented source codes were grouped into
          “other/unknown,” and repayment codes -2 and 0 were retained as observed source values.
        - Class weighting improves minority-class attention but can distort probability calibration.
          Any real deployment requires calibration, policy review, adverse-action reason validation,
          monitoring, security, human oversight, and jurisdiction-specific legal review.
        """
    )
