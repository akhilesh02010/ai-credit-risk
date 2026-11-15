# Model card

## Intended use

An educational benchmark of next-month credit-card payment default using UCI dataset 350.
The analysis compares two fixed model specifications and demonstrates reproducible evaluation,
threshold trade-offs, descriptive group auditing, SQL analysis, and SHAP explanations.

The dataset concerns existing Taiwanese card customers in 2005. It contains no rejected
applicants, exposure-at-default or loss labels, or suitable row-level origination dates.
Results do not establish current applicant, loan-product, geographic, or temporal validity.

## Protocol

Identical profiles across all 23 input features are kept together using a fixed grouped,
approximately stratified split with seed 42. There are 17,999 training, 6,001 validation,
and 6,000 test observations. IDs and outcomes are excluded from model inputs.

All learned imputations, encodings, and scaling are fit on training observations only.
Logistic regression uses balanced class weighting; XGBoost weights positive examples by
the negative-to-positive ratio in the training set. Model parameters are fixed in source.
The candidate with the highest validation average precision is selected. The final test
set is not used for model or threshold selection.

The threshold minimizes validation cost on a predeclared grid from 0.05 to 0.95, with
false-negative cost 5 and false-positive cost 1. These are illustrative units, not financial
losses. The selected pipeline remains trained on the training partition; it is not refit
on train plus validation after threshold selection.

## Final results

XGBoost has validation average precision 0.571 and test average precision 0.559, ROC-AUC
0.787, and Brier score 0.175. At the validation-selected threshold 0.39, test precision is
0.381 and recall is 0.763. Its test confusion counts are TN 3,027, FP 1,646, FN 314, TP 1,013.

For comparison, logistic regression has test average precision 0.503 and ROC-AUC 0.721.
The constant training-prevalence baseline has test Brier score 0.172, slightly better than
XGBoost. Weighted model scores therefore must not be presented as calibrated probabilities
of default. External validation and calibration are needed before operational interpretation.

Full figures and machine-readable results are in `reports/`. The reported benchmark is
one fixed historical split; confidence intervals and repeated or out-of-time validation
are not provided. Small numerical differences across hardware/library platforms are possible.

## Explanations and fairness

SHAP explanations are computed for the XGBoost candidate on a deterministic sample of
2,000 test observations. Values are raw-margin (log-odds) contributions; they are neither
probability-point changes nor causal effects. The leading mean absolute contributions are
September repayment status, credit limit, and September bill amount. Correlated billing,
payment, and repayment-history variables can share or shift attribution.

SEX, AGE, EDUCATION, and MARRIAGE are retained for transparent research. They may be sensitive
attributes or proxies and are not a recommendation for underwriting. Held-out audits by
sex, age, and education report sample count, observed default prevalence, flag rate,
TPR, FPR, precision, and mean model score. Undefined TPR/FPR values remain missing rather
than being represented as evidence of zero error.

These descriptive metrics do not establish a cause for group differences or certify fairness.
They omit intersectional analysis and uncertainty estimates. The source's binary sex coding
and coarse socioeconomic categories limit what can be concluded.

## Deployment limits

Do not use this artifact to approve, deny, price, or market credit. It has no production
monitoring, appeals, adverse-action validation, temporal validation, security review, or
jurisdiction-specific governance. The model serves a reproducible analysis and learning goal.
