# Data dictionary

Source: I-Cheng Yeh, *Default of Credit Card Clients*, UCI dataset 350,
<https://doi.org/10.24432/C55S3H>, CC BY 4.0.

The processed dataset has 30,000 rows and 25 columns: an identifier, 23 features,
and one next-month payment-default outcome. Amounts are New Taiwan dollars (NT$).

| Processed column(s) | Meaning and handling |
| --- | --- |
| `ID` | Source row/customer identifier; unique; excluded from all model inputs. |
| `DEFAULT` | Next-month payment-default label, 1 = default and 0 = no default. Never an input. |
| `LIMIT_BAL` | Granted credit amount, including individual and supplementary family credit. |
| `SEX` | Source coding: 1 = male, 2 = female. A coarse, dated binary measure. One-hot encoded. |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = other. Source codes 0, 5, 6 map to 4 (other/unknown). One-hot encoded. |
| `MARRIAGE` | 1 = married, 2 = single, 3 = other. Source code 0 maps to 3 (other/unknown). One-hot encoded. |
| `AGE` | Age in years; numeric. |
| `PAY_SEPT`, `PAY_AUG`, `PAY_JUL`, `PAY_JUN`, `PAY_MAY`, `PAY_APR` | Historical repayment-status codes in 2005. Positive codes denote months of delay; UCI documents -1 as duly paid. Source-observed -2 and 0 are retained without assigning an unsupported meaning. |
| `BILL_SEPT`, `BILL_AUG`, `BILL_JUL`, `BILL_JUN`, `BILL_MAY`, `BILL_APR` | Monthly bill-statement amounts. Negative balances are retained. |
| `PAID_SEPT`, `PAID_AUG`, `PAID_JUL`, `PAID_JUN`, `PAID_MAY`, `PAID_APR` | Monthly previous-payment amounts as named by the source. |

`PAY_0` maps to September, while `PAY_2` through `PAY_6` map to August through April.
`BILL_AMT1`/`PAY_AMT1` map to September, through index 6 for April. These deterministic
renames and category mappings require no fitted statistics and do not use outcomes.

Numeric fields use training-only median imputation; categorical fields use training-only
mode imputation and one-hot encoding with unknown-category handling. Logistic regression
also standardizes numeric fields using training-only parameters. Repayment statuses are
treated numerically, which imposes an approximate ordering even for the ambiguous negative
codes; this is a modeling limitation to investigate with alternative encodings.

The source has no missing values in this run. Identifier and outcome missingness, infinite
values, repeated IDs, unexpected columns, and invalid categorical/target codes are rejected.
Feature imputation exists for robustness; it does not establish reliability under a new
missing-data process.
