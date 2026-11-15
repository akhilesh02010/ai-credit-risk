# Local verification — November 15, 2026

Verified on this Apple Silicon Mac using Python 3.12.16 and uv 0.13.8. Exact Python
dependencies are recorded in `uv.lock`; installed modeling-library versions and parameters
are recorded in `reports/run_metadata.json`. Native libomp installation is documented in
the README because it is not managed by uv.

| Check | Result |
| --- | --- |
| `make setup` | Locked environment synchronized successfully. |
| `make all` | Official UCI archive checksum verified; 30,000 rows prepared; both model pipelines trained; validation selection, final evaluation, SHAP, and all five SQL analyses completed. |
| `make check` | Ruff passed; 16 tests passed, including real SQL execution, training-only preprocessing statistics, unknown categories, grouped split isolation, schema rejection, fairness edge cases, and Streamlit controls. |
| `make notebooks` | Both notebooks executed successfully, with plots and no cell errors; EDA has 5 executed code cells and model evaluation has 6. |
| Saved artifacts | Reloaded selected pipeline reproduced saved test scores within floating-point tolerance; all three partitions had disjoint sets of exact feature profiles. |
| Dashboard web launch | Started at `127.0.0.1:8501`; `/_stcore/health` returned `ok` and the application page returned HTML. |
| JupyterLab web launch | Started on localhost; `/lab` returned the expected HTTP 302 redirect to the authenticated login flow. |
| Requirements export | Runtime and development requirements exported from the lock with SHA-256 hashes. |
| Git exclusions | Downloaded data, trained models, row-level predictions, split manifest, virtual environment, `.env`, and Streamlit secrets are ignored. |

Notebook kernels and web-service checks required local-socket permission outside the app's
sandbox. They completed successfully; notebook execution emitted a local-kernel transport
warning but no cell errors. The desktop browser integration was unavailable, so UI checks
used Streamlit's interaction-testing API plus live HTTP checks rather than a screenshot review.

The final benchmark supersedes the preliminary simple random split: identical feature
profiles now remain within a single partition. XGBoost test average precision is 0.558779
and ROC-AUC is 0.786856; threshold 0.39 was selected on validation data. No model
hyperparameters were changed in response to final test metrics.

## GitHub handoff

GitHub CLI authentication was verified for `akhilesh02010`, with credentials stored in
the macOS keychain. The private repository is
[akhilesh02010/ai-credit-risk](https://github.com/akhilesh02010/ai-credit-risk).
The checkout uses repository-local author settings and GitHub's no-reply email format.
No credentials, downloaded data, trained models, or row-level predictions are committed.

The checks above describe local verification. The latest remote workflow status is
available in [GitHub Actions](https://github.com/akhilesh02010/ai-credit-risk/actions).
