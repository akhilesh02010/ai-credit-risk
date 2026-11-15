from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from credit_risk.config import PROJECT_ROOT, REPORTS_DIR

APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"
HAS_ARTIFACTS = (REPORTS_DIR / "test_predictions.csv").exists()


def test_dashboard_shows_setup_instructions_without_data(monkeypatch, tmp_path):
    import credit_risk.data as data

    monkeypatch.setattr(data, "PROCESSED_PATH", Path(tmp_path) / "missing.csv")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert "Generated artifacts are missing" in app.error[0].value


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="Run make all to generate integration artifacts")
def test_dashboard_renders_and_interactive_controls_work():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 6
    app.slider[0].set_value(0.5).run(timeout=30)
    assert not app.exception
    for grouping in ("age", "education", "sex"):
        app.selectbox[0].select(grouping).run(timeout=30)
        assert not app.exception
    for query_name in app.selectbox[1].options:
        app.selectbox[1].select(query_name).run(timeout=30)
        assert not app.exception
