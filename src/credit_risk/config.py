from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_REPORTS_DIR = REPORTS_DIR / "sql"
SQL_FILE = PROJECT_ROOT / "sql" / "analysis.sql"

UCI_DATASET_ID = 350
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_DATASET_DOI = "https://doi.org/10.24432/C55S3H"
UCI_ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
RANDOM_STATE = 42
TARGET = "DEFAULT"
ID_COLUMN = "ID"


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, SQL_REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
