from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent



RAW_DATA_PATH=(
    BASE_DIR
    / "data"
    / "raw"
    / "house_price.csv"
)

PROCESSED_DATA_PATH=(
    BASE_DIR
    / "data"
    / "processed"
    / "processed_house_price.csv"
)
FEATURE_ENGINEERED_DATA_PATH=(
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_house_price.csv"
)


MODEL_DIR=BASE_DIR / "models"
MODEL_PATH=MODEL_DIR / "best_model.pkl"
FIGURE_DIR=BASE_DIR / "outputs" /"figures"
REPORT_DIR=BASE_DIR / "outputs" / "reports"
PREDICTION_DIR=BASE_DIR / "outputs"/ "prediction"

TARGET_COLUMN= "Price"

TEST_SIZE=0.2

RANDOM_STATE=42

SHOW_FULL_REPORT=True
SHOW_EDA=True