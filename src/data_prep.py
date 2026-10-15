"""
Data validation + preprocessing stage.

MLOps concepts demonstrated here:
- Data validation (schema + range checks) BEFORE training ever happens
- Reproducible train/test split (fixed seed)
- Versioned, saved splits so training is auditable/repeatable
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "raw_students.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

FEATURE_COLUMNS = [
    "study_hours", "attendance_pct", "sleep_hours",
    "previous_score", "extra_curricular", "study_group",
]
TARGET_COLUMN = "exam_score"

SCHEMA = {
    "study_hours": (0, 24),
    "attendance_pct": (0, 100),
    "sleep_hours": (0, 24),
    "previous_score": (0, 100),
    "extra_curricular": (0, 1),
    "study_group": (0, 1),
    "exam_score": (0, 100),
}


class DataValidationError(Exception):
    pass


def validate(df: pd.DataFrame) -> None:
    """Fail fast if incoming data doesn't match the expected contract."""
    missing_cols = set(SCHEMA) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    if df.isnull().any().any():
        raise DataValidationError("Null values detected in dataset")

    for col, (lo, hi) in SCHEMA.items():
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if len(out_of_range) > 0:
            raise DataValidationError(
                f"Column '{col}' has {len(out_of_range)} rows outside [{lo}, {hi}]"
            )

    print(f"[validate] OK — {len(df)} rows passed schema + range checks")


def prepare(test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(RAW_PATH)
    validate(df)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"[prepare] train={len(train_df)} rows -> {TRAIN_PATH}")
    print(f"[prepare] test={len(test_df)} rows -> {TEST_PATH}")
    return train_df, test_df


if __name__ == "__main__":
    prepare()
