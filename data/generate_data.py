"""
Data ingestion stage.

In a real MLOps pipeline this would pull from a database, data lake (S3),
or a feature store. Here we simulate a "Student Exam Score" dataset so the
whole pipeline is runnable offline and deterministically reproducible
(fixed random seed = reproducibility, a core MLOps principle).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 2000

RAW_DATA_PATH = Path(__file__).parent / "raw_students.csv"


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(4.5, 2.0, n_samples), 0, 12)
    attendance_pct = np.clip(rng.normal(80, 12, n_samples), 30, 100)
    sleep_hours = np.clip(rng.normal(6.5, 1.3, n_samples), 3, 10)
    previous_score = np.clip(rng.normal(65, 15, n_samples), 0, 100)
    extra_curricular = rng.integers(0, 2, n_samples)  # 0/1
    study_group = rng.integers(0, 2, n_samples)        # 0/1: studies in a group

    noise = rng.normal(0, 3, n_samples)

    # Ground-truth relationship (what the model has to learn)
    score = (
        1.9 * study_hours
        + 0.35 * attendance_pct
        + 0.9 * sleep_hours
        + 0.30 * previous_score
        + 2.0 * extra_curricular
        + 1.5 * study_group
        + noise
        - 15
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance_pct": attendance_pct.round(1),
        "sleep_hours": sleep_hours.round(2),
        "previous_score": previous_score.round(1),
        "extra_curricular": extra_curricular,
        "study_group": study_group,
        "exam_score": score.round(1),
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(df)} rows -> {RAW_DATA_PATH}")
    print(df.describe())
