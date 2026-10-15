"""
Monitoring stage.

Two things a production ML system must watch, demonstrated here:
1. Prediction logging -- every request/response is appended to a log so
   we can audit what the model said and when (predictions.log).
2. Data drift detection -- compare the distribution of INCOMING request
   features against the distribution the model was TRAINED on, using the
   Kolmogorov-Smirnov two-sample test. If live traffic drifts far from
   training data, the model's accuracy assumptions may no longer hold,
   and this should trigger a retraining pipeline in a real system.
"""
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp

DATA_DIR = Path(__file__).parent.parent / "data"
LOG_PATH = Path(__file__).parent / "predictions.log"
TRAIN_PATH = DATA_DIR / "train.csv"

DRIFT_P_VALUE_THRESHOLD = 0.05  # below this -> distributions are significantly different


def log_prediction(features: dict, result: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": result["predicted_score"],
        "model_version": result["model_version"],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def check_drift(recent_features: list[dict]) -> dict:
    """Compare a batch of recently-served feature dicts against training distribution."""
    if not TRAIN_PATH.exists():
        return {"status": "no_baseline", "detail": "train.csv not found"}

    train_df = pd.read_csv(TRAIN_PATH)
    live_df = pd.DataFrame(recent_features)

    report = {}
    drifted_any = False
    for col in ["study_hours", "attendance_pct", "sleep_hours", "previous_score"]:
        if col not in live_df.columns or len(live_df) < 5:
            continue
        stat, p_value = ks_2samp(train_df[col], live_df[col])
        drifted = p_value < DRIFT_P_VALUE_THRESHOLD
        drifted_any = drifted_any or drifted
        report[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p_value), 4),
                        "drifted": drifted}

    return {
        "status": "drift_detected" if drifted_any else "ok",
        "n_live_samples": len(live_df),
        "per_feature": report,
    }


def read_recent_predictions(n: int = 200) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().splitlines()[-n:]
    return [json.loads(l)["features"] for l in lines if l]


if __name__ == "__main__":
    recent = read_recent_predictions()
    print(check_drift(recent))
