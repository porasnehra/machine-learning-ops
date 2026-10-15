"""
Inference / prediction stage.

This module is intentionally the ONLY place that knows how to turn raw
input features into a prediction. Both the CLI and the FastAPI service
import this, so training-serving skew (a classic MLOps failure mode) is
avoided -- there's one code path for feature order and one for the model.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model_registry import get_production_model
from data_prep import FEATURE_COLUMNS

import pandas as pd


def predict_one(features: dict) -> dict:
    model, meta = get_production_model()
    row = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    score = float(model.predict(row)[0])
    score = max(0.0, min(100.0, score))
    return {
        "predicted_score": round(score, 1),
        "model_version": meta["version"],
        "model_metrics": meta["metrics"],
    }


if __name__ == "__main__":
    example = {
        "study_hours": 5.5,
        "attendance_pct": 88,
        "sleep_hours": 7,
        "previous_score": 72,
        "extra_curricular": 1,
        "study_group": 1,
    }
    print(predict_one(example))
