"""
Training stage.

MLOps concepts demonstrated:
- Experiment tracking (every run's params + metrics logged to experiments.json,
  the same idea as MLflow / Weights & Biases, just file-based so it runs anywhere)
- Model = code + data + config, all versioned together
- The trained model is hashed by run and pushed into the model registry,
  NOT silently overwritten -- this gives full lineage/audit trail
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

sys.path.append(str(Path(__file__).parent))
from data_prep import TRAIN_PATH, TEST_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from model_registry import register_model

EXPERIMENTS_LOG = Path(__file__).parent.parent / "models" / "experiments.json"


def log_experiment(params: dict, metrics: dict) -> None:
    EXPERIMENTS_LOG.parent.mkdir(exist_ok=True)
    history = []
    if EXPERIMENTS_LOG.exists():
        history = json.loads(EXPERIMENTS_LOG.read_text())
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics,
    })
    EXPERIMENTS_LOG.write_text(json.dumps(history, indent=2))


def train(params: dict | None = None):
    params = params or {"n_estimators": 200, "max_depth": 8, "random_state": 42}

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 3),
        "rmse": round(mean_squared_error(y_test, preds) ** 0.5, 3),
        "r2": round(r2_score(y_test, preds), 4),
    }

    print(f"[train] params={params}")
    print(f"[train] metrics={metrics}")

    log_experiment(params, metrics)
    version = register_model(model, metrics, params, data_version="v1")
    return version, metrics


if __name__ == "__main__":
    train()
