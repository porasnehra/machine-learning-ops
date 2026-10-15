"""
Automated tests — this is what a CI pipeline (GitHub Actions, Jenkins,
GitLab CI) runs on every pull request BEFORE anything is allowed to merge
or deploy. Run locally with:  pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from data.generate_data import generate_dataset
from src.data_prep import validate, DataValidationError, FEATURE_COLUMNS, TARGET_COLUMN


def test_generated_data_has_expected_columns():
    df = generate_dataset(n_samples=50)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns


def test_generated_data_passes_validation():
    df = generate_dataset(n_samples=50)
    validate(df)  # should not raise


def test_validation_rejects_out_of_range_values():
    df = generate_dataset(n_samples=10)
    df.loc[0, "attendance_pct"] = 150  # invalid: > 100
    with pytest.raises(DataValidationError):
        validate(df)


def test_validation_rejects_nulls():
    df = generate_dataset(n_samples=10)
    df.loc[0, "study_hours"] = None
    with pytest.raises(DataValidationError):
        validate(df)


class _DummyModel:
    """Module-level (not local) so joblib/pickle can serialize it in tests."""
    def predict(self, X):
        return [50.0] * len(X)


def test_model_registry_roundtrip(tmp_path, monkeypatch):
    import src.model_registry as registry
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_INDEX", tmp_path / "registry.json")

    version = registry.register_model(_DummyModel(), {"r2": 0.9, "mae": 2.0}, {"n_estimators": 10})
    registry.promote(version, "Production")
    model, meta = registry.get_production_model()
    assert meta["version"] == version
    assert meta["stage"] == "Production"


def test_prediction_is_within_valid_score_range():
    """End-to-end sanity check using whatever model is currently in Production."""
    from src.predict import predict_one
    sample = {
        "study_hours": 5.0, "attendance_pct": 85, "sleep_hours": 7,
        "previous_score": 70, "extra_curricular": 1, "study_group": 1,
    }
    try:
        result = predict_one(sample)
    except RuntimeError:
        pytest.skip("No production model registered yet — run pipeline.py first")
    assert 0 <= result["predicted_score"] <= 100
