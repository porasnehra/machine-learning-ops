"""
A minimal Model Registry, built from scratch, so the pipeline demonstrates
the CONCEPT of a registry (what tools like MLflow Model Registry / SageMaker
Model Registry / Vertex Model Registry do) without needing an external server.

Responsibilities of a model registry in real MLOps:
1. Store every trained model as an immutable, versioned artifact
2. Store the metrics + hyperparameters + data version alongside it (lineage)
3. Track a "stage" per version: Staging -> Production -> Archived
4. Let serving code always ask for "the current Production model"
   without caring which version number that is
"""
import json
import joblib
from pathlib import Path
from datetime import datetime, timezone

REGISTRY_DIR = Path(__file__).parent.parent / "models"
REGISTRY_INDEX = REGISTRY_DIR / "registry.json"


def _load_index() -> dict:
    if REGISTRY_INDEX.exists():
        return json.loads(REGISTRY_INDEX.read_text())
    return {"versions": []}


def _save_index(index: dict) -> None:
    REGISTRY_INDEX.write_text(json.dumps(index, indent=2))


def register_model(model, metrics: dict, params: dict, data_version: str = "v1") -> int:
    """Save a trained model as a new immutable version and record its metadata."""
    REGISTRY_DIR.mkdir(exist_ok=True)
    index = _load_index()
    version = len(index["versions"]) + 1

    model_path = REGISTRY_DIR / f"model_v{version}.joblib"
    joblib.dump(model, model_path)

    entry = {
        "version": version,
        "model_path": str(model_path.name),
        "metrics": metrics,
        "params": params,
        "data_version": data_version,
        "stage": "Staging",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    index["versions"].append(entry)
    _save_index(index)
    print(f"[registry] Registered model v{version} (stage=Staging) metrics={metrics}")
    return version


def promote(version: int, stage: str = "Production") -> None:
    """Move a version to a new stage. Only one model may hold 'Production' at a time."""
    index = _load_index()
    for entry in index["versions"]:
        if stage == "Production" and entry["stage"] == "Production":
            entry["stage"] = "Archived"
        if entry["version"] == version:
            entry["stage"] = stage
    _save_index(index)
    print(f"[registry] Promoted model v{version} -> {stage}")


def get_production_model():
    """What the serving layer (API) calls — never hardcode a version in the app."""
    index = _load_index()
    for entry in reversed(index["versions"]):
        if entry["stage"] == "Production":
            model = joblib.load(REGISTRY_DIR / entry["model_path"])
            return model, entry
    raise RuntimeError("No model is currently in Production stage.")


def list_versions() -> list:
    return _load_index()["versions"]


if __name__ == "__main__":
    for v in list_versions():
        print(v)
