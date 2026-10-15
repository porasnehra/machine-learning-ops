"""
Evaluation / Quality Gate stage.

This is the "Continuous Training / Continuous Delivery" gate: a new model
is only promoted to Production if it beats a minimum bar AND is not worse
than the model currently serving traffic. This prevents silent regressions
from ever reaching users -- one of the most important ideas in MLOps.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model_registry import list_versions, promote, get_production_model

MIN_R2 = 0.75          # absolute quality bar a model must clear
MAX_MAE = 8.0          # absolute error bar (exam-score points)


def gate_check(version: int) -> bool:
    versions = list_versions()
    entry = next(v for v in versions if v["version"] == version)
    metrics = entry["metrics"]

    passes_bar = metrics["r2"] >= MIN_R2 and metrics["mae"] <= MAX_MAE

    # Compare against current production model, if one exists
    beats_incumbent = True
    try:
        _, prod_entry = get_production_model()
        beats_incumbent = metrics["r2"] >= prod_entry["metrics"]["r2"]
        print(f"[evaluate] incumbent v{prod_entry['version']} r2={prod_entry['metrics']['r2']} "
              f"vs candidate v{version} r2={metrics['r2']}")
    except RuntimeError:
        print("[evaluate] no incumbent production model yet — first deploy")

    passed = passes_bar and beats_incumbent
    verdict = "PASS" if passed else "FAIL"
    print(f"[evaluate] v{version} metrics={metrics} -> {verdict}")
    return passed


def evaluate_and_promote(version: int) -> bool:
    if gate_check(version):
        promote(version, stage="Production")
        return True
    print(f"[evaluate] v{version} did NOT pass the quality gate — staying in Staging")
    return False


if __name__ == "__main__":
    latest = list_versions()[-1]["version"]
    evaluate_and_promote(latest)
