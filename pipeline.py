"""
====================================================================
 LIVE, RUNNABLE MLOPS PIPELINE — run this file top to bottom:

    python pipeline.py

 It chains every stage a real MLOps system goes through:

   1) Data Ingestion        data/generate_data.py
   2) Data Validation/Prep  src/data_prep.py
   3) Training              src/train.py        (+ experiment tracking)
   4) Evaluation / Gate     src/evaluate.py      (+ promote to Production)
   5) Model Registry        src/model_registry.py
   6) Serving (simulated)   src/predict.py       (a live inference call)
   7) Monitoring            monitoring/drift_check.py

 Run this whenever new data arrives or code changes — this whole file
 IS the "Continuous Training" pipeline that a scheduler (Airflow,
 GitHub Actions cron, SageMaker Pipelines) would trigger automatically.
====================================================================
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from data.generate_data import generate_dataset, RAW_DATA_PATH
from src.data_prep import prepare
from src.train import train
from src.evaluate import evaluate_and_promote
from src.predict import predict_one
from src.model_registry import list_versions
from monitoring.drift_check import log_prediction, check_drift, read_recent_predictions


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def run_pipeline():
    banner("STAGE 1/7 — DATA INGESTION")
    df = generate_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Ingested {len(df)} rows -> {RAW_DATA_PATH.name}")

    banner("STAGE 2/7 — DATA VALIDATION + PREPROCESSING")
    prepare()

    banner("STAGE 3/7 — TRAINING (+ experiment tracking)")
    version, metrics = train()

    banner("STAGE 4/7 — EVALUATION / QUALITY GATE + PROMOTION")
    promoted = evaluate_and_promote(version)

    banner("STAGE 5/7 — MODEL REGISTRY STATE")
    for v in list_versions():
        marker = " <-- current" if v["stage"] == "Production" else ""
        print(f"  v{v['version']:<3} stage={v['stage']:<10} r2={v['metrics']['r2']:<7} "
              f"mae={v['metrics']['mae']}{marker}")

    banner("STAGE 6/7 — SERVING: LIVE INFERENCE EXAMPLE")
    sample_students = [
        {"study_hours": 6.0, "attendance_pct": 92, "sleep_hours": 7.5,
         "previous_score": 78, "extra_curricular": 1, "study_group": 1},
        {"study_hours": 1.5, "attendance_pct": 55, "sleep_hours": 5.0,
         "previous_score": 40, "extra_curricular": 0, "study_group": 0},
        {"study_hours": 3.5, "attendance_pct": 75, "sleep_hours": 6.5,
         "previous_score": 60, "extra_curricular": 1, "study_group": 0},
    ]
    for student in sample_students:
        result = predict_one(student)
        log_prediction(student, result)
        print(f"  input={student}")
        print(f"    -> predicted_score={result['predicted_score']} "
              f"(served by model v{result['model_version']})\n")

    banner("STAGE 7/7 — MONITORING: DRIFT CHECK ON RECENT TRAFFIC")
    recent = read_recent_predictions()
    drift_report = check_drift(recent)
    print(drift_report)

    banner("PIPELINE COMPLETE")
    print(f"Production model: v{version}  |  promoted={promoted}  |  metrics={metrics}")


if __name__ == "__main__":
    run_pipeline()
