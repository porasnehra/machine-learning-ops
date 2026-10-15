# Student Exam Score Predictor — a Live MLOps Pipeline

A small but *complete* MLOps pipeline: synthetic data → validation →
training with experiment tracking → a quality gate → a model registry →
a FastAPI serving layer → prediction logging + drift monitoring → CI/CD.

Every stage is a real, runnable Python file — nothing here is a slide,
it is meant to be executed.

## Quickstart

```bash
pip install -r requirements.txt

# Run the entire pipeline end-to-end in one command:
python pipeline.py

# Serve the production model over HTTP:
uvicorn api.app:app --reload --port 8000

# Try it:
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "study_hours": 6, "attendance_pct": 90, "sleep_hours": 7,
  "previous_score": 75, "extra_curricular": 1, "study_group": 1
}'

# Run tests (what CI runs):
pytest tests/ -v

# Build the container:
docker build -t score-predictor .
docker run -p 8000:8000 score-predictor
```

## Project layout

```
score-predictor-mlops/
├── data/
│   ├── generate_data.py      # Stage 1: Data ingestion (synthetic dataset)
│   ├── raw_students.csv      # generated on run
│   ├── train.csv / test.csv  # generated on run
├── src/
│   ├── data_prep.py           # Stage 2: Validation + train/test split
│   ├── train.py                # Stage 3: Training + experiment tracking
│   ├── evaluate.py             # Stage 4: Quality gate + promotion
│   ├── model_registry.py       # Versioned model storage (mini MLflow)
│   └── predict.py              # Shared inference logic (CLI + API use this)
├── api/
│   └── app.py                  # Stage 5: FastAPI serving layer
├── monitoring/
│   └── drift_check.py          # Stage 6: Prediction logging + drift detection
├── tests/
│   └── test_pipeline.py        # Stage 0: CI test suite
├── .github/workflows/ci.yml     # Stage 7: CI/CD pipeline definition
├── pipeline.py                  # Orchestrates every stage — run this
├── Dockerfile
└── requirements.txt
```

## What each stage maps to in "real" MLOps tooling

| This project           | Industry equivalent                          |
|-------------------------|-----------------------------------------------|
| `generate_data.py`       | Data lake / warehouse / feature store          |
| `data_prep.py` validation | Great Expectations, TFX Data Validation        |
| `train.py` experiment log | MLflow Tracking, Weights & Biases              |
| `model_registry.py`       | MLflow Model Registry, SageMaker Model Registry|
| `evaluate.py` gate         | CI quality gates, champion/challenger testing   |
| `api/app.py`               | SageMaker Endpoint, KServe, Vertex AI Endpoint  |
| `drift_check.py`           | Evidently AI, WhyLabs, SageMaker Model Monitor  |
| `.github/workflows/ci.yml` | Jenkins, GitLab CI, SageMaker Pipelines         |
| `Dockerfile`                | Same idea in every real deployment              |

See **MLOps_Explained.docx** for the full write-up of every concept this
project touches, plus the ones intentionally left out of this small demo
(feature stores, Kubernetes orchestration, A/B testing, etc.) for further
reading as you move into the Cloud/AWS/MLOps stage of your roadmap.
