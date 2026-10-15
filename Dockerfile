# --- Containerization stage --------------------------------------------
# This is what makes the model "runs the same everywhere": laptop, CI
# runner, and cloud (AWS ECS/EKS/SageMaker, GCP Cloud Run, etc).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake a trained + promoted model into the image at build time.
# In a real pipeline, this step would instead be "COPY the model artifact
# pulled from the registry" rather than training inside the image.
RUN python pipeline.py

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
