"""FastAPI service; model weights load lazily on first prediction."""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from llm_finetuning_and_serving import __version__
from llm_finetuning_and_serving.datasets import inspect
from llm_finetuning_and_serving.experiments import get_run, list_runs
from llm_finetuning_and_serving.inference import ModelManager
from llm_finetuning_and_serving.integrations.finetuning_regression_tester import (
    status_dict as regression_tester_status_dict,
)
from llm_finetuning_and_serving.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="LLM Fine-tuning and Serving API",
    version=__version__,
    description="Local banking support ticket model adaptation and evaluation",
)
origins = [
    value.strip()
    for value in os.getenv("LLM_FINETUNING_AND_SERVING_CORS_ORIGINS", "http://localhost:5173").split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
manager = ModelManager()


@app.get("/api/v1/health")
def health() -> dict:
    tester_status = regression_tester_status_dict()
    return {
        "status": "ok",
        "version": __version__,
        "mlx_loaded": bool(manager._loaded),
        "finetuning_regression_tester": tester_status,
        "aftertune": tester_status,
    }


@app.get("/api/v1/models")
def models() -> dict:
    return {
        "base": manager.model_id,
        "adapter": manager.current["adapter_id"],
        "adapter_configured": bool(manager.adapter_path),
        "adapter_status": manager.adapter_status(),
    }


@app.get("/api/v1/models/current")
def current_model() -> dict:
    return manager.current


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return manager.predict(request.text, request.model)
    except Exception as exc:
        raise HTTPException(
            503, detail={"code": "inference_unavailable", "message": str(exc)}
        ) from exc


@app.post("/api/v1/compare")
def compare(request: PredictionRequest) -> dict:
    predictions = {}
    for variant in ("base", "adapted"):
        try:
            predictions[variant] = manager.predict(request.text, variant).model_dump()
        except Exception as exc:
            predictions[variant] = {"error": str(exc)}
    return {"predictions": predictions, "sequential": True}


@app.get("/api/v1/experiments")
def experiments() -> list[dict]:
    return list_runs()


@app.get("/api/v1/experiments/{run_id}")
def experiment(run_id: str) -> dict:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(404, "Experiment not found")
    return record


@app.get("/api/v1/evaluations/{run_id}")
def evaluation(run_id: str) -> dict:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(404, "Experiment not found")
    artifacts = record.get("artifacts", {})
    artifact = artifacts.get("evaluation") or next(
        (path for key, path in reversed(list(artifacts.items())) if key.startswith("evaluation_")),
        None,
    )
    if not artifact or not Path(artifact).is_file():
        raise HTTPException(404, "Evaluation report not available")
    return json.loads(Path(artifact).read_text())


@app.get("/api/v1/evaluations/{run_id}/examples")
def examples(run_id: str) -> list[dict]:
    report = evaluation(run_id)
    return report.get("examples", [])


@app.get("/api/v1/comparisons/{run_id}")
def comparison(run_id: str) -> dict:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(404, "Experiment not found")
    artifact = record.get("artifacts", {}).get("comparison")
    if not artifact or not Path(artifact).is_file():
        raise HTTPException(404, "Paired comparison not available")
    return json.loads(Path(artifact).read_text())


@app.get("/api/v1/regressions/{run_id}")
def regressions(run_id: str) -> dict:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(404, "Experiment not found")
    artifact = record.get("artifacts", {}).get("finetuning_regression_tester") or record.get(
        "artifacts", {}
    ).get("aftertune")
    if not artifact or not Path(artifact).is_file():
        raise HTTPException(
            404, "No finetuning-regression-tester report imported for this experiment"
        )
    return json.loads(Path(artifact).read_text())


@app.get("/api/v1/datasets/summary")
def dataset_summary() -> dict:
    path = Path(os.getenv("LLM_FINETUNING_AND_SERVING_DATA", "data/banking77"))
    try:
        return inspect(path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(404, f"Dataset not prepared: {exc}") from exc
