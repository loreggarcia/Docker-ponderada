"""Serve the previously trained monthly Bitcoin model without retraining it."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from bitcoin import forecast

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    app.state.artifact = None
    app.state.load_error = None
    try:
        app.state.artifact = forecast.load_artifact(path)
    except FileNotFoundError:
        app.state.load_error = "Modelo ausente. Execute o treinamento e reinicie a API."
        LOGGER.warning("Model artifact not found: %s", path)
    except Exception:
        app.state.load_error = "Modelo inválido ou incompatível. Treine novamente e reinicie a API."
        LOGGER.exception("Unable to load model artifact")
    yield


app = FastAPI(
    title="Previsão mensal de Bitcoin",
    description=(
        "Previsão experimental de BTC/USD a partir de dados históricos desde 2020. "
        "O modelo e o mês de referência são fixados durante o treinamento."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def artifact_for(request: Request) -> dict:
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error)
    return artifact


@app.get("/health", tags=["Serviço"])
def health(request: Request):
    """Return readiness; an unavailable model yields HTTP 503."""
    artifact = request.app.state.artifact
    if artifact is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "detail": request.app.state.load_error,
            },
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "target_month": artifact["target_month"],
    }


@app.get("/predict", tags=["Modelo"])
def predict(
    request: Request,
    target_month: str | None = Query(
        default=None,
        pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$",
        description="Mês suportado pelo artefato, no formato YYYY-MM. Omita para usar o padrão.",
    ),
):
    """Forecast one of the target months supported by the loaded artifact."""
    artifact = artifact_for(request)
    try:
        return forecast.predict(artifact, target_month=target_month)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/metrics", tags=["Modelo"])
def metrics(request: Request):
    """Temporal validation results, including the last-price baseline."""
    return artifact_for(request)["metrics"]


@app.get("/model", tags=["Modelo"])
def model(request: Request):
    """Data provenance, training cutoff, target month and model metadata."""
    return artifact_for(request)["metadata"]
