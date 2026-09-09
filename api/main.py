"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.predictor import Predictor
from api.schemas import ForecastRequest, ForecastResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor.load()
    yield


app = FastAPI(title="M4 Daily Forecasting API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": app.state.predictor.model_version}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        return app.state.predictor.predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc