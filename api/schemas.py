"""Pydantic request/response models for the forecasting API."""

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    series_id: str
    history: list[float] = Field(..., min_length=1, description="chronologically ordered, most recent last")


class ForecastResponse(BaseModel):
    series_id: str
    point_forecast: list[float]
    conformal_forecast: list[float]
    horizon: int
    model_version: str