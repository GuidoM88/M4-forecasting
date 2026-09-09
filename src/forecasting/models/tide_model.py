"""TiDE model wrapper: cross-learned training with per-series scaling."""

from __future__ import annotations

import numpy as np
from darts import TimeSeries
from darts.models import TiDEModel

from forecasting.config import settings


class TiDEForecaster:
    def __init__(
        self,
        input_chunk_length: int = settings.input_chunk_length,
        output_chunk_length: int = settings.output_chunk_length,
        n_epochs: int = settings.n_epochs,
        batch_size: int = settings.batch_size,
        random_state: int = settings.random_state,
        accelerator: str = "cpu",
    ) -> None:
        self.model = TiDEModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "devices": 1,
                "enable_progress_bar": True,
                "enable_checkpointing": False,
                "logger": False,
            },
        )

    @staticmethod
    def _scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        mean, std = float(values.mean()), float(values.std())
        std = std if std > 0 else 1.0
        return (values - mean) / std, mean, std

    def fit(self, series: list[np.ndarray], max_samples_per_ts: int | None = None) -> "TiDEForecaster":
        scaled = [self._scale(s)[0].astype(np.float32) for s in series]
        ts = [TimeSeries.from_values(s) for s in scaled]
        self.model.fit(series=ts, max_samples_per_ts=max_samples_per_ts)
        return self

    def predict_batch(self, contexts: list[np.ndarray], horizon: int) -> list[np.ndarray]:
        stats = [self._scale(c) for c in contexts]
        ts = [TimeSeries.from_values(s.astype(np.float32)) for s, _, _ in stats]
        forecasts = self.model.predict(n=horizon, series=ts)
        return [f.values().flatten() * std + mean for f, (_, mean, std) in zip(forecasts, stats)]

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "TiDEForecaster":
        instance = cls.__new__(cls)
        instance.model = TiDEModel.load(path)
        return instance