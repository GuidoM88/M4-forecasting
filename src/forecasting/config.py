"""Application settings loaded from environment variables or defaults."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    data_dir: Path = PROJECT_ROOT / "data" / "M4"
    train_file: str = "Daily-train.csv"
    test_file: str = "Daily-test.csv"
    results_dir: Path = PROJECT_ROOT / "results"

    frequency: str = "Daily"
    forecast_length: int = 14
    context_length: int = 90
    season_length: int = 7

    n_series_calib: int = 500
    n_series_test: int = 1000

    input_chunk_length: int = 28
    output_chunk_length: int = 14
    n_epochs: int = 20
    batch_size: int = 256
    random_state: int = 42
    train_series_sample: int = 1000
    max_samples_per_ts: int = 20

    cost_underage: float = 2.0
    cost_overage: float = 1.0
    coverage_level: float = 0.90

    mlflow_tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow_experiment_name: str = "m4-daily-tide"
    mlflow_registered_model_name: str = "tide-m4-daily"

    @property
    def min_series_length(self) -> int:
        # buffer ensures both context and target windows fit within the series
        return self.context_length + self.forecast_length + 2

    @property
    def critical_ratio(self) -> float:
        # newsvendor critical fractile: cu / (cu + co)
        return self.cost_underage / (self.cost_underage + self.cost_overage)

    model_config = SettingsConfigDict(env_prefix="M4_")


settings = Settings()