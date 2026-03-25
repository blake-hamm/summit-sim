"""Application settings configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    openrouter_api_key: str = ""
    default_model: str = "google/gemini-3.1-flash-lite-preview"
    mlflow_experiment_name: str = "summit-sim"
    base_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
