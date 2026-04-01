"""Application settings configuration."""

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    openrouter_api_key: str = ""
    default_model: str = "google/gemini-3.1-flash-lite-preview"
    mlflow_experiment_name: str = "summit-sim"
    base_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    max_turns: int = Field(default=5, description="Maximum turns per scenario")
    ui_timeout: int = Field(
        default=300, description="Timeout in seconds for UI interactions (5 minutes)"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for LangGraph persistence",
    )
    mlflow_env: str = Field(
        default="local",
        description="Deployment environment for MLflow traces: local or prod",
    )
    image_generation_model: str = Field(
        default="google/gemini-3.1-flash-image-preview",
        description="OpenRouter model for scenario image generation",
    )
    image_generation_timeout: int = Field(
        default=120,
        description="Timeout in seconds for image generation",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
