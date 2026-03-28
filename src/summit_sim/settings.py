"""Application settings configuration."""

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    openrouter_api_key: str = ""
    default_model: str = "openai/gpt-4.1-nano"
    mlflow_experiment_name: str = "summit-sim"
    base_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    max_turns: int = Field(default=5, description="Maximum turns per scenario")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings instance

    """
    return settings


logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
