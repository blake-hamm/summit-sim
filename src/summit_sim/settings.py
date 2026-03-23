"""Application settings configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    openrouter_api_key: str = ""
    default_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
