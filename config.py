from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class Settings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

    train_data_path: FilePath
    test_data_path: FilePath
    model_save_path: DirectoryPath
    model_name: str
    log_level: str


settings = Settings()
logger.remove()
logger.add(
    "app.log",
    rotation="1 day",
    retention="2 days",
    compression="zip",
    level=settings.log_level
)
