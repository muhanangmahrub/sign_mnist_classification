from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger
from sqlalchemy import create_engine


class Settings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8'
    )

    model_save_path: DirectoryPath
    model_name: str
    log_level: str
    db_conn_str: str
    train_table_name: str
    test_table_name: str


settings = Settings()
logger.remove()
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="2 days",
    compression="zip",
    level=settings.log_level
)

engine = create_engine(settings.db_conn_str)
