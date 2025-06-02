from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class LoggerSettings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    log_level: str


def configure_logging(log_level: str):
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="2 days",
        compression="zip",
        level=log_level
    )


configure_logging(LoggerSettings().log_level)
