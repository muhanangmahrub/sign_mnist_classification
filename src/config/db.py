from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


class DbSettings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    db_conn_str: str
    train_table_name: str
    test_table_name: str


db_settings = DbSettings()


engine = create_engine(db_settings.db_conn_str)
