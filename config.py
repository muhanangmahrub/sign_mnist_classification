from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

    train_data_path: FilePath
    test_data_path: FilePath
    model_save_path: DirectoryPath
    model_name: str

settings = Settings()
