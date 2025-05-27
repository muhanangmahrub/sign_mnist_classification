import pandas as pd
from loguru import logger

def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    sign_df = pd.read_csv(file_path, sep=',')
    return sign_df
