"""Collection module for loading and processing data."""

import pandas as pd
from loguru import logger
from config.config import engine
from sqlalchemy import select


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    sign_df = pd.read_csv(file_path, sep=',')
    return sign_df


def load_data_from_db(schema):
    logger.info("Loading data from database")
    query = select(schema)
    return pd.read_sql(query, engine)
