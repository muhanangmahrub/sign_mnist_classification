"""Collection module for loading and processing data."""

import pandas as pd
from loguru import logger


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
