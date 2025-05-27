import pandas as pd
import torch
import numpy as np
from collection import load_data
from loguru import logger


def prepare_data(file_path: str):
    logger.info(f"Preparing data from {file_path}")
    data = load_data(file_path)
    data = convert_array(data)
    X, y = split_labels(data)
    X = reshape_data(X)
    X, y = convert_torch(X, y)
    return X, y


def split_labels(data: pd.DataFrame):
    logger.info("Splitting labels from data")
    X = data[:, 1:]/255
    y = data[:, 0]
    return X, y


def convert_array(data: pd.DataFrame):
    logger.info("Converting DataFrame to numpy array")
    data = np.array(data, dtype='float32')
    return data


def reshape_data(X: pd.DataFrame):
    logger.info("Reshaping data to fit model input")
    X = X.reshape(X.shape[0], * (28, 28, 1))
    return X


def convert_torch(X: pd.DataFrame, y: pd.DataFrame):
    logger.info("Converting data to PyTorch tensors")
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X, y
