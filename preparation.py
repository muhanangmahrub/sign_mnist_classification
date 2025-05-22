from collection import load_data
import pandas as pd
import torch
import numpy as np

def prepare_data(file_path: str):
    data = load_data(file_path)
    data = convert_array(data)
    X, y = split_labels(data)
    X = reshape_data(X)
    X, y = convert_torch(X, y)
    return X, y

def split_labels(data: pd.DataFrame):
    X = data[:, 1:]/255
    y = data[:, 0]
    return X, y

def convert_array(data: pd.DataFrame):
    data = np.array(data, dtype='float32')
    return data

def reshape_data(X: pd.DataFrame):
    X = X.reshape(X.shape[0], * (28, 28, 1))
    return X

def convert_torch(X: pd.DataFrame, y: pd.DataFrame):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X, y