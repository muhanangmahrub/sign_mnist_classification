import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    sign_df = pd.read_csv(file_path, sep=',')
    return sign_df
