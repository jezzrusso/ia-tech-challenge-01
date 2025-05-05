import pandas as pd

def load_dataset(path, sep=";", decimal=","):
    return pd.read_csv(path, sep=sep, decimal=decimal)