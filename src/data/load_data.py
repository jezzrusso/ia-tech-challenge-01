import pandas as pd

def load_dataset(path, sep=";", decimal=","):
    return pd.read_csv(path, sep=sep, decimal=decimal)

def load_insurance_data():
    return load_dataset("data/processed/simple_clean/insurance.csv", sep=",")

def load_survival_data():
    return load_dataset("data/processed/simple_clean/nvsr_66_04.csv")

def load_state_region_data():
    return load_dataset("data/processed/simple_clean/states_by_region.csv")

def load_income_data():
    return load_dataset("data/processed/simple_clean/stateonline_13(Sheet1).csv")

def load_poverty_data():
    return load_dataset("data/processed/simple_clean/state.csv", decimal=",")
