import pandas as pd
from src.features.challenge_age import get_next_challenge_age
from src.features.region_by_state import associate_and_aggregate_state_by_region

def prepare_dataframes(main_df, chance_df, state_region_df, income_df):
    # Deixar todas as colunas em minúsculo para padronização
    for df in [main_df, chance_df, state_region_df, income_df]:
        df.columns = df.columns.str.lower()

    # Normalização de colunas e tipos
    main_df['age'] = main_df['age'].astype(int)
    main_df['charges'] = main_df['charges'].astype(float)
    main_df['bmi'] = main_df['bmi'].astype(float)
    main_df['smoker'] = main_df['smoker'].map({'yes': 1, 'no': 0})
    main_df = main_df.drop_duplicates()

    # Mescla com chance_of_survive
    available_ages = chance_df['age'].unique()
    available_ages.sort()
    main_df = main_df.copy()
    main_df['next_challenge_age'] = main_df['age'].apply(lambda x: get_next_challenge_age(x, available_ages))


    chance_male = chance_df[['age', 'male']].rename(columns={'male': 'survival_chance'})
    chance_male['sex'] = 'male'
    chance_female = chance_df[['age', 'female']].rename(columns={'female': 'survival_chance'})
    chance_female['sex'] = 'female'
    chance_combined = pd.concat([chance_male, chance_female], ignore_index=True)

    merged_df = pd.merge(main_df, chance_combined, left_on=['next_challenge_age', 'sex'], right_on=['age', 'sex'], how='left')
    merged_df = merged_df.drop(columns=['age_y'], errors='ignore').rename(columns={'age_x': 'age'})
    merged_df['survival_chance'] = merged_df['survival_chance'].fillna(0)

    # Mescla com income por região
    income_df["income"] = income_df["income"].str.replace('.', '', regex=False).astype(int)
    income_by_region = associate_and_aggregate_state_by_region(income_df, state_region_df, "income")
    merged_df = pd.merge(merged_df, income_by_region, on='region', how='left')

    # Encoding e tratamento booleanos
    merged_df = pd.get_dummies(merged_df, columns=['sex'], drop_first=True)
    merged_df = pd.get_dummies(merged_df, columns=['region'], drop_first=True)
    bool_cols = merged_df.select_dtypes(include='bool').columns
    merged_df[bool_cols] = merged_df[bool_cols].astype(int)

    return merged_df
