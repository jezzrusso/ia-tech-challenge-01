import pandas as pd
import numpy as np
from src.data.load_data import load_dataset

if __name__ == "__main__":
    def get_next_challenge_age(age, available_ages):
        """Retorna a próxima idade de desafio a partir de uma lista de idades disponíveis."""
        next_ages = available_ages[available_ages >= age]
        if len(next_ages) > 0:
            return next_ages.min()
        return np.nan  # Caso não haja idade maior (ex.: idade > máxima disponível)

    # Carregando datasets
    main_df = load_dataset("data/processed/simple_clean/insurance.csv", sep=",")
    chance_of_survive_df = load_dataset("data/processed/simple_clean/nvsr_66_04.csv")

    # Verificar colunas para depuração
    print("Colunas de main_df:", list(main_df.columns))
    print("Colunas de chance_of_survive_df:", list(chance_of_survive_df.columns))
    print("Valores únicos de sex em main_df:", main_df['sex'].value_counts())

    # Padronizar nomes de colunas
    chance_of_survive_df.columns = chance_of_survive_df.columns.str.lower()

    # Garantir que 'age' seja inteiro
    main_df['age'] = main_df['age'].astype(int)
    chance_of_survive_df['age'] = chance_of_survive_df['age'].astype(int)

    # Obter idades disponíveis em nvsr_66_04.csv
    available_ages = chance_of_survive_df['age'].unique()
    available_ages.sort()  # Ordenar para eficiência

    # Calcular a próxima idade de desafio para cada registro em main_df
    main_df['next_challenge_age'] = main_df['age'].apply(
        lambda x: get_next_challenge_age(x, available_ages)
    )

    # Transformar chance_of_survive_df para ter uma coluna 'sex' e 'survival_chance'
    chance_of_survive_df_male = chance_of_survive_df[['age', 'male']].copy()
    chance_of_survive_df_male['sex'] = 'male'
    chance_of_survive_df_male = chance_of_survive_df_male.rename(columns={'male': 'survival_chance'})

    chance_of_survive_df_female = chance_of_survive_df[['age', 'female']].copy()
    chance_of_survive_df_female['sex'] = 'female'
    chance_of_survive_df_female = chance_of_survive_df_female.rename(columns={'female': 'survival_chance'})

    # Concatenar as versões male e female
    chance_of_survive_df_transformed = pd.concat(
        [chance_of_survive_df_male, chance_of_survive_df_female], ignore_index=True
    )

    # Mesclar os datasets com base em 'next_challenge_age' (mapeado para 'age') e 'sex'
    merged_df = pd.merge(
        main_df,
        chance_of_survive_df_transformed,
        left_on=['next_challenge_age', 'sex'],
        right_on=['age', 'sex'],
        how='left'
    )

    # Remover a coluna 'age' redundante do chance_of_survive_df (se necessário)
    merged_df = merged_df.drop(columns=['age_y'], errors='ignore')
    merged_df = merged_df.rename(columns={'age_x': 'age'})

    # Tratar valores ausentes
    merged_df['survival_chance'] = merged_df['survival_chance'].fillna(0)

    # Exibir o resultado com todas as colunas
    print("Dataset mesclado com próxima idade de desafio:")
    print(merged_df.head())