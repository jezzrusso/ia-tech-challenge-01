import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_dataset
from src.features.region_by_state import associate_and_aggregate_state_by_region
from src.features.challenge_age import get_next_challenge_age
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":


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
    main_df['charges'] = main_df['age'].astype(int)
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

    # lendo dataset de renda média
    # Carregando datasets
    state_by_region_df = load_dataset("data/processed/simple_clean/states_by_region.csv")
    income_df = load_dataset("data/processed/simple_clean/stateonline_13(Sheet1).csv")
    income_df["Income"] = income_df["Income"].str.replace('.', '', regex=False)  # remove separador de milhar
    income_df["Income"] = income_df["Income"].astype(int)

    income_by_region_df = associate_and_aggregate_state_by_region(income_df, state_by_region_df, "Income")
    print(income_by_region_df.head())
    merged_df = pd.merge(
        merged_df,
        income_by_region_df,
        left_on=['region'],
        right_on=['region'],
        how='left'
    )

    print(merged_df.head())
    print(merged_df.columns)
    print(merged_df.isnull().sum())

    # Exemplo com pandas
    merged_df['age'].hist(bins=20)
    plt.title('Histograma da Renda')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.show()

    # Ou com seaborn para um visual mais limpo
    sns.histplot(merged_df['age'], bins=20, kde=True)
    plt.title('Distribuição da Idade')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.show()

    # Seleciona só colunas numéricas
    numerical_df = merged_df.select_dtypes(include='number')
    # Padronize (média=0, desvio padrão=1)
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_df)

    # Crie um DataFrame com as variáveis normalizadas
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_df.columns)

    print(numerical_scaled_df.head())

    # Calcula correlação
    correlation_matrix = numerical_scaled_df.corr(method="kendall")

    # Plota o heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de Correlação entre Variáveis Numéricas')
    plt.show()

    sns.scatterplot(x=merged_df['age'], y=merged_df['survival_chance'])
    plt.title('Idade vs Chance de Sobrevivência')
    plt.xlabel('Idade')
    plt.ylabel('Chance de Sobrevivência')
    plt.show()