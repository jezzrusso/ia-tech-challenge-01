import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_dataset
from src.features.region_by_state import associate_and_aggregate_state_by_region
from src.features.challenge_age import get_next_challenge_age
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.charts import plot_all_histograms
from src.visualization.metadata import show_basic_infos

if __name__ == "__main__":
    # Carregando datasets
    main_df = load_dataset("data/processed/simple_clean/insurance.csv", sep=",")
    chance_of_survive_df = load_dataset("data/processed/simple_clean/nvsr_66_04.csv")
    state_by_region_df = load_dataset("data/processed/simple_clean/states_by_region.csv")
    income_df = load_dataset("data/processed/simple_clean/stateonline_13(Sheet1).csv")
    poverty_df = load_dataset("data/processed/simple_clean/state.csv", decimal=",")

    # Adicionando nome a cada dataframe
    main_df._name = "main_df"
    chance_of_survive_df._name = "chance_of_survive_df"
    state_by_region_df._name = "state_by_region_df"
    income_df._name = "income_df"
    poverty_df._name = "poverty_df"

    # Deixando todas as colunas com nome em minusculo
    main_df.columns = main_df.columns.str.lower()
    chance_of_survive_df.columns = chance_of_survive_df.columns.str.lower()
    state_by_region_df.columns = state_by_region_df.columns.str.lower()
    income_df.columns = income_df.columns.str.lower()
    poverty_df.columns = poverty_df.columns.str.lower()

    # Descrição básica dos dataframes
    show_basic_infos(main_df)
    show_basic_infos(chance_of_survive_df)
    show_basic_infos(state_by_region_df)
    show_basic_infos(income_df)
    show_basic_infos(poverty_df)

    # Garantindo que as colunas sejam tratadas como numericas
    main_df['age'] = main_df['age'].astype(int)
    main_df['charges'] = main_df['charges'].astype(float)
    main_df['bmi'] = main_df['bmi'].astype(float)

    chance_of_survive_df['age'] = chance_of_survive_df['age'].astype(int)

    # Analisando histogramas de todos os dfs
    plot_all_histograms(main_df)
    plot_all_histograms(chance_of_survive_df)
    plot_all_histograms(state_by_region_df)
    plot_all_histograms(income_df)
    plot_all_histograms(poverty_df)

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

    income_df["income"] = income_df["income"].str.replace('.', '', regex=False)  # remove separador de milhar
    income_df["income"] = income_df["income"].astype(int)

    income_by_region_df = associate_and_aggregate_state_by_region(income_df, state_by_region_df, "income")
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
    plt.title('Histograma da Idade')
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

    print(numerical_scaled_df.head().to_string(max_cols=None))

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

    correlation_matrix = merged_df.select_dtypes(include='number').corr(method="pearson")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
