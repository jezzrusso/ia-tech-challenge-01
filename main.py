import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.data.load_data import load_dataset
from src.features.region_by_state import associate_and_aggregate_state_by_region
from src.features.challenge_age import get_next_challenge_age
from src.visualization.charts import plot_exploratory_graphs
from src.visualization.metadata import show_basic_infos

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Carregando datasets
    main_df = load_dataset("data/processed/simple_clean/insurance.csv", sep=",")
    chance_of_survive_df = load_dataset("data/processed/simple_clean/nvsr_66_04.csv")
    state_by_region_df = load_dataset("data/processed/simple_clean/states_by_region.csv")
    income_df = load_dataset("data/processed/simple_clean/stateonline_13(Sheet1).csv")
    poverty_df = load_dataset("data/processed/simple_clean/state.csv", decimal=",")

    # Deixar colunas em minúsculo
    for df in [main_df, chance_of_survive_df, state_by_region_df, income_df, poverty_df]:
        df.columns = df.columns.str.lower()

    # Garantir tipos corretos
    main_df['age'] = main_df['age'].astype(int)
    main_df['charges'] = main_df['charges'].astype(float)
    main_df['bmi'] = main_df['bmi'].astype(float)
    chance_of_survive_df['age'] = chance_of_survive_df['age'].astype(int)

    # Converter smoker para numérico
    main_df['smoker'] = main_df['smoker'].map({'yes': 1, 'no': 0})

    # Remoção de duplicados do dataset principal
    main_df = main_df.drop_duplicates()

    # Adicionando nome a cada dataframe
    main_df._name = "main_df"
    chance_of_survive_df._name = "chance_of_survive_df"
    state_by_region_df._name = "state_by_region_df"
    income_df._name = "income_df"
    poverty_df._name = "poverty_df"

    # EDA antes da junção
    show_basic_infos(main_df)
    show_basic_infos(chance_of_survive_df)
    show_basic_infos(state_by_region_df)
    show_basic_infos(income_df)
    show_basic_infos(poverty_df)
    plot_exploratory_graphs(main_df)

    # MESCLAR DATASETS
    available_ages = chance_of_survive_df['age'].unique()
    available_ages.sort()
    main_df['next_challenge_age'] = main_df['age'].apply(lambda x: get_next_challenge_age(x, available_ages))

    chance_male = chance_of_survive_df[['age', 'male']].rename(columns={'male': 'survival_chance'}).copy()
    chance_male['sex'] = 'male'
    chance_female = chance_of_survive_df[['age', 'female']].rename(columns={'female': 'survival_chance'}).copy()
    chance_female['sex'] = 'female'
    chance_df = pd.concat([chance_male, chance_female], ignore_index=True)

    merged_df = pd.merge(main_df, chance_df, left_on=['next_challenge_age', 'sex'], right_on=['age', 'sex'], how='left')
    merged_df = merged_df.drop(columns=['age_y'], errors='ignore').rename(columns={'age_x': 'age'})
    merged_df['survival_chance'] = merged_df['survival_chance'].fillna(0)

    income_df["income"] = income_df["income"].str.replace('.', '', regex=False).astype(int)
    income_by_region_df = associate_and_aggregate_state_by_region(income_df, state_by_region_df, "income")
    merged_df = pd.merge(merged_df, income_by_region_df, on='region', how='left')

    # Encoding categóricos e conversão de booleanos
    merged_df = pd.get_dummies(merged_df, columns=['sex'], drop_first=True)
    merged_df = pd.get_dummies(merged_df, columns=['region'], drop_first=True)
    bool_cols = merged_df.select_dtypes(include='bool').columns
    merged_df[bool_cols] = merged_df[bool_cols].astype(int)

    # DEFINIR FEATURES E TARGET
    target = 'charges'
    feature_cols = merged_df.drop(columns=[target]).select_dtypes(include='number').columns.tolist()

    X = merged_df[feature_cols]
    y = merged_df[target]

    # DIVIDIR TREINO E TESTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ESCALAR APENAS AS FEATURES
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TREINAR O MODELO
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # PREDIÇÃO
    y_pred = model.predict(X_test_scaled)

    # MÉTRICAS DE AVALIAÇÃO
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Resultados do Modelo:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")

    plt.figure(figsize=(8, 6))
    sns.histplot(main_df['charges'], bins=30, kde=True)
    plt.title('Distribuição dos Encargos (charges)')
    plt.xlabel('Charges')
    plt.ylabel('Frequência')
    plt.show()

    input("Pressione Enter para continuar...")

    plt.figure(figsize=(8, 6))
    sns.histplot(main_df['charges'], bins=30, kde=True)
    plt.title('Distribuição dos Encargos (charges)')
    plt.xlabel('Charges')
    plt.ylabel('Frequência')
    plt.show()

    for col in ['smoker', 'region', 'sex']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=col, y='charges', data=main_df)
        plt.title(f'Boxplot de Charges por {col}')
        plt.show()

    for col in ['age', 'bmi']:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=col, y='charges', data=main_df)
        plt.title(f'{col} vs Charges')
        plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(main_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Mapa de Correlação')
    plt.show()

    for col in ['smoker', 'region', 'sex']:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=main_df)
        plt.title(f'Distribuição de {col}')
        plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', size='bmi', data=main_df, sizes=(20, 200))
    plt.title('Idade vs Charges (Colorido por Smoker e Tamanho pelo BMI)')
    plt.xlabel('Idade')
    plt.ylabel('Charges')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='age',
        y='charges',
        hue='smoker',  # cor = smoker (0 ou 1)
        size='bmi',  # tamanho = bmi
        style='sex',  # estilo de marcador = sexo
        data=main_df,
        sizes=(20, 200),
        legend='full'
    )
    plt.title('Idade vs Charges (Cor=Smoker, Tamanho=BMI, Marcador=Sexo)')
    plt.xlabel('Idade')
    plt.ylabel('Charges')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='age',
        y='charges',
        hue='smoker',  # cor = smoker (0 ou 1)
        size='bmi',  # tamanho = bmi
        style='region',  # estilo de marcador = region
        data=main_df,
        sizes=(20, 200),
        legend='full'
    )
    plt.title('Idade vs Charges (Cor=Smoker, Tamanho=BMI, Marcador=Region)')
    plt.xlabel('Idade')
    plt.ylabel('Charges')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()

    # Selecionar as features mais relevantes
    selected_features = ['age', 'bmi', 'smoker']

    X = merged_df[selected_features]
    y = merged_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Treinar modelo Random Forest
    model = RandomForestRegressor(
        n_estimators=100,  # Número de árvores
        max_depth=None,  # Sem limite de profundidade inicial
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Fazer predições
    y_pred = model.predict(X_test_scaled)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)



    print("\n📊 Resultados do Random Forest com 3 Features:")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Visualizar Real vs Predito
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Real vs Predito - Random Forest')
    plt.show()

    # Visualizar distribuição dos resíduos
    residuos = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuos, kde=True)
    plt.title('Distribuição dos Resíduos - Random Forest')
    plt.xlabel('Erro')
    plt.show()

    print("\n📊 Resultados do Decision Tree")
    model_dt = DecisionTreeRegressor(random_state=42)
    model_dt.fit(X_train_scaled, y_train)
    y_pred_dt = model_dt.predict(X_test_scaled)

    mse_dt = mean_squared_error(y_test, y_pred_dt)
    rmse_dt = mse_dt ** 0.5
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)

    print(f"MSE: {mse_dt:.2f}")
    print(f"RMSE: {rmse_dt:.2f}")
    print(f"MAE: {mae_dt:.2f}")
    print(f"R²: {r2_dt:.4f}")

    print("\n📊 Resultados do Gradient Boosting")
    model_gb = GradientBoostingRegressor(random_state=42)
    model_gb.fit(X_train_scaled, y_train)
    y_pred_gb = model_gb.predict(X_test_scaled)

    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = mse_gb ** 0.5
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    print(f"MSE: {mse_gb:.2f}")
    print(f"RMSE: {rmse_gb:.2f}")
    print(f"MAE: {mae_gb:.2f}")
    print(f"R²: {r2_gb:.4f}")