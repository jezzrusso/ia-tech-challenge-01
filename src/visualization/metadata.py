import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_basic_infos(df: pd.DataFrame, max_unique_values=10):
    print(f"\n============== Início de {df._name} ==============\n")

    pd.set_option('display.max_columns', None)  # Mostrar todas as colunas

    print("\nPrimeiras 5 linhas do DataFrame:")
    print(df.head())

    print("\nInformações gerais do DataFrame:")
    print(df.info())

    print("\nTipos de dados por coluna:")
    print(df.dtypes)

    print("\nQuantidade de valores nulos por coluna:")
    print(df.isnull().sum())

    print("\nQuantidade de valores únicos por coluna:")
    print(df.nunique())

    print("\nAnálise de valores únicos (para colunas categóricas ou com poucos valores distintos):")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= max_unique_values:
            print(f"\nColuna: {col}")
            print(df[col].value_counts(dropna=False))

    print("\nEstatísticas descritivas (somente para colunas numéricas):")
    print(df.describe())

    print("\nVerificação de colunas com tipos mistos (dentro de colunas do tipo 'object'):")
    for col in df.select_dtypes(include='object').columns:
        tipos = df[col].map(type).value_counts()
        if len(tipos) > 1:
            print(f"\nColuna '{col}' contém múltiplos tipos de dados:")
            print(tipos)

    print("\nQuantidade de linhas duplicadas no DataFrame:")
    print(df.duplicated().sum())

    print("\nColunas com baixa variância (apenas um valor distinto):")
    low_variance = df.columns[df.nunique() == 1]
    if len(low_variance) > 0:
        print(low_variance.tolist())
    else:
        print("Nenhuma coluna com baixa variância.")

    print(f"\n============== Fim de {df._name} ==============\n")

    input("Pressione Enter para continuar...")
