import pandas as pd

def show_basic_infos(df: pd.DataFrame, max_unique_values=10):
    print(f"\n============== Início de {df._name} ==============\n")

    pd.set_option('display.max_columns', None)  # Mostrar todas as colunas

    print("\n🔍 Primeiras 5 linhas do DataFrame:")
    print(df.head())

    print("\n📋 Informações gerais do DataFrame:")
    print(df.info())

    print("\n🧩 Tipos de dados por coluna:")
    print(df.dtypes)

    print("\n🚨 Quantidade de valores nulos por coluna:")
    print(df.isnull().sum())

    print("\n📊 Quantidade de valores únicos por coluna:")
    print(df.nunique())

    print("\n🔎 Análise de valores únicos (para colunas categóricas ou com poucos valores distintos):")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= max_unique_values:
            print(f"\nColuna: {col}")
            print(df[col].value_counts(dropna=False))

    print("\n📈 Estatísticas descritivas (somente para colunas numéricas):")
    print(df.describe())

    print("\n❓ Verificação de colunas com tipos mistos (dentro de colunas do tipo 'object'):")
    for col in df.select_dtypes(include='object').columns:
        tipos = df[col].map(type).value_counts()
        if len(tipos) > 1:
            print(f"\nColuna '{col}' contém múltiplos tipos de dados:")
            print(tipos)

    print(f"\n============== Fim de {df._name} ==============\n")
