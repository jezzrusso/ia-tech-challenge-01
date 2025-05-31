import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_exploratory_graphs(df, bins=20, kde=True, cols_per_row=3, figsize=(15, 10)):
    """
    Plota visualizações exploratórias para o DataFrame:
    - Histogramas para todas as colunas numéricas
    - Gráficos de contagem para colunas categóricas
    - Mapa de correlação (se houver mais de uma coluna numérica)
    - Scatterplot de 'age' vs 'charges' (se ambas existirem)
    """
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # 1️⃣ Histogramas para colunas numéricas
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / cols_per_row)
    if n_cols > 0:
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(data=df, x=col, kde=kde, bins=bins, ax=axes[i])
            axes[i].set_title(f'Distribuição de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequência')

        # Remove eixos vazios se houver
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Nenhuma coluna numérica para plotar histogramas.")

    # 2️⃣ Gráficos de contagem para colunas categóricas
    if len(cat_cols) > 0:
        for col in cat_cols:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col, data=df)
            plt.title(f'Distribuição de {col}')
            plt.show()
    else:
        print("⚠️ Nenhuma coluna categórica para plotar contagens.")

    # 3️⃣ Mapa de correlação
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Mapa de Correlação entre Variáveis Numéricas')
        plt.show()
    else:
        print("⚠️ Não há colunas numéricas suficientes para mapa de correlação.")

def plot_scatterplot(df, x_col, y_col):
    """
    Plota um scatterplot (gráfico de dispersão) entre duas colunas.
    """
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=x_col, y=y_col, data=df)
        plt.title(f'{x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    else:
        print(f"⚠️ As colunas '{x_col}' e/ou '{y_col}' não foram encontradas no DataFrame.")

    input("⏸️ Pressione Enter para continuar...")