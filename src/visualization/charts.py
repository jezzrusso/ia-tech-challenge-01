import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_all_histograms(df, bins=20, kde=True, cols_per_row=3, figsize=(15, 10)):
    """
    Plota histogramas com seaborn para todas as colunas numéricas do DataFrame.

    Parâmetros:
    - df: DataFrame pandas.
    - bins: número de divisões para os histogramas.
    - kde: se True, mostra curva de densidade.
    - cols_per_row: número de gráficos por linha.
    - figsize: tamanho total da figura.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / cols_per_row)
    if n_rows > 0:
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