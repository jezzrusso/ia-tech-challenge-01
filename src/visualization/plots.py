import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.title(f'Real vs Predito - {model_name}')
    plt.show()

def plot_residuals(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 4))
    sns.histplot(y_test - y_pred, kde=True)
    plt.title(f'Residuais - {model_name}')
    plt.xlabel('Erro')
    plt.show()

def plot_age_bmi_smoker_vs_charges(df, style_col=None):
    """
    Plota a relação entre idade, IMC e status de fumante com o valor de charges.
    - X: idade
    - Y: charges
    - Cor (hue): status de fumante
    - Tamanho do ponto: IMC
    - Estilo (style): coluna opcional (sexo, região, children)
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='age',
        y='charges',
        hue='smoker',
        size='bmi',
        style=style_col,
        data=df,
        sizes=(20, 200),
        legend='full'
    )
    title = f'Relação entre Idade, BMI e Smoker com Charges'
    if style_col:
        title += f' (Estilo: {style_col})'
    plt.title(title)
    plt.xlabel('Idade')
    plt.ylabel('Charges')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()


def plot_boxplot_with_outliers_count(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(f'Boxplot de {y_col} por {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # Contar outliers manualmente por categoria
    for category in df[x_col].unique():
        subset = df[df[x_col] == category][y_col]
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
        count_outliers = len(outliers)
        plt.text(category, subset.max(), f'Outliers: {count_outliers}',
                 horizontalalignment='center', color='red', fontsize=10)

    plt.show()