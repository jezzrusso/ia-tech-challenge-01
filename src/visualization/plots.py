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
