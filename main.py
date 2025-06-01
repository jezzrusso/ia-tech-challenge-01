from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import numpy as np
import statsmodels.api as sm

from src.data.load_data import load_insurance_data, load_survival_data, load_state_region_data, load_income_data, load_poverty_data
from src.features.feature_engineering import prepare_dataframes
from src.visualization.metadata import show_basic_infos
from src.visualization.charts import plot_exploratory_graphs
from src.models.train import get_models, split_and_scale, train_model
from src.visualization.plots import plot_predictions, plot_residuals

if __name__ == "__main__":
    # Carregar datasets
    main_df = load_insurance_data()
    chance_df = load_survival_data()
    state_region_df = load_state_region_data()
    income_df = load_income_data()
    poverty_df = load_poverty_data()

    main_df._name = "main_df"
    chance_df._name = "chance_of_survive_df"
    state_region_df._name = "state_by_region_df"
    income_df._name = "income_df"
    poverty_df._name = "poverty_df"

    show_basic_infos(main_df)
    plot_exploratory_graphs(main_df)

    merged_df = prepare_dataframes(main_df, chance_df, state_region_df, income_df)
    merged_df._name = "merged_df"
    show_basic_infos(merged_df)
    plot_exploratory_graphs(merged_df)

    selected_features = ['age', 'bmi', 'smoker']
    X, y = merged_df[selected_features], merged_df['charges']

    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    for name, model in get_models().items():
        result = train_model(model, X_train, X_test, y_train, y_test)
        print(f"\nResultados do {name}:")
        print(f"MSE: {result['mse']:.2f}")
        print(f"RMSE: {result['rmse']:.2f}")
        print(f"MAE: {result['mae']:.2f}")
        print(f"R²: {result['r2']:.4f}")
        plot_predictions(y_test, result['y_pred'], name)
        plot_residuals(y_test, result['y_pred'], name)

        # Intervalo de confiança via bootstrap para R²
        print(f"Intervalo de confiança para R² (bootstrap) - {name}:")
        r2_scores = []
        for _ in range(1000):
            X_resampled, y_resampled = resample(X_test, y_test)
            y_pred_resampled = model.predict(X_resampled)
            r2_scores.append(r2_score(y_resampled, y_pred_resampled))
        r2_mean = np.mean(r2_scores)
        r2_ci = np.percentile(r2_scores, [2.5, 97.5])
        print(f"R² médio: {r2_mean:.4f}")
        print(f"Intervalo de confiança 95%: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

        # Validação estatística com p-values e intervalos (apenas para Linear Regression)
        if name == 'Linear Regression':
            print("\nValidação estatística para Regressão Linear:")
            X_train_const = sm.add_constant(X_train)
            ols_model = sm.OLS(y_train, X_train_const).fit()
            print(ols_model.summary())

    # Avaliar KNN separado
    print("\nResultados do KNN Regressor")
    model_knn = KNeighborsRegressor(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)

    mse_knn = mean_squared_error(y_test, y_pred_knn)
    rmse_knn = mse_knn ** 0.5
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    print(f"MSE: {mse_knn:.2f}")
    print(f"RMSE: {rmse_knn:.2f}")
    print(f"MAE: {mae_knn:.2f}")
    print(f"R²: {r2_knn:.4f}")
    plot_predictions(y_test, y_pred_knn, "KNN Regressor")
    plot_residuals(y_test, y_pred_knn, "KNN Regressor")

    # Intervalo de confiança via bootstrap para R² do KNN
    print("Intervalo de confiança para R² (bootstrap) - KNN Regressor:")
    r2_scores_knn = []
    for _ in range(1000):
        X_resampled, y_resampled = resample(X_test, y_test)
        y_pred_resampled = model_knn.predict(X_resampled)
        r2_scores_knn.append(r2_score(y_resampled, y_pred_resampled))
    r2_mean_knn = np.mean(r2_scores_knn)
    r2_ci_knn = np.percentile(r2_scores_knn, [2.5, 97.5])
    print(f"R² médio: {r2_mean_knn:.4f}")
    print(f"Intervalo de confiança 95%: [{r2_ci_knn[0]:.4f}, {r2_ci_knn[1]:.4f}]")
