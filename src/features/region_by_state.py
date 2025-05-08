import pandas as pd

def associate_and_aggregate_state_by_region(data_df, region_df, value_column, agg_func='median'):

    def standardize_state_column(df):
        # Procurar coluna que corresponda a 'state' (ignora maiúsculas/minúsculas)
        state_col = next((col for col in df.columns if col.lower() == 'state'), None)
        if state_col is None:
            raise ValueError("Nenhuma coluna 'State' (ou variante) encontrada no DataFrame.")
        # Criar uma cópia do DataFrame para evitar SettingWithCopyWarning
        df = df.copy()
        # Renomear para 'State'
        df.rename(columns={state_col: 'State'}, inplace=True)
        return df

    data_df = standardize_state_column(data_df)
    region_df = standardize_state_column(region_df)

    # Mesclar os DataFrames usando a coluna 'State'
    merged_df = pd.merge(data_df, region_df, on='State', how='left')

    # Verificar valores ausentes
    if merged_df.isnull().sum().sum() > 0:
        print("Atenção: Há valores ausentes no dataset mesclado:")
        print(merged_df.isnull().sum())

    # Calcular a estatística desejada por região
    result = merged_df.groupby('region')[value_column].agg(agg_func).reset_index()

    # Renomear a coluna de resultado para maior clareza
    result.columns = ['region', f'{agg_func}_{value_column}']

    return result