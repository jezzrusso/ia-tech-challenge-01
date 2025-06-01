# Tech Challenge: Modelo Preditivo de Custos Médicos

## Visão Geral
Este projeto faz parte de um desafio técnico para desenvolver um modelo preditivo de regressão que prevê os custos médicos individuais cobrados pelo seguro de saúde. O objetivo é construir um modelo confiável que utilize variáveis como idade, sexo, IMC, número de filhos, status de fumante, região e outras características para prever os encargos médicos. Os dados utilizados foram enriquecidos com informações adicionais (renda média por estado/região e chances de sobrevivência) para aprimorar as previsões.

## O Problema
O desafio proposto inclui:
- Exploração e análise descritiva dos dados.
- Pré-processamento, tratamento de dados ausentes e conversão de variáveis categóricas.
- Mesclagem com dados externos (renda e chances de sobrevivência).
- Treinamento de modelos de regressão (Linear Regression, Decision Tree, Random Forest, Gradient Boosting).
- Avaliação com métricas como MSE, RMSE, MAE e R².
- Visualizações comparando valores reais e preditos, análise de resíduos e distribuição.
- Criação de gráficos para apresentação.

## Metodologia

### Fonte de Dados
- `insurance.csv`: Dados sobre idade, sexo, IMC, filhos, fumante, região e encargos médicos.
- `nvsr_66_04.csv`: Tabelas do CDC com chances de sobrevivência por idade e sexo.
- `stateonline_13(Sheet1).csv`: Renda média por estado.
- `states_by_region.csv`: Mapeamento de estados para regiões.

### Etapas Realizadas
- Carregamento e exploração inicial dos dados com `pandas`.
- Análise descritiva, gráficos de distribuição e mapa de correlação.
- Padronização de colunas e tratamento de variáveis categóricas.
- Criação de features como `next_challenge_age` e `survival_chance`.
- Mesclagem com dados externos e codificação one-hot.
- Treinamento e avaliação de modelos:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Visualizações de comparações real vs predito e análise de resíduos.

## Como Executar
1. Instale as dependências:

pip install pandas numpy scikit-learn matplotlib seaborn

2. Execute o script principal

python main.py

3. Observe os resultados e gráficos gerados.
