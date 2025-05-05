# Tech Challenge: Modelo Preditivo de Custos Médicos

## Visão Geral

Este projeto é parte de um desafio técnico para desenvolver um modelo preditivo de regressão que prevê os custos médicos individuais cobrados pelo seguro de saúde. O objetivo é criar um modelo confiável que utilize características como idade, gênero, IMC, número de filhos, status de fumante, região e outras variáveis relevantes para estimar os encargos médicos.

Os dados utilizados são de aproximadamente 2014 e foram enriquecidos para melhorar a qualidade das previsões.

---

## O Problema

O desafio consiste em:

- Explorar e analisar uma base de dados com variáveis como idade, gênero, IMC, filhos, fumante, região e encargos.
- Pré-processar os dados, tratando valores ausentes e convertendo variáveis categóricas.
- Construir um modelo de regressão (ex.: Regressão Linear, Árvores de Decisão) para prever os custos médicos.
- Treinar e avaliar o modelo, utilizando métricas estatísticas (p-value, intervalos de confiança).
- Apresentar resultados visuais (gráficos de previsões vs. valores reais) e um relatório com análise, insights e validação estatística.

---

## Metodologia

### 1. Fonte de Dados

**Base Principal**:  
Dados obtidos do Kaggle (`insurance.csv`), contendo informações sobre idade, gênero, IMC, filhos, fumante, região e encargos médicos.

**Enriquecimento**:

- **Census**: Dados de renda média por região foram adicionados, com a hipótese de que a renda média impacta os custos hospitalares devido à maior qualidade de vida em regiões mais ricas.
- **Life Tables do CDC**: Incorporação de chances de sobrevivência por idade e gênero, extraídas de tabelas de mortalidade do CDC (`nvsr_66_04.csv`), para contextualizar os riscos de saúde.

### 2. Tarefas Realizadas

#### Exploração de Dados:

- Carregamento e análise estatística descritiva dos dados.
- Visualização de distribuições (ex.: histogramas de idade, IMC, encargos).

#### Pré-processamento:

- Tratamento de valores ausentes (se necessário).
- Mesclagem dos dados do Kaggle com renda média (Census) e chances de sobrevivência (CDC), utilizando `pandas` para alinhar por idade, gênero e região.

#### Modelagem:

- A fazer

#### Treinamento e Avaliação:

- A fazer

#### Resultados:

- A fazer

