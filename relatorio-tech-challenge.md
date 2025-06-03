# Relatório de Análise e Modelagem de Dados de Seguro Saúde

Este relatório apresenta uma análise exploratória e modelagem de dados para prever gastos com seguro saúde com base em características individuais.

## 1. Introdução

O objetivo deste estudo é explorar o dataset de seguros saúde e construir modelos preditivos para estimar os gastos (`charges`).

## 2. Exploração e Análise de Dados

O dataset `insurance.csv` foi carregado utilizando a biblioteca pandas.

- **Visualização das Primeiras Linhas:** A função `.head()` foi utilizada para inspecionar as primeiras entradas do dataset.
- **Dimensões do Dataset:** O dataset possui `1338` linhas e `7` colunas, conforme verificado por `.shape`.
- **Informações Gerais:** A função `.info()` revelou que o dataset contém colunas numéricas (`age`, `bmi`, `children`, `charges`) e categóricas (`sex`, `smoker`, `region`). Não há valores nulos em nenhuma coluna.
- **Verificação de Valores Nulos:** A contagem de valores nulos por coluna, utilizando `.isnull().sum()`, confirmou a ausência de dados faltantes.
- **Distribuição da Variável Target (`charges`):** A contagem de valores únicos para a coluna `charges` foi realizada, mostrando a diversidade dos gastos. Uma visualização da distribuição dos gastos foi gerada utilizando `seaborn.histplot`, indicando uma distribuição assimétrica, concentrada em valores mais baixos.
- **Distribuição de Gênero:** A contagem de valores para a coluna `sex` mostrou a distribuição de indivíduos por gênero (masculino e feminino).
- **Estatísticas Descritivas:** A função `.describe()` forneceu um resumo estatístico das colunas numéricas, incluindo média, mediana, desvio padrão, valores mínimos e máximos.
- **Histogramas:** Histogramas das colunas numéricas (`age`, `bmi`, `children`, `charges`) foram gerados para visualizar suas distribuições. A distribuição de `charges` foi plotada separadamente para melhor visualização.

## 3. Pré-processamento de Dados

Para preparar os dados para a modelagem, foram aplicados transformadores:

- **Separação de Features e Target:** As colunas de features (`X`) e a variável target (`y`, `charges`) foram separadas.
- **Definição de Colunas Categóricas e Numéricas:** As colunas foram categorizadas em `categorical_features` (`'sex', 'smoker', 'region'`) e `numeric_features` (`'age', 'bmi', 'children'`).
- **Transformadores:**
    - Um `StandardScaler` foi aplicado às colunas numéricas para padronizá-las.
    - Um `OneHotEncoder` com `drop='first'` foi aplicado às colunas categóricas para convertê-las em representação numérica, evitando a multicolinearidade.
- **Combinação de Transformadores:** Um `ColumnTransformer` foi utilizado para aplicar os transformadores apropriados a cada tipo de coluna.
- **Aplicação do Pré-processamento:** O `preprocessor` foi aplicado aos dados de features (`X`) para obter `X_processed`.

## 4. Análise de Correlação

Uma matriz de correlação foi gerada para visualizar a relação entre as variáveis após o pré-processamento. As colunas numéricas originais, as colunas one-hot encoded e a variável target (`charges`) foram incluídas. O heatmap mostrou a força e a direção das correlações. A correlação mais notável parece ser entre `smoker_yes` e `charges`.

## 5. Separação de Dados de Treino e Teste

Os dados pré-processados (`X_processed`) e a variável target (`y`) foram divididos em conjuntos de treino e teste utilizando `train_test_split` com um tamanho de teste de 20% e `random_state=42` para reprodutibilidade. Os conjuntos resultantes foram `X_train`, `X_test`, `y_train` e `y_test`. Nota-se que para a modelagem, `X_train` e `X_test` foram criados a partir de um DataFrame gerado com os dados pré-processados.

## 6. Modelagem Preditiva

Foram aplicados três modelos de regressão: Árvore de Decisão, Regressão Linear e Random Forest. As métricas Mean Absolute Error (MAE), Mean Squared Error (MSE) e R² Score foram utilizadas para avaliar o desempenho de cada modelo nos dados de teste.

### 6.1. Árvore de Decisão

Um `DecisionTreeRegressor` foi treinado nos dados de treino pré-processados (`X_train`, `y_train`).

- **Métricas de Avaliação:**
    - MAE: [Valor do MAE]
    - MSE: [Valor do MSE]
    - R²: [Valor do R²]

### 6.2. Regressão Linear

Um `LinearRegression` foi treinado nos dados de treino pré-processados (`X_train`, `y_train`).

- **Métricas de Avaliação:**
    - MSE: [Valor do MSE]
    - R²: [Valor do R²]

As primeiras 5 previsões (`y_pred`) foram comparadas com os valores reais (`y_test`) para fins de inspeção.

### 6.3. Random Forest Regressor

Um `RandomForestRegressor` com `n_estimators=100` e `random_state=42` foi treinado nos dados de treino pré-processados (`X_train`, `y_train`).

- **Métricas de Avaliação:**
    - MSE: [Valor do MSE]
    - R²: [Valor do R²]

As primeiras 5 previsões (`y_pred`) foram comparadas com os valores reais (`y_test`) para fins de inspeção.

## 7. Análise de Componentes Principais (PCA)

O PCA foi aplicado aos dados pré-processados (`X_processed`) para reduzir a dimensionalidade para 2 componentes.

- A variância explicada por cada componente e a variância total explicada foram calculadas.
- Um novo DataFrame (`df_pca`) foi criado com os componentes principais e a variável target.
- Foi observado que a variância total explicada por 2 componentes principais foi de [Valor da Variância Total]%

## 8. Conclusão

Com base nas métricas de avaliação, o **RandomForestRegressor obteve o melhor desempenho com um R² de [Melhor Valor do R²]**.

A aplicação do PCA não foi considerada benéfica para este dataset, pois a redução para 2 componentes resultou em uma perda significativa de variância explicada, e o dataset original já possuía um número limitado de features (7). Portanto, a modelagem foi realizada com as features pré-processadas completas.

O RandomForestRegressor parece ser o modelo mais adequado para prever os gastos com seguro saúde neste dataset.