# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **Fase 2: Divisão dos Dados**
# Este arquivo realiza a **divisão dos dados**, preparando-os para as fases subsequentes de modelagem e treinamento dos algoritmos de aprendizado de máquina. Abaixo está um resumo das principais etapas e funcionalidades implementadas nesta fase.
#
# ## **Etapas da Divisão dos Dados**
#
# ### **1. Importação das Bibliotecas**
# As bibliotecas necessárias para a criação e avaliação dos modelos são importadas, como:
# - `pandas` para manipulação de dados
# - `sklearn.model_selection` para a função train_test_split, que realiza a divisão dos dados
#
# ### **2. Definição das Variáveis Independentes e Dependentes**
# As variáveis são definidas para a divisão:
#
# - `X`: Contém as variáveis independentes, ou seja, todas as colunas exceto a variável alvo.
# - `y`: Contém a variável dependente, que neste caso é a coluna que indica se houve um ataque cardíaco `HadHeartAttack`.
#
# ### **3. Divisão dos Dados**
#  Os dados são divididos em conjuntos de treino e teste para avaliar o desempenho dos modelos. Esta divisão é realizada utilizando a função `train_test_split` do sklearn.
#
# - `Treino (X_train, y_train)`: Usado para treinar os modelos.
# - `Teste (X_test, y_test)`: Usado para avaliar a performance do modelo.
#
# ### **4. Normalização dos Dados**
# Após a divisão, é realizada a normalização dos dados, utilizando um dos seguintes métodos, conforme necessário para o modelo:
#
# - **MinMaxScaler:** Escala os dados para o intervalo entre 0 e 1, o que é útil para modelos que são sensíveis à escala.
# - **Z-score (StandardScaler):** Normaliza os dados para que tenham média 0 e desvio padrão 1.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('C:/Users/heloi/tcc/processed_data.csv')

# %%
data.head(10)

# %%
# Definir as variáveis independentes (X) e dependentes (y)
X = data.drop(columns=['HadHeartAttack'])  # Todas as colunas exceto a variável alvo
y = data['HadHeartAttack']  # A variável alvo

# %%
# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024, stratify=y)

# %%
# Inicializar o MinMaxScaler para normalizar os dados
scaler = MinMaxScaler()

# Ajustar o scaler no conjunto de treino e transformar os dados de treino
X_train_scaled = scaler.fit_transform(X_train)

# Transformar os dados de teste com o scaler ajustado aos dados de treino
X_test_scaled = scaler.transform(X_test)

# Exibir as formas dos conjuntos resultantes
print(f'Treinamento: {X_train_scaled.shape}, {y_train.shape}')
print(f'Teste: {X_test_scaled.shape}, {y_test.shape}')

# Salvar os dados divididos e normalizados
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), 'data/data_splits_normalized.pkl')
print("Dados divididos e normalizados salvos em .pkl")
# Salvar os dados de teste normalizados
joblib.dump((X_test_scaled, y_test), 'data/data_test_normalized.pkl')
print("Dados de teste salvos em data_test_normalized.pkl")

# %%
# Normalização com ZScore
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Exibir as formas dos conjuntos resultantes
print(f'Treinamento: {X_train_scaled.shape}, {y_train.shape}')
print(f'Teste: {X_test_scaled.shape}, {y_test.shape}')

# Salvar os dados divididos e normalizados (conjunto de treino e teste)
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), 'data/data_splits_normalized.pkl')
print("Dados divididos e normalizados salvos em data_splits_normalized.pkl")

# Salvar os dados de teste normalizados com Zscore (arquivo separado)
joblib.dump((X_test_scaled, y_test), 'data/data_test_normalized.pkl')
print("Dados de teste salvos em data_test_normalized.pkl")

# %%
sns.countplot(x=y_train)
plt.show()

# Ver proporção das classes
print(y_train.value_counts(normalize=True))

# %% [markdown]
# ### Divisão dos dados sem normalização

# %%
# Definir as variáveis independentes (X) e dependentes (y)
X = data.drop(columns=['HadHeartAttack'])  # Todas as colunas exceto a variável alvo
y = data['HadHeartAttack']  # A variável alvo

# %%
# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Exibir as formas dos conjuntos resultantes
print(f'Treinamento: {X_train.shape}, {y_train.shape}')
print(f'Teste: {X_test.shape}, {y_test.shape}')

# %%
# Salvar os dados divididos
joblib.dump((X_train, X_test, y_train, y_test), 'data/data_splits.pkl')
print("Dados divididos salvos em data_splits.pkl")

# %%
# Salvar os dados de teste
joblib.dump((X_test, y_test), 'data/data_test.pkl')
print("Dados de teste salvos em data_test.pkl")
