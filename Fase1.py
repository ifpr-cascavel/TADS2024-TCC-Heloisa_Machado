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
# # **Fase 1: Processamento dos Dados**
# Este arquivo realiza o pré-processamento dos dados, preparando-os para as fases subsequentes de modelagem e análise. Abaixo está um resumo das principais etapas e funcionalidades implementadas nesta etapa.
#
# ## **Etapas do Processamento**
# ### **1. Importação das Bibliotecas**
# As bibliotecas necessárias para o processamento dos dados são importadas no início do arquivo, como:
#
# - **pandas:** para manipulação de dados
# - **numpy:** para operações numéricas
# ### **2. Carregamento do Dataset**
# O arquivo carrega os dados a partir de um arquivo CSV utilizando pandas.
#
# ### **3. Transformação de Variáveis Categóricas**
# As variáveis categóricas presentes no dataset são convertidas em variáveis numéricas para garantir que os modelos de aprendizado de máquina possam interpretá-las corretamente. Isso é feito utilizando técnicas como a codificação One-Hot Encoding ou Label Encoding, dependendo da natureza da variável.
#
# ### **4. Tratamento de Valores Nulos**
# Os valores ausentes nas variáveis são tratados para evitar que afetem negativamente o modelo. Dependendo da estratégia adotada, os valores nulos podem ser preenchidos com a média, a mediana.
#
# ### **5. Salvamento dos Dados Processados**
# Após as transformações e tratamentos realizados, os dados processados são salvos em um novo arquivo para uso nas etapas seguintes do modelo. O arquivo gerado é armazenado no formato CSV.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('C:/Users/heloi/tcc/heart_2022_with_nans.csv')

# %%
data.head()
data.columns

# %%
linhas, colunas = data.shape

print(f"A base de dados possui {linhas} linhas e {colunas} colunas.")

# %% [markdown]
# ## Substituir valores nulos

# %%
# Substituir valores nulos nas variáveis numéricas pela média
data.fillna(data.select_dtypes(include='number').mean(), inplace=True)

# Substituir valores nulos nas variáveis categóricas pela moda
data.fillna(data.select_dtypes(include='object').mode().iloc[0], inplace=True)

# %% [markdown]
# ## Transformação de variáveis categóricas em numéricas

# %%
# Cópia do dataframe
data_copy = data.copy()
data_copy.head()

# %%
# Lista de colunas de Sim ou Não a serem mapeadas
yes_no_columns = [
    'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
    'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing',
    'BlindOrVisionDifficulty', 'DifficultyConcentrating',
    'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
    'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'HighRiskLastYear'
]

# Mapeando "Yes" para 1 e "No" para 0
data_copy[yes_no_columns] = data_copy[yes_no_columns].replace({'Yes': 1, 'No': 0})  #.astype(int) para converter para inteiro após remoção de valores nulos

# %%
# Dicionário de mapeamento coluna GeneralHealth
health_mapping = {
    'Excellent': 5,
    'Very good': 4,
    'Good': 3,
    'Fair': 2,
    'Poor': 1
}

data_copy['GeneralHealth'] = data_copy['GeneralHealth'].map(health_mapping)

# %%
# Mapeamento coluna Sex
sex_mapping = {
    'Male': 0,
    'Female': 1
}

data_copy['Sex'] = data_copy['Sex'].map(sex_mapping)

# %%
# Mapeamento coluna RemovedTeeth
removed_teeth_mapping = {
    'None of them': 0,
    '1 to 5': 1,
    '6 or more, but not all': 2,
    'All': 3
}

data_copy['RemovedTeeth'] = data_copy['RemovedTeeth'].map(removed_teeth_mapping)

# %%
# Mapeamento coluna AgeCategory
age_mapping = {
    'Age 18 to 24': 0,
    'Age 25 to 29': 1,
    'Age 30 to 34': 2,
    'Age 35 to 39': 3,
    'Age 40 to 44': 4,
    'Age 45 to 49': 5,
    'Age 50 to 54': 6,
    'Age 55 to 59': 7,
    'Age 60 to 64': 8,
    'Age 65 to 69': 9,
    'Age 70 to 74': 10,
    'Age 75 to 79': 11,
    'Age 80 or older': 12
}

data_copy['AgeCategory'] = data_copy['AgeCategory'].map(age_mapping)

# %%
# Mapeamento coluna TetanusLast10Tdap
tetanus_mapping = {
    'No, did not receive any tetanus shot in the past 10 years': 0,
    'Yes, received tetanus shot but not sure what type': 1,
    'Yes, received Tdap': 2,
    'Yes, received tetanus shot, but not Tdap': 3
}

data_copy['TetanusLast10Tdap'] = data_copy['TetanusLast10Tdap'].map(tetanus_mapping)

# %%
# Mapeamento da coluna LastCheckupTime
checkup_mapping = {
    "Within past year (anytime less than 12 months ago)": 0,
    "Within past 2 years (1 year but less than 2 years ago)": 1,
    "Within past 5 years (2 years but less than 5 years ago)": 2,
    "5 or more years ago": 3
}

# Aplicar o mapeamento à coluna LastCheckupTime
data_copy['LastCheckupTime'] = data_copy['LastCheckupTime'].map(checkup_mapping)

# %%
# Aplicar one-hot encoding
data_copy = pd.get_dummies(data_copy, columns=['SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'CovidPos', 'HadDiabetes'], drop_first=True, dtype='int')

# %%
data_copy.head()

# %%
if 'State' in data_copy.columns:
  data_copy = data_copy.drop('State', axis=1)

# %%
data_copy.to_csv('processed_data.csv', index=False)

# %%
data_copy.columns

# %%
# Contar o número de colunas
num_colunas = data_copy.shape[1]
num_colunas

# %%
data_copy.columns.shape[0]

# %%
data_copy.columns

# %%
# Variáveis que não foram transformadas
variaveis_nao_transformadas = [
    'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters',
    'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear'
]

# Verificar as colunas presentes na base de dados
colunas_presentes = data_copy.columns

# Verificar se as variáveis não transformadas estão na base de dados
variaveis_nao_transformadas_presentes = [
    col for col in variaveis_nao_transformadas if col in colunas_presentes
]

# Exibir as variáveis presentes na base de dados
print("Variáveis não transformadas presentes na base de dados:", variaveis_nao_transformadas_presentes)

# Verificar os valores únicos das variáveis não transformadas
for col in variaveis_nao_transformadas_presentes:
    print(f"Valores únicos da coluna '{col}':")
    print(data_copy[col].unique())
    print("-" * 50)


# %%
# ### FEATURE IMPORTANCE

import pandas as pd
import numpy as np

feature_names = ['Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
       'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth', 'HadAngina', 'HadStroke', 'HadAsthma',
       'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease',
       'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan',
       'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI',
       'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
       'TetanusLast10Tdap', 'HighRiskLastYear',
       'SmokerStatus_Current smoker - now smokes some days',
       'SmokerStatus_Former smoker', 'SmokerStatus_Never smoked',
       'ECigaretteUsage_Not at all (right now)',
       'ECigaretteUsage_Use them every day',
       'ECigaretteUsage_Use them some days', 'RaceEthnicityCategory_Hispanic',
       'RaceEthnicityCategory_Multiracial, Non-Hispanic',
       'RaceEthnicityCategory_Other race only, Non-Hispanic',
       'RaceEthnicityCategory_White only, Non-Hispanic',
       'CovidPos_Tested positive using home test without a health professional',
       'CovidPos_Yes', 'HadDiabetes_No, pre-diabetes or borderline diabetes',
       'HadDiabetes_Yes',
       'HadDiabetes_Yes, but only during pregnancy (female)']

# # Cria um DataFrame com as importâncias
df_importancias = pd.DataFrame({'feature': feature_names, 'importancia': importances})

# # Ordena o DataFrame pelas importâncias em ordem decrescente
df_importancias = df_importancias.sort_values('importancia', ascending=False)

# # Seleciona as 17 features mais importantes
top_17_features = df_importancias['feature'].head(10).tolist()

# Cria um novo DataFrame com apenas as features selecionadas
df_filtrado = data[top_10_features]

# Exibe as primeiras linhas do DataFrame filtrado para verificar
print(df_filtrado.head())

# Visualização das importâncias
plt.figure(figsize=(10, 6))
plt.barh(df_importancias['feature'].head(10), df_importancias['importancia'].head(10))
plt.xlabel('Importância')
plt.ylabel('Característica')
plt.title('Top 17 Características Mais Importantes')
plt.show()
