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
# # Fase 4 - Treinamento e Teste
#
# Este arquivo realiza o treinamento e a avaliação de dois modelos de classificação – Regressão Logística e Random Forest – utilizando dados de treino previamente balanceados com SMOTE ou ADASYN. Abaixo, estão as principais etapas executadas no script:
#
# ## **Etapas do Treinamento e Teste**
#
# ### **1. Carregamento dos Dados**  
# São carregados os dados de treino balanceados e normalizados, e também o conjunto de teste.
#
# ### **2. Treinamento dos Modelos**
# * **Regressão Logística**
# * **Random Forest**
# * **XGBoost**
#
# ### **3. Avaliação dos Modelos**
# * **Predições com o Conjunto de Teste:** Os modelos são utilizados para prever os rótulos do conjunto de teste. Os resultados são armazenados para posterior análise.
# * **Exibição dos Resultados:** Para cada modelo, são exibidas as métricas de desempenho, incluindo precisão, recall, acurácia e a matriz de confusão, facilitando a análise de desempenho.
#
# ### **4. Salvar os Modelos Treinados**
# * **Persistência dos Modelos:** Os modelos treinados são salvos para uso posterior. Eles são armazenados como ```log_model.pkl``` para o modelo de Regressão Logística, ```rf_model.pkl``` para o modelo de Random Forest e ```xgb_model``` pra o XGBoost.

# %%
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### Treino e teste sem normalização

# %%
# Carregar os dados 
X_resampled, y_resampled = joblib.load('data/data_resampled.pkl')

# Carregar os dados de teste
X_resampled, y_resampled = joblib.load('data/data_test.pkl')

# Treinamento com Regressão Logística
log_model = LogisticRegression(random_state=42)
log_model.fit(X_resampled, y_resampled)

rf_model = RandomForestClassifier(
    n_estimators=300,           
    criterion='entropy',       
    max_depth=20  ,           
    min_samples_split=10,       
    min_samples_leaf=5,         
    max_features='sqrt',       
    class_weight='balanced', 
    bootstrap=True,
    oob_score=True,
    random_state=5218,
    n_jobs=-1
)
rf_model.fit(X_resampled, y_resampled)

# %%
# Avaliação com o conjunto de teste
y_pred_log = log_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Exibir resultados
print("Resultados da Regressão Logística:")
print(classification_report(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))

print("\nResultados da Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# %%
# Salvar os modelos treinados
joblib.dump(log_model, 'data/log_model.pkl')
joblib.dump(rf_model, 'data/rf_model.pkl')

# %% [markdown]
# ### Treino e teste com normalização

# %%
# Carregar os dados de treino balanceados e normalizados
X_resampled, y_resampled = joblib.load('data/data_resampled_normalized.pkl')

# Carregar os dados de teste
X_test, y_test = joblib.load('data/data_test_normalized.pkl')

# %%
print(X_resampled_adasyn.shape)
print(y_resampled_adasyn.shape)

# %%
# Carregar os dados normalizados
X_resampled, y_resampled = joblib.load('data/data_resampled_normalized.pkl')
print("Dados carregados com sucesso!")

# Treinamento com Regressão Logística
log_model = LogisticRegression(random_state=42)
log_model.fit(X_resampled, y_resampled)  

# Treinamento com Random Forest
# Random params
rf_model = RandomForestClassifier(
    n_estimators=300,           
    criterion='entropy',       
    max_depth=20  ,           
    min_samples_split=10,       
    min_samples_leaf=5,         
    max_features='sqrt',       
    class_weight='balanced', 
    bootstrap=True,
    oob_score=True,
    random_state=5218,
    n_jobs=-1
)
rf_model.fit(X_resampled, y_resampled)

# Avaliação com o conjunto de teste (X_test, y_test)
y_pred_log = log_model.predict(X_test)  # Previsão com Regressão Logística
y_pred_rf = rf_model.predict(X_test)   # Previsão com Random Forest

# Exibir resultados para Regressão Logística
print("Resultados da Regressão Logística:")
print(classification_report(y_test, y_pred_log))  # Relatório de métricas
print("Matriz de Confusão (Regressão Logística):")
print(confusion_matrix(y_test, y_pred_log))  # Matriz de confusão

# Exibir resultados para Random Forest
print("\nResultados da Random Forest:")
print(classification_report(y_test, y_pred_rf))  # Relatório de métricas
print("Matriz de Confusão (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))  # Matriz de confusão


# %%
# Treinamento com Regressão Logística
log_model = LogisticRegression(random_state=42)
log_model.fit(X_resampled, y_resampled)


### Random params
rf_model = RandomForestClassifier(
    n_estimators=300,           
    criterion='entropy',       
    max_depth=20  ,           
    min_samples_split=10,       
    min_samples_leaf=5,         
    max_features='sqrt',       
    class_weight='balanced', 
    bootstrap=True,
    oob_score=True,
    random_state=5218,
    n_jobs=-1
)
rf_model.fit(X_resampled, y_resampled)

# %%
balanced_accuracy_score(y_test, y_pred_rf)

# %% [markdown]
# # TESTES

# %%
# Carregar os dados 
X_resampled, y_resampled = joblib.load('data/data_resampled_normalized.pkl')

# Carregar os dados de teste
X_resampled, y_resampled = joblib.load('data/data_test_normalized.pkl')
# Treinamento com Regressão Logística
log_model = LogisticRegression(random_state=42)
log_model.fit(X_resampled, y_resampled)

# %%
rf_model = RandomForestClassifier(
    n_estimators=300,
    criterion='entropy',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    oob_score=True,
    random_state=5218,
    n_jobs=-1
)

# %%
rf_model = RandomForestClassifier(
    n_estimators=300,                # número de árvores
    max_depth=20,                     # profundidade máxima das árvores
    min_samples_split=10,             # número mínimo de amostras para dividir um nó
    min_samples_leaf=5,               # número mínimo de amostras em um nó folha
    max_features='sqrt',             # número de características a serem usadas por árvore
    criterion='gini',             # critério de divisão das árvores
    class_weight='balanced',         # ponderação das classes (útil quando o conjunto de dados está desbalanceado)
    bootstrap=True,                  # não usar amostragem com reposição
    random_state=5218,                # definir seed para reprodutibilidade
    n_jobs=-1                         # utilizar todos os núcleos de CPU
)

# %%
# Validação cruzada
skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='f1')

print("F1 Scores (Random Forest):", scores.mean())

# %%
# Treinamento final do modelo Random Forest
rf_model.fit(X_resampled, y_resampled)

# Avaliação com o conjunto de teste
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# %%
# Exibindo resultados da Regressão Logística
print("Resultados da Regressão Logística:")
print(classification_report(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))

# Exibindo resultados do Random Forest
print("\nResultados da Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# %%
# Avaliação no conjunto de treino
y_pred_train_rf = rf_model.predict(X_resampled)
print("\nResultados no Conjunto de Treino (Random Forest):")
print(classification_report(y_resampled, y_pred_train_rf))
print(confusion_matrix(y_resampled, y_pred_train_rf))

# %% [markdown]
# # Treinamento com Xgboost

# %%
# !pip install xgboost

# %%
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# Carregar os dados de treino balanceados e normalizados
X_resampled_adasyn, y_resampled_adasyn = joblib.load('data/data_resampled_normalized_async.pkl')

# Carregar os dados de teste
X_test_scaled, y_test = joblib.load('data/data_test_normalized.pkl')

# Definir o peso para a classe minoritária (aproximadamente 84005/5022)
scale_pos_weight = len(y_resampled_adasyn[y_resampled_adasyn == 0]) / len(y_resampled_adasyn[y_resampled_adasyn == 1])

# Treinamento com XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

# Validação cruzada para avaliar o modelo
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(xgb_model, y_resampled_adasyn, y_resampled_adasyn, cv=skf, scoring='f1')
print("F1 Score médio (Validação Cruzada - XGBoost):", scores.mean())

# Treinamento final com os dados completos de treino
xgb_model.fit(X_resampled_adasyn, y_resampled_adasyn)

# Previsão no conjunto de teste (com ajuste de limiar)
y_pred_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.3  # Ajuste o limiar para aumentar o recall da classe 1
y_pred_adjusted = (y_pred_prob > threshold).astype(int)

# Avaliação do modelo
print("\nResultados no Conjunto de Teste (XGBoost - Threshold Ajustado):")
print(classification_report(y_test, y_pred_adjusted))
print(confusion_matrix(y_test, y_pred_adjusted))

# Exibição de matriz de confusão e relatório no conjunto de treino
y_pred_train = xgb_model.predict(X_resampled_adasyn)
print("\nResultados no Conjunto de Treino (XGBoost):")
print(classification_report(y_resampled_adasyn, y_pred_train))
print(confusion_matrix(y_resampled_adasyn, y_pred_train))

