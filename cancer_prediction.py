import pandas as pd   # Biblioteca para manipulação e análise de dados (DataFrames).
from sklearn.model_selection import train_test_split # Função para dividir o dataset em conjuntos de treino e teste. 
from sklearn.naive_bayes import GaussianNB  # Algoritmo Naive Bayes Gaussiano, usado para classificação.
from sklearn.metrics import accuracy_score # Função para calcular a precisão do modelo.

# Cabeçalho explicativo
"""
Projeto: Predição de Câncer de Pulmão
Modelo: Naive Bayes
Objetivo: Utilizar um modelo de Naive Bayes para prever a probabilidade de uma pessoa desenvolver câncer de pulmão com base em fatores como idade, gênero, hábitos de fumo, etc.
"""

# Dados de entrada
dados = {
    'Gender': ['M', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'], # Gênero: M (Masculino), F (Feminino)
    'Age': [65, 45, 50, 60, 48, 54, 67, 70, 58, 62], # Idade dos pacientes
    'Smoking': [2, 1, 2, 2, 1, 1, 2, 2, 1, 2], # Fuma: 1 (Não), 2 (Sim)
    'Yellow fingers': [2, 1, 2, 2, 1, 1, 2, 2, 1, 1], # Dedos amarelados: 1 (Não), 2 (Sim)
    'Anxiety': [1, 2, 2, 1, 1, 2, 2, 1, 2, 1], # Ansiedade: 1 (Não), 2 (Sim)
    'Peer_pressure': [2, 1, 1, 2, 2, 1, 2, 1, 1, 2], # Pressão de colegas: 1 (Não), 2 (Sim)
    'Chronic Disease': [1, 2, 2, 1, 1, 2, 2, 1, 2, 1], # Doença crônica: 1 (Não), 2 (Sim)
    'Fatigue': [2, 1, 2, 2, 1, 1, 2, 2, 1, 2], # Fadiga: 1 (Não), 2 (Sim)
    'Allergy': [1, 2, 2, 1, 1, 2, 1, 1, 2, 1], # Alergia: 1 (Não), 2 (Sim)
    'Wheezing': [2, 1, 2, 2, 1, 2, 2, 1, 2, 1], # Chiado no peito: 1 (Não), 2 (Sim)
    'Alcohol': [1, 2, 1, 2, 1, 2, 1, 2, 2, 1], # Consumo de álcool: 1 (Não), 2 (Sim)
    'Coughing': [2, 1, 2, 2, 1, 2, 1, 1, 2, 1], # Tosse: 1 (Não), 2 (Sim)
    'Shortness of Breath': [2, 1, 2, 2, 1, 2, 1, 1, 2, 1], # Falta de ar: 1 (Não), 2 (Sim)
    'Swallowing Difficulty': [1, 2, 2, 1, 1, 2, 1, 1, 2, 1], # Dificuldade para engolir: 1 (Não), 2 (Sim)
    'Chest pain': [2, 1, 2, 1, 2, 2, 1, 1, 2, 1], # Dor no peito: 1 (Não), 2 (Sim)
    'Lung Cancer': ['YES', 'NO', 'YES', 'YES', 'NO', 'NO', 'YES', 'NO', 'YES', 'YES']  # Diagnóstico de câncer de pulmão: YES (Sim), NO (Não)
}

# Criando o DataFrame
df = pd.DataFrame(dados)

# Convertendo 'Lung Cancer' em binário (YES=1, NO=0)
df['Lung Cancer'] = df['Lung Cancer'].map({'YES': 1, 'NO': 0})

# Variáveis de entrada (X) e alvo (y)
X = df.drop(columns=['Lung Cancer'])  
y = df['Lung Cancer']

# Convertendo as variáveis categóricas para numéricas
X = pd.get_dummies(X, drop_first=True)  

# Dividindo o conjunto de dados em treinamento e teste (80% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo Naive Bayes Gaussiano
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Calculando a acurácia do modelo
acuracia = accuracy_score(y_test, y_pred)
print('Acurácia:', acuracia)
