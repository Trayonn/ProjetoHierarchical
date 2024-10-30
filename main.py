import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Criando a base de dados fictícia de jogadores e estatísticas
dados_jogadores = {
    'Jogador': ['João Silva', 'Pedro Santos', 'Lucas Ferreira', 
                'Carlos Lima', 'Rafael Souza', 'Marcos Reis'],
    'Gols': [12, 8, 5, 15, 3, 20],
    'Assistências': [7, 12, 4, 6, 9, 10],
    'Dribles Completos': [30, 20, 25, 35, 22, 40],
    'Interceptações': [12, 5, 20, 10, 18, 8],
    'Passes Completos': [800, 950, 600, 870, 900, 1000],
    'Distância Percorrida (km)': [10.2, 11.5, 9.7, 11.0, 10.8, 12.0]
}

# Criando o DataFrame com os dados fictícios
df_jogadores = pd.DataFrame(dados_jogadores)

# Separando os atributos  dos jogadores

X = df_jogadores.drop('Jogador', axis=1)

# Padronizando os dados para que todas as variáveis tenham a mesma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando o Hierarchical Clustering
linked = linkage(X_scaled, method='ward')

# dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linked, 
           labels=df_jogadores['Jogador'].values,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrograma de Jogadores de Futebol')
plt.xlabel('Jogadores')
plt.ylabel('Distância Euclidiana')
plt.show()