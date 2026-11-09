import math
from pathlib import Path
from pickle import dump

import matplotlib
matplotlib.use('Agg')  # apenas salva o grafico em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans

PASTA = Path(__file__).parent

# Colunas categoricas da base Boston Housing:
#   CHAS -> 1 se o lote margeia o rio Charles, 0 caso contrario
#   RAD  -> indice (codigo) de acessibilidade a rodovias radiais: 1 a 8 e 24
# Todas as demais colunas sao quantitativas
COLUNAS_CAT = ['CHAS', 'RAD']

dados = pd.read_csv(PASTA / 'HousingData.csv')

# Separar as categoricas das numericas
dados_num = dados.drop(columns=COLUNAS_CAT)
dados_cat = dados[COLUNAS_CAT]

# A base tem valores ausentes (NA): preencher as numericas com a mediana
# e as categoricas com a moda. Os valores sao salvos para uso na inferencia
valores_ausentes_num = dados_num.median()
valores_ausentes_cat = dados_cat.mode().iloc[0].astype(int)
dados_num = dados_num.fillna(valores_ausentes_num)
dados_cat = dados_cat.fillna(valores_ausentes_cat).astype(int)

# NORMALIZAR DADOS NUMERICOS
# Instanciar o normalizador
normalizador = preprocessing.MinMaxScaler()
# Treinar o modelo normalizador a partir dos dados numericos
modelo_normalizador = normalizador.fit(dados_num)
# Normalizar os dados numericos
dados_num_normalizados = modelo_normalizador.transform(dados_num)
# Recompor o dataframe com os dados numericos normalizados, atribuindo os rotulos de colunas originais
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

# NORMALIZAR DADOS CATEGORICOS
dados_cat_normalizados = pd.get_dummies(dados_cat, columns=COLUNAS_CAT, prefix=COLUNAS_CAT, dummy_na=False, dtype=int)

# Juntar numericos e categoricos normalizados
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)

# METODO DO COTOVELO (DISTANCIAS)
# Para cada K, calcular a distorcao: media das distancias de cada instancia ao seu centroide
K = range(1, 11)
distortions = []
for k in K:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10).fit(dados_normalizados)
    distortions.append(
        sum(np.min(cdist(dados_normalizados, modelo.cluster_centers_, 'euclidean'), axis=1)) / dados_normalizados.shape[0]
    )

# O numero otimo de clusters e o ponto da curva mais distante da reta
# que liga o primeiro ponto (x0, y0) ao ultimo ponto (xn, yn)
x0, y0 = K[0], distortions[0]
xn, yn = K[-1], distortions[-1]
distancias = []
for i in range(len(distortions)):
    x, y = K[i], distortions[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

numero_clusters_otimo = K[distancias.index(max(distancias))]
print('Numero otimo de clusters:', numero_clusters_otimo)

# Salvar o grafico do cotovelo
fig, ax = plt.subplots()
ax.plot(K, distortions, marker='o')
ax.plot([x0, xn], [y0, yn], linestyle='--', color='gray')
ax.axvline(numero_clusters_otimo, linestyle=':', color='red', label=f'K otimo = {numero_clusters_otimo}')
ax.set(xlabel='K (numero de clusters)', ylabel='Distorcao (distancia media ao centroide)', title='Metodo do cotovelo')
ax.legend()
fig.savefig(PASTA / 'cotovelo.png', dpi=120)

# TREINAR O MODELO FINAL
modelo_kmeans = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10).fit(dados_normalizados)
print('Instancias por cluster:')
print(pd.Series(modelo_kmeans.labels_).value_counts().sort_index().to_string())

# SALVAR OS MODELOS para uso posterior na inferencia
dump(modelo_normalizador, open(PASTA / 'modelo_normalizador.model', 'wb'))
dump(modelo_kmeans, open(PASTA / 'modelo_kmeans.model', 'wb'))
# Informacoes necessarias para normalizar e desnormalizar uma nova instancia
info_colunas = {
    'colunas_originais': list(dados.columns),
    'colunas_num': list(dados_num.columns),
    'colunas_cat': COLUNAS_CAT,
    'colunas_cat_normalizadas': list(dados_cat_normalizados.columns),
    'valores_ausentes_num': valores_ausentes_num.to_dict(),
    'valores_ausentes_cat': valores_ausentes_cat.to_dict(),
}
dump(info_colunas, open(PASTA / 'info_colunas.model', 'wb'))
print('Modelos salvos em', PASTA)
