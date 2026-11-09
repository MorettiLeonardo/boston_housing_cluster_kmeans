from pathlib import Path
from pickle import load

import pandas as pd

PASTA = Path(__file__).parent

# Carregar os modelos salvos no treinamento
modelo_normalizador = load(open(PASTA / 'modelo_normalizador.model', 'rb'))
modelo_kmeans = load(open(PASTA / 'modelo_kmeans.model', 'rb'))
info_colunas = load(open(PASTA / 'info_colunas.model', 'rb'))

colunas_originais = info_colunas['colunas_originais']
colunas_num = info_colunas['colunas_num']
colunas_cat = info_colunas['colunas_cat']
colunas_cat_normalizadas = info_colunas['colunas_cat_normalizadas']

DESCRICOES = {
    'CRIM': 'taxa de criminalidade per capita',
    'ZN': '% de terrenos residenciais com lotes acima de 25.000 pes2',
    'INDUS': '% de area de comercio nao varejista',
    'CHAS': 'margeia o rio Charles (1 = sim, 0 = nao)',
    'NOX': 'concentracao de oxidos nitricos',
    'RM': 'numero medio de comodos',
    'AGE': '% de imoveis construidos antes de 1940',
    'DIS': 'distancia ponderada aos centros de emprego',
    'RAD': 'indice de acessibilidade a rodovias (1 a 8 ou 24)',
    'TAX': 'imposto predial por US$ 10.000',
    'PTRATIO': 'razao aluno-professor',
    'B': 'indice B da populacao',
    'LSTAT': '% de populacao de baixa renda',
    'MEDV': 'valor mediano do imovel (US$ 1.000)',
}

# Residencia de exemplo, usada como valor padrao de cada campo
EXEMPLO = {
    'CRIM': 0.02731, 'ZN': 0, 'INDUS': 7.07, 'CHAS': 0, 'NOX': 0.469, 'RM': 6.421, 'AGE': 78.9,
    'DIS': 4.9671, 'RAD': 2, 'TAX': 242, 'PTRATIO': 17.8, 'B': 396.9, 'LSTAT': 9.14, 'MEDV': 21.6,
}

# RECEBER OS DADOS DA NOVA RESIDENCIA
print('Informe os dados da nova residencia (Enter usa o valor do exemplo entre colchetes):')
nova_residencia = {}
for coluna in colunas_originais:
    while True:
        try:
            texto = input(f'  {coluna} - {DESCRICOES.get(coluna, "")} [{EXEMPLO[coluna]}]: ').strip()
        except EOFError:
            texto = ''
        if texto == '':
            nova_residencia[coluna] = EXEMPLO[coluna]
            break
        try:
            nova_residencia[coluna] = float(texto.replace(',', '.'))
            break
        except ValueError:
            print('    Valor invalido, digite um numero.')

nova_residencia = pd.DataFrame([nova_residencia])

# NORMALIZAR OS DADOS DA NOVA RESIDENCIA
# Numericos: usar o modelo normalizador salvo e o metodo transform() (sem treinar de novo)
instancia_num = nova_residencia[colunas_num]
instancia_num_normalizada = modelo_normalizador.transform(instancia_num)
instancia_num_normalizada = pd.DataFrame(instancia_num_normalizada, columns=colunas_num)

# Categoricos: get_dummies e reposicao das colunas vistas no treinamento
instancia_cat = nova_residencia[colunas_cat].astype(int)
instancia_cat_normalizada = pd.get_dummies(instancia_cat, columns=colunas_cat, prefix=colunas_cat, dtype=int)
colunas_desconhecidas = set(instancia_cat_normalizada.columns) - set(colunas_cat_normalizadas)
if colunas_desconhecidas:
    print('Atencao: categoria(s) nao vista(s) no treinamento:', ', '.join(sorted(colunas_desconhecidas)))
instancia_cat_normalizada = instancia_cat_normalizada.reindex(columns=colunas_cat_normalizadas, fill_value=0)

instancia_normalizada = instancia_num_normalizada.join(instancia_cat_normalizada)

# INFERIR O GRUPO DA NOVA RESIDENCIA
grupo = modelo_kmeans.predict(instancia_normalizada)[0]
print(f'\nA nova residencia pertence ao grupo {grupo} (de {modelo_kmeans.n_clusters} grupos)')

# DESNORMALIZAR O CENTROIDE DO GRUPO INFERIDO
centroide = pd.DataFrame([modelo_kmeans.cluster_centers_[grupo]], columns=instancia_normalizada.columns)

# Numericos: metodo inverse_transform do modelo normalizador
centroide_num = modelo_normalizador.inverse_transform(centroide[colunas_num])
centroide_natural = pd.DataFrame(centroide_num, columns=colunas_num)

# Categoricos: no centroide cada coluna dummy guarda a proporcao da categoria no grupo,
# entao a categoria desnormalizada e a de maior proporcao (a mais frequente)
for coluna in colunas_cat:
    dummies = [c for c in colunas_cat_normalizadas if c.startswith(coluna + '_')]
    categoria = centroide[dummies].iloc[0].idxmax()
    centroide_natural[coluna] = int(categoria[len(coluna) + 1:])

centroide_natural = centroide_natural[colunas_originais]

print('\nDados desnormalizados do centroide do grupo', grupo, '(comparados com a nova residencia):')
comparacao = pd.DataFrame({
    'centroide': centroide_natural.iloc[0],
    'nova_residencia': nova_residencia[colunas_originais].iloc[0],
})
print(comparacao.round(4).to_string())
