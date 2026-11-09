# KMeans - Boston Housing

Clusterização das residências da base Boston Housing com KMeans. O projeto tem dois módulos: o treinamento, que agrupa as residências e salva o modelo, e a inferência, que indica a qual grupo pertence uma nova residência.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `HousingData.csv` | Base de dados Boston Housing (506 residências, 14 colunas) |
| `treinamento.py` | Normaliza a base, escolhe o número de grupos e treina o modelo |
| `inferencia.py` | Recebe uma nova residência e informa o grupo dela |

## Requisitos

- Python 3.10 ou superior
- Bibliotecas: pandas, numpy, scipy, scikit-learn e matplotlib

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

## Como executar

Rode primeiro o treinamento, porque a inferência usa os modelos salvos por ele:

```bash
python treinamento.py
python inferencia.py
```

### Com Docker

Sem instalar Python, rodando a partir da pasta do projeto (no PowerShell):

```powershell
docker run --rm -it -v "${PWD}:/app" -w /app python:3.12-slim sh -c "pip install -q pandas numpy scipy scikit-learn matplotlib && python treinamento.py && python inferencia.py"
```

## Treinamento

1. **Valores ausentes:** a base tem campos vazios. Nas colunas numéricas eles recebem a mediana e nas categóricas, a moda.
2. **Normalização:**
   - Colunas quantitativas: `MinMaxScaler` (valores entre 0 e 1).
   - Colunas categóricas (`CHAS` e `RAD`): `get_dummies`, que cria uma coluna 0/1 para cada categoria.
3. **Número de grupos:** método do cotovelo. Para K de 1 a 10, calcula a distância média de cada residência ao centroide do seu grupo. O K escolhido é o ponto da curva mais distante da reta que liga o primeiro ao último ponto.
4. **Modelo final:** treina o KMeans com o K escolhido e salva os modelos.

Arquivos gerados:

| Arquivo | Conteúdo |
|---|---|
| `modelo_kmeans.model` | Modelo KMeans treinado |
| `modelo_normalizador.model` | Normalizador das colunas quantitativas |
| `info_colunas.model` | Colunas e valores de preenchimento usados no treinamento |
| `cotovelo.png` | Gráfico do método do cotovelo |

## Inferência

1. Pede os dados da nova residência, um campo por vez. Se o campo ficar vazio (só Enter), o script usa o valor de exemplo mostrado entre colchetes.
2. Normaliza os dados com os modelos salvos, sem treinar de novo.
3. Informa a qual grupo a residência pertence.
4. Mostra o centroide desse grupo desnormalizado, ao lado dos dados da nova residência:
   - Colunas quantitativas: `inverse_transform` do normalizador.
   - Colunas categóricas: a categoria mais frequente no grupo.

## Exemplo de resultado

Treinamento:

```
Numero otimo de clusters: 4
Instancias por cluster:
0    115
1    132
2    110
3    149
```

Inferência com a residência de exemplo:

```
A nova residencia pertence ao grupo 3 (de 4 grupos)

Dados desnormalizados do centroide do grupo 3 (comparados com a nova residencia):
         centroide  nova_residencia
CRIM        0.1497           0.0273
ZN         17.9799           0.0000
INDUS       6.3064           7.0700
CHAS        0.0000           0.0000
NOX         0.4751           0.4690
RM          6.5453           6.4210
AGE        55.6181          78.9000
DIS         4.9355           4.9671
RAD         3.0000           2.0000
TAX       292.1879         242.0000
PTRATIO    17.8805          17.8000
B         388.5424         396.9000
LSTAT       9.5407           9.1400
MEDV       26.3584          21.6000
```

## Colunas da base

| Coluna | Tipo | Descrição |
|---|---|---|
| CRIM | quantitativa | Taxa de criminalidade per capita |
| ZN | quantitativa | % de terrenos residenciais com lotes acima de 25.000 pés² |
| INDUS | quantitativa | % de área de comércio não varejista |
| CHAS | categórica | Margeia o rio Charles (1 = sim, 0 = não) |
| NOX | quantitativa | Concentração de óxidos nítricos |
| RM | quantitativa | Número médio de cômodos |
| AGE | quantitativa | % de imóveis construídos antes de 1940 |
| DIS | quantitativa | Distância ponderada aos centros de emprego |
| RAD | categórica | Índice de acessibilidade a rodovias (1 a 8 ou 24) |
| TAX | quantitativa | Imposto predial por US$ 10.000 |
| PTRATIO | quantitativa | Razão aluno-professor |
| B | quantitativa | Índice B da população |
| LSTAT | quantitativa | % de população de baixa renda |
| MEDV | quantitativa | Valor mediano do imóvel (US$ 1.000) |
