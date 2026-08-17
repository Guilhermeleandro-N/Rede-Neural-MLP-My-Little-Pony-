from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import kagglehub
import pandas as pd
import os


# ==========================================
# 1. Carregamento do dataset
# ==========================================

path = kagglehub.dataset_download(
    "wenruliu/adult-income-dataset"
)

print("Dataset baixado em:")
print(path)

arquivo = os.path.join(
    path,
    "adult.csv"
)

df = pd.read_csv(
    arquivo,
    encoding="latin1"
)


# ==========================================
# 2. Tratamento dos valores ausentes
# ==========================================

df = df.replace(
    '?',
    pd.NA
)

print("Valores faltantes antes:")
print(df.isnull().sum())


for coluna in df.columns:

    if df[coluna].isnull().sum() > 0:

        df[coluna] = df[coluna].fillna(
            df[coluna].mode()[0]
        )


print("\nValores faltantes depois:")
print(df.isnull().sum())


# ==========================================
# 3. Separação entre X e y
# ==========================================

X = df.drop(
    'income',
    axis=1
)

y = df['income']


print("\nFormato de X:")
print(X.shape)

print("\nFormato de y:")
print(y.shape)


# ==========================================
# 4. One-Hot Encoding
# ==========================================

X = pd.get_dummies(
    X
)


print("\nFormato de X após One-Hot Encoding:")
print(X.shape)


print("\nPrimeiras colunas:")
print(X.columns[:20])


# ==========================================
# 5. Transformação da variável alvo
# ==========================================

y = y.str.strip()

y = y.map({
    '<=50K': 0,
    '>50K': 1
})


print("\nValores únicos de y:")
print(y.unique())


print("\nDistribuição de y:")
print(y.value_counts())


print("\nProporção das classes:")
print(
    y.value_counts(
        normalize=True
    ) * 100
)


# ==========================================
# 6. Divisão treino, validação e teste
# ==========================================

#
# Primeira divisão:
#
# 70% -> treino
# 30% -> conjunto temporário
#
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


#
# Segunda divisão:
#
# Os 30% restantes são divididos igualmente:
#
# 15% -> validação
# 15% -> teste
#
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nFormato dos dados:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_val:",
    y_val.shape
)

print(
    "y_test:",
    y_test.shape
)


print("\nDistribuição de y_train:")

print(
    y_train.value_counts(
        normalize=True
    ) * 100
)


print("\nDistribuição de y_val:")

print(
    y_val.value_counts(
        normalize=True
    ) * 100
)


print("\nDistribuição de y_test:")

print(
    y_test.value_counts(
        normalize=True
    ) * 100
)


# ==========================================
# 7. Padronização
# ==========================================

colunas_num = [
    'age',
    'fnlwgt',
    'educational-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]


scaler = StandardScaler()


#
# O scaler aprende média e desvio padrão
# SOMENTE com o conjunto de treino.
#
X_train[colunas_num] = scaler.fit_transform(
    X_train[colunas_num]
)


#
# A validação utiliza os mesmos parâmetros
# aprendidos no treino.
#
X_val[colunas_num] = scaler.transform(
    X_val[colunas_num]
)


#
# O teste também utiliza os mesmos parâmetros
# aprendidos no treino.
#
X_test[colunas_num] = scaler.transform(
    X_test[colunas_num]
)


print("\nDados numéricos após padronização:")

print(
    X_train[
        colunas_num
    ].head()
)


# ==========================================
# 8. Conversão dos dados para float
# ==========================================

X_train = X_train.astype(
    float
)

X_val = X_val.astype(
    float
)

X_test = X_test.astype(
    float
)


print("\nValores ausentes em X_train:")
print(
    X_train.isnull().sum().sum()
)

print("\nValores ausentes em X_val:")
print(
    X_val.isnull().sum().sum()
)

print("\nValores ausentes em X_test:")
print(
    X_test.isnull().sum().sum()
)


print("\nTipos de X_train:")
print(
    X_train.dtypes.value_counts()
)


# ==========================================
# 9. Conversão para NumPy
# ==========================================

X_train = X_train.to_numpy()

X_val = X_val.to_numpy()

X_test = X_test.to_numpy()


y_train = (
    y_train
    .to_numpy()
    .reshape(-1, 1)
)

y_val = (
    y_val
    .to_numpy()
    .reshape(-1, 1)
)

y_test = (
    y_test
    .to_numpy()
    .reshape(-1, 1)
)


print("\nShapes finais:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_val:",
    y_val.shape
)

print(
    "y_test:",
    y_test.shape
)