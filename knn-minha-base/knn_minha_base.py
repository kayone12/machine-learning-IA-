import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

pd.set_option("display.max_columns", None)



# leitura da base de dados 

df = pd.read_csv("cardekho_dataset.csv")

df = df.drop(columns=["Unnamed: 0"])

print(df.head())
print("\nInformações da base:")
print(df.info())

print(70*"-")


# estapa 1 

colunas_numericas = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats", "selling_price"]

for col in colunas_numericas:
    media = df[col].mean()
    df[col] = df[col].fillna(media)

print(df.isnull().sum())

print(70*"-")


# etapa 2

for col in colunas_numericas:
    minimo = df[col].min()
    maximo = df[col].max()
    df[col] = (df[col] - minimo) / (maximo - minimo)

print(df[colunas_numericas].head())

print(70*"-")


# etapa 3


df = df.drop(columns=["car_name", "model", "brand"])


mapa_seller = {"Individual": 0, "Dealer": 1, "Trustmark Dealer": 2}
df["seller_type"] = df["seller_type"].map(mapa_seller)

fuel_encoded = pd.get_dummies(df["fuel_type"], prefix="fuel")
transmissao_encoded = pd.get_dummies(df["transmission_type"], prefix="transmissao")

df_final = pd.concat(
    [df.drop(["fuel_type", "transmission_type"], axis=1),
     fuel_encoded, transmissao_encoded],
    axis=1
)

print(df_final.head())

print(70*"-")


# etapa 4

X = df_final.drop(columns=["seller_type"])
y = df_final["seller_type"]

print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")

print(70*"-")


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# KNN
print(" KNN ")

melhor_k = None
melhor_acuracia_knn = 0

for k in range(1, 11):

    modelo_knn = KNeighborsClassifier(n_neighbors=k)

    resultados = cross_val_score(
        modelo_knn,
        X,
        y,
        cv=kfold,
        scoring='accuracy'
    )

    media = resultados.mean()

    print(f"K = {k} -> Acurácia média: {media:.2%}")

    if media > melhor_acuracia_knn:
        melhor_acuracia_knn = media
        melhor_k = k


print("\n O melhor resultado do KNN foi :")
print(f"K = {melhor_k}")
print(f"Acurácia = {melhor_acuracia_knn:.2%}")



# NAIVE BAYES

print("\n NAIVE BAYES ")

modelo_nb = GaussianNB()

resultados_nb = cross_val_score(
    modelo_nb,
    X,
    y,
    cv=kfold,
    scoring='accuracy'
)

media_nb = resultados_nb.mean()

print("Acurácias dos folds:")
print(resultados_nb)

print(f"Acuracia media: {media_nb:.2%}")



# comparaçãop entre os algoritnos
print("\n COMPARAÇÃO FINAL ")

print(f"KNN (melhor K = {melhor_k}): {melhor_acuracia_knn :.2%}")
print(f"Naive Bayes: {media_nb :.2%}")

if melhor_acuracia_knn > media_nb:
    print(f"O KNN apresentou melhor por que deu  {melhor_acuracia_knn :.2%} % de acuracia")
elif media_nb > melhor_acuracia_knn:
    print("O Naive Bayes apresentou melhor resultado.")
else:
    print("Os dois algoritmos apresentaram o mesmo resultado.")
