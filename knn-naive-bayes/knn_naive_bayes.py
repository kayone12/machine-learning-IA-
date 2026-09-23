

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np


iris = load_iris()

X = iris.data
y = iris.target


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# knn 
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

    print(f"K = {k} -> Acurácia média: {media:.4f}")

    if media > melhor_acuracia_knn:
        melhor_acuracia_knn = media
        melhor_k = k


print("\n O melhor resultado do KNN foi :")
print(f"K = {melhor_k}")
print(f"Acurácia = {melhor_acuracia_knn:.4f}")



# naive bayes 
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

print(f"Acuracia media: {media_nb:.4f}")


# Comparação entre os algoritmos


print("\n COMPARAÇÃO FINAL ")

print(f"KNN (melhor K = {melhor_k}): {melhor_acuracia_knn:.4f}")
print(f"Naive Bayes: {media_nb:.4f}")

if melhor_acuracia_knn > media_nb:
    print(f"O KNN apresentou melhor por que deu % {melhor_acuracia_knn} de acuracia")
elif media_nb > melhor_acuracia_knn:
    print("O Naive Bayes apresentou melhor resultado.")
else:
    print("Os dois algoritmos apresentaram o mesmo resultado.")
