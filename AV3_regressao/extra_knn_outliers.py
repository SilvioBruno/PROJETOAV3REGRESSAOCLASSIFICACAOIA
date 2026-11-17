"""Apoio a analise do Z-score: onde o KNN erra mais com Z-score vs Min-Max?
Mostra, por fold, o maior erro absoluto e quantas previsoes erram mais de 30 pontos."""
import numpy as np
from experimentos_mlp import carregar_dados, criar_folds, normalizar, KNNVetorizado, FOLDS

X, y, _ = carregar_dados()
folds = criar_folds(np.arange(len(y)))
total = {"zscore": 0, "minmax": 0}
print("fold | norm    | RMSE  | MAE   | erro maximo | erros > 30")
for i in range(FOLDS):
    te = folds[i]
    tr = np.concatenate([folds[j] for j in range(FOLDS) if j != i])
    for norm in ["zscore", "minmax"]:
        a, b = normalizar(X[tr], X[te], norm)
        e = np.abs(KNNVetorizado(5, "euclidiana").fit(a, y[tr]).predict(b) - y[te])
        total[norm] += int((e > 30).sum())
        print(f"{i + 1:>4} | {norm:<7} | {np.sqrt(np.mean(e ** 2)):.2f}  | {e.mean():.2f}  | "
              f"{e.max():>11.1f} | {int((e > 30).sum()):>9}")
print("Total de erros > 30 pontos nos 5 folds:", total)
