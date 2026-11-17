
import time
import numpy as np

from modelos import (
    KNN,
    BayesUnivariado,
    BayesMultivariado,
    PerceptronMulticlasse
)

from metricas import calcular_metricas

CLASSES = np.array([0, 1, 2, 3])
FOLDS = 5
SEMENTE = 42


def criar_folds(y):
    rng = np.random.default_rng(SEMENTE)
    folds = [[] for _ in range(FOLDS)]

    for classe in CLASSES:
        indices = np.where(y == classe)[0]
        rng.shuffle(indices)
        partes = np.array_split(indices, FOLDS)

        for i, parte in enumerate(partes):
            folds[i].extend(parte.tolist())

    return [np.array(sorted(fold), dtype=int) for fold in folds]


def preparar_dados(X_treino, X_teste):
    medianas = np.nanmedian(X_treino, axis=0)
    medianas = np.where(np.isfinite(medianas), medianas, 0)

    X_treino = np.where(np.isnan(X_treino), medianas, X_treino)
    X_teste = np.where(np.isnan(X_teste), medianas, X_teste)

    medias = np.mean(X_treino, axis=0)
    desvios = np.std(X_treino, axis=0)
    desvios = np.where(desvios == 0, 1, desvios)

    return (
        (X_treino - medias) / desvios,
        (X_teste - medias) / desvios
    )


def avaliar_modelo(nome, configuracao, criar_modelo, X, y, folds):
    resultados_folds = []
    matriz_total = np.zeros((4, 4), dtype=int)

    for numero_fold, indices_teste in enumerate(folds):
        indices_treino = np.setdiff1d(
            np.arange(len(y)),
            indices_teste
        )

        X_treino = X[indices_treino].copy()
        X_teste = X[indices_teste].copy()
        y_treino = y[indices_treino]
        y_teste = y[indices_teste]

        X_treino, X_teste = preparar_dados(X_treino, X_teste)
        modelo = criar_modelo()

        inicio = time.perf_counter()

        if isinstance(modelo, PerceptronMulticlasse):
            modelo.treinar(X_treino, y_treino, SEMENTE + numero_fold)
        else:
            modelo.treinar(X_treino, y_treino)

        tempo_treino = time.perf_counter() - inicio

        inicio = time.perf_counter()
        previsoes = modelo.prever(X_teste)
        tempo_teste = time.perf_counter() - inicio

        metricas = calcular_metricas(y_teste, previsoes)
        matriz_total += metricas["matriz"]

        resultados_folds.append({
            "acuracia": metricas["acuracia"],
            "precisao": metricas["precisao"],
            "recall": metricas["recall"],
            "especificidade": metricas["especificidade"],
            "f1": metricas["f1"],
            "tempo_treino": tempo_treino,
            "tempo_teste": tempo_teste
        })

    resultado = {
        "modelo": nome,
        "configuracao": configuracao,
        "matriz": matriz_total
    }

    for metrica in [
        "acuracia",
        "precisao",
        "recall",
        "especificidade",
        "f1",
        "tempo_treino",
        "tempo_teste"
    ]:
        valores = [item[metrica] for item in resultados_folds]
        resultado[metrica + "_media"] = np.mean(valores)
        resultado[metrica + "_desvio"] = np.std(valores, ddof=1)

    return resultado


def gerar_experimentos():
    experimentos = []

    for distancia in ["Euclidiana", "Manhattan"]:
        for k in [3, 5, 7]:
            experimentos.append((
                f"KNN {distancia}",
                f"k={k}",
                lambda k=k, distancia=distancia: KNN(k, distancia)
            ))

    experimentos.append((
        "Bayesiano Univariado",
        "Gaussiano",
        lambda: BayesUnivariado()
    ))

    experimentos.append((
        "Bayesiano Multivariado",
        "Gaussiano",
        lambda: BayesMultivariado()
    ))

    for taxa in [0.001, 0.01, 0.05, 0.1]:
        for epocas in [10, 50, 100, 200]:
            experimentos.append((
                "Perceptron",
                f"taxa={taxa}; epocas={epocas}",
                lambda taxa=taxa, epocas=epocas:
                    PerceptronMulticlasse(taxa, epocas)
            ))

    return experimentos
