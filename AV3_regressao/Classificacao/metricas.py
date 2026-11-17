
import numpy as np

CLASSES = np.array([0, 1, 2, 3])


def calcular_metricas(y_real, y_previsto):
    matriz = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    for real, previsto in zip(y_real, y_previsto):
        matriz[int(real), int(previsto)] += 1

    acuracia = np.trace(matriz) / np.sum(matriz)

    precisao_lista = []
    recall_lista = []
    especificidade_lista = []
    f1_lista = []

    for i in range(len(CLASSES)):
        vp = matriz[i, i]
        fp = np.sum(matriz[:, i]) - vp
        fn = np.sum(matriz[i, :]) - vp
        vn = np.sum(matriz) - vp - fp - fn

        precisao = vp / (vp + fp) if vp + fp else 0
        recall = vp / (vp + fn) if vp + fn else 0
        especificidade = vn / (vn + fp) if vn + fp else 0

        f1 = (
            2 * precisao * recall / (precisao + recall)
            if precisao + recall else 0
        )

        precisao_lista.append(precisao)
        recall_lista.append(recall)
        especificidade_lista.append(especificidade)
        f1_lista.append(f1)

    return {
        "acuracia": acuracia,
        "precisao": np.mean(precisao_lista),
        "recall": np.mean(recall_lista),
        "especificidade": np.mean(especificidade_lista),
        "f1": np.mean(f1_lista),
        "matriz": matriz
    }
