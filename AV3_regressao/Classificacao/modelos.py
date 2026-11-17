
import numpy as np

CLASSES = np.array([0, 1, 2, 3])


class KNN:
    def __init__(self, k, distancia):
        self.k = k
        self.distancia = distancia

    def treinar(self, X, y):
        self.X_treino = X
        self.y_treino = y

    def prever(self, X):
        previsoes = []

        for amostra in X:
            diferencas = np.abs(self.X_treino - amostra)

            if self.distancia == "Euclidiana":
                distancias = np.sqrt(np.sum(diferencas ** 2, axis=1))
            else:
                distancias = np.sum(diferencas, axis=1)

            indices = np.argsort(distancias)[:self.k]
            votos = self.y_treino[indices]
            contagens = np.bincount(votos, minlength=len(CLASSES))
            previsoes.append(np.argmax(contagens))

        return np.array(previsoes)


class BayesUnivariado:
    def treinar(self, X, y):
        self.parametros = {}

        for classe in CLASSES:
            dados = X[y == classe]
            medias = np.mean(dados, axis=0)
            variancias = np.maximum(np.var(dados, axis=0), 1e-9)
            prior = len(dados) / len(y)

            self.parametros[classe] = (medias, variancias, prior)

    def prever(self, X):
        pontuacoes = []

        for classe in CLASSES:
            medias, variancias, prior = self.parametros[classe]

            log_probabilidades = (
                -0.5 * np.log(2 * np.pi * variancias)
                - ((X - medias) ** 2) / (2 * variancias)
            )

            pontuacoes.append(
                np.sum(log_probabilidades, axis=1) + np.log(prior)
            )

        return CLASSES[np.argmax(np.column_stack(pontuacoes), axis=1)]


class BayesMultivariado:
    def treinar(self, X, y):
        self.parametros = {}

        for classe in CLASSES:
            dados = X[y == classe]
            media = np.mean(dados, axis=0)
            covariancia = np.atleast_2d(np.cov(dados, rowvar=False))
            covariancia += np.eye(X.shape[1]) * 1e-6

            sinal, log_determinante = np.linalg.slogdet(covariancia)

            if sinal <= 0:
                covariancia += np.eye(X.shape[1]) * 1e-3
                sinal, log_determinante = np.linalg.slogdet(covariancia)

            inversa = np.linalg.pinv(covariancia)
            prior = len(dados) / len(y)

            self.parametros[classe] = (
                media,
                inversa,
                log_determinante,
                prior
            )

    def prever(self, X):
        pontuacoes = []

        for classe in CLASSES:
            media, inversa, log_determinante, prior = self.parametros[classe]
            diferencas = X - media

            distancia = np.einsum(
                "ij,jk,ik->i",
                diferencas,
                inversa,
                diferencas
            )

            log_probabilidade = -0.5 * (
                X.shape[1] * np.log(2 * np.pi)
                + log_determinante
                + distancia
            )

            pontuacoes.append(log_probabilidade + np.log(prior))

        return CLASSES[np.argmax(np.column_stack(pontuacoes), axis=1)]


class Perceptron:
    def __init__(self, taxa, epocas):
        self.taxa = taxa
        self.epocas = epocas

    def treinar(self, X, y, rng):
        self.pesos = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epocas):
            indices = rng.permutation(len(X))

            for indice in indices:
                entrada = X[indice]
                alvo = y[indice]
                saida = np.dot(entrada, self.pesos) + self.bias

                if alvo * saida <= 0:
                    self.pesos += self.taxa * alvo * entrada
                    self.bias += self.taxa * alvo

    def pontuar(self, X):
        return np.dot(X, self.pesos) + self.bias


class PerceptronMulticlasse:
    def __init__(self, taxa, epocas):
        self.taxa = taxa
        self.epocas = epocas
        self.modelos = []

    def treinar(self, X, y, semente):
        self.modelos = []

        for i, classe in enumerate(CLASSES):
            rng = np.random.default_rng(semente + i)
            y_binario = np.where(y == classe, 1, -1)

            modelo = Perceptron(self.taxa, self.epocas)
            modelo.treinar(X, y_binario, rng)
            self.modelos.append(modelo)

    def prever(self, X):
        pontuacoes = np.column_stack([
            modelo.pontuar(X) for modelo in self.modelos
        ])

        return CLASSES[np.argmax(pontuacoes, axis=1)]
