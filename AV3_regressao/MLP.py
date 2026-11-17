"""
MLP para regressao implementado manualmente (somente NumPy).

Componentes:
  - Funcoes de ativacao: relu, tanh, sigmoid (camadas ocultas); saida linear.
  - Forward / backpropagation escritos a mao (sem sklearn, sem pandas).
  - Perda: MSE.  Otimizador: gradiente descendente em mini-batches.
  - Metricas: MSE, RMSE, MAE, R2 e R2 ajustado.

Uso rapido:
    modelo = MLP(camadas_ocultas=[32, 16], ativacao="relu",
                 taxa_aprendizado=0.01, epocas=100)
    modelo.fit(X_treino, y_treino)
    y_prev = modelo.predict(X_teste)

Observacao: X deve chegar ja padronizado (Z-score) - a padronizacao e feita
fora da classe, por fold, como na AV2. O alvo y e padronizado internamente
(somente para treinar) e a previsao volta para a escala original.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Funcoes de ativacao: (f(z), f'(z) em funcao da saida a = f(z) ou de z)
# ----------------------------------------------------------------------------
def _relu(z):
    return np.maximum(0.0, z)


def _relu_deriv(z, a):
    return (z > 0).astype(z.dtype)


def _tanh(z):
    return np.tanh(z)


def _tanh_deriv(z, a):
    return 1.0 - a ** 2


def _sigmoid(z):
    # versao estavel numericamente
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def _sigmoid_deriv(z, a):
    return a * (1.0 - a)


ATIVACOES = {
    "relu": (_relu, _relu_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "sigmoid": (_sigmoid, _sigmoid_deriv),
}


# ----------------------------------------------------------------------------
# Metricas de regressao
# ----------------------------------------------------------------------------
def calcular_metricas(y_real, y_prev, n_atributos):
    """Retorna dict com MSE, RMSE, MAE, R2 e R2 ajustado."""
    y_real = np.asarray(y_real, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    n = len(y_real)

    erro = y_real - y_prev
    mse = float(np.mean(erro ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(erro)))

    ss_res = float(np.sum(erro ** 2))
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if n - n_atributos - 1 > 0 and not np.isnan(r2):
        r2_aj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_atributos - 1)
    else:
        r2_aj = np.nan

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "R2_aj": r2_aj}


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, camadas_ocultas=(32, 16), ativacao="relu",
                 taxa_aprendizado=0.01, epocas=100, tamanho_lote=64,
                 semente=42):
        if ativacao not in ATIVACOES:
            raise ValueError(f"Ativacao invalida: {ativacao}")
        self.camadas_ocultas = list(camadas_ocultas)
        self.ativacao = ativacao
        self.lr = taxa_aprendizado
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.semente = semente
        self.divergiu = False
        self.historico = {"treino": [], "validacao": []}

    # ---------------- inicializacao ----------------
    def _inicializar(self, n_entrada, rng):
        tamanhos = [n_entrada] + self.camadas_ocultas + [1]
        self.W, self.b = [], []
        for i in range(len(tamanhos) - 1):
            fan_in, fan_out = tamanhos[i], tamanhos[i + 1]
            ultima = i == len(tamanhos) - 2
            if self.ativacao == "relu" and not ultima:
                escala = np.sqrt(2.0 / fan_in)                  # He
            else:
                escala = np.sqrt(2.0 / (fan_in + fan_out))      # Xavier
            self.W.append(rng.normal(0.0, escala, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    @property
    def n_parametros(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    # ---------------- forward ----------------
    def _forward(self, X, guardar=False):
        f, _ = ATIVACOES[self.ativacao]
        a = X
        zs, as_ = [], [X]
        for i in range(len(self.W) - 1):           # camadas ocultas
            z = a @ self.W[i] + self.b[i]
            a = f(z)
            if guardar:
                zs.append(z)
                as_.append(a)
        saida = a @ self.W[-1] + self.b[-1]        # saida linear
        if guardar:
            return saida, zs, as_
        return saida

    # ---------------- backward ----------------
    def _gradientes(self, X, y):
        """Gradientes do MSE em relacao a todos os pesos (backprop)."""
        _, df = ATIVACOES[self.ativacao]
        m = X.shape[0]
        saida, zs, as_ = self._forward(X, guardar=True)

        delta = (2.0 / m) * (saida - y.reshape(-1, 1))       # dL/dsaida
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = as_[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                # usa os pesos ANTES da atualizacao
                delta = (delta @ self.W[i].T) * df(zs[i - 1], as_[i])
        return gW, gb

    # ---------------- treino ----------------
    def fit(self, X, y, X_val=None, y_val=None):
        """Treina com gradiente descendente em mini-batches.
        Se X_val/y_val forem passados, guarda a curva de aprendizado."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.semente)
        self._inicializar(X.shape[1], rng)

        # padroniza o alvo apenas para treinar (estabiliza o gradiente)
        self.y_media = y.mean()
        self.y_desvio = y.std() if y.std() > 0 else 1.0
        yn = (y - self.y_media) / self.y_desvio
        if y_val is not None:
            yn_val = (np.asarray(y_val, dtype=float) - self.y_media) / self.y_desvio

        n = X.shape[0]
        self.divergiu = False
        self.historico = {"treino": [], "validacao": []}

        for _ in range(self.epocas):
            ordem = rng.permutation(n)
            for ini in range(0, n, self.tamanho_lote):
                idx = ordem[ini:ini + self.tamanho_lote]
                gW, gb = self._gradientes(X[idx], yn[idx])
                for i in range(len(self.W)):
                    self.W[i] -= self.lr * gW[i]
                    self.b[i] -= self.lr * gb[i]

            if X_val is not None:
                self.historico["treino"].append(
                    float(np.mean((self._forward(X).ravel() - yn) ** 2)))
                self.historico["validacao"].append(
                    float(np.mean((self._forward(np.asarray(X_val, dtype=float)).ravel()
                                   - yn_val) ** 2)))

            # deteccao de divergencia (pesos viram inf/nan)
            if not np.isfinite(self.W[-1]).all():
                self.divergiu = True
                break
        return self

    # ---------------- predicao ----------------
    def predict(self, X):
        saida = self._forward(np.asarray(X, dtype=float)).ravel()
        return saida * self.y_desvio + self.y_media
