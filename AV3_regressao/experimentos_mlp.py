"""
AV3 - Regressao com MLP (cpu_small, alvo = usr)

Reaproveita a organizacao experimental da AV2:
  * mesmos folds (validacao cruzada 5-fold, semente 42, funcao criar_folds);
  * normalizacao Z-score calculada SOMENTE com o treino de cada fold;
  * tempo de treino e de teste medidos por fold (time.perf_counter);
  * media +- desvio padrao (np.nanmean / np.nanstd) entre os folds.

Experimentos:
  E1 topologias | E2 taxa de aprendizado | E3 epocas | E4 ativacao x taxa
  E5 configuracao combinada + selecao (melhor desempenho x melhor equilibrio)
  E6 impacto do Z-score (MLP, KNN e regressao linear)
  Figuras: curvas de aprendizado, previsto x real, desempenho x custo.

Uso:  python experimentos_mlp.py          (gera pastas resultados/ e figuras/)
"""

import os

# 1 thread de BLAS: tempos mais estaveis/reprodutiveis para matrizes pequenas
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MLP import MLP, calcular_metricas

PASTA = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(PASTA, "cpu_small.csv")
PASTA_RES = os.path.join(PASTA, "resultados")
PASTA_FIG = os.path.join(PASTA, "figuras")
COLUNA_ALVO = "usr"
FOLDS = 5
SEMENTE = 42
K = 5

# configuracao-base (os experimentos variam um fator por vez a partir dela)
BASE = dict(camadas=[32, 16], ativacao="relu", lr=0.05, epocas=100)


# ----------------------------------------------------------------------------
# Dados e folds (iguais aos da AV2)
# ----------------------------------------------------------------------------
def carregar_dados(caminho=CAMINHO_CSV, alvo=COLUNA_ALVO):
    data = np.genfromtxt(caminho, delimiter=",", names=True,
                         encoding="utf-8", dtype=float, ndmin=1)
    nomes = list(data.dtype.names)
    atributos = [n for n in nomes if n != alvo]
    X = np.column_stack([data[n] for n in atributos]).astype(float)
    y = np.asarray(data[alvo], dtype=float)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("O CSV contem valores ausentes ou invalidos.")
    return X, y, atributos


def criar_folds(indices, quantidade=FOLDS):          # identica a da AV2
    rng = np.random.default_rng(SEMENTE)
    indices = rng.permutation(indices)
    return np.array_split(indices, quantidade)


# ----------------------------------------------------------------------------
# Normalizacao (parametros estimados no treino do fold)
# ----------------------------------------------------------------------------
def normalizar(X_tr, X_te, tipo):
    if tipo == "zscore":
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd[sd == 0] = 1
        return (X_tr - mu) / sd, (X_te - mu) / sd
    if tipo == "minmax":
        mn, mx = X_tr.min(axis=0), X_tr.max(axis=0)
        amp = mx - mn
        amp[amp == 0] = 1
        return (X_tr - mn) / amp, (X_te - mn) / amp
    if tipo == "robusta":                              # mediana / IQR
        med = np.median(X_tr, axis=0)
        iqr = np.percentile(X_tr, 75, axis=0) - np.percentile(X_tr, 25, axis=0)
        iqr[iqr == 0] = 1
        return (X_tr - med) / iqr, (X_te - med) / iqr
    if tipo == "log_zscore":                           # log(1+x) e depois Z-score
        return normalizar(np.log1p(X_tr), np.log1p(X_te), "zscore")
    return X_tr, X_te                                  # "nenhuma"


# ----------------------------------------------------------------------------
# Modelos de referencia (AV2): KNN (versao vetorizada) e regressao linear
# ----------------------------------------------------------------------------
class KNNVetorizado:
    """Mesma regra do KNN da AV2 (media dos K vizinhos), calculada em blocos
    com NumPy. Resultado equivalente ao das classes KNN_Euclidiana /
    KNN_Manhattan, so que muito mais rapido."""

    def __init__(self, k=5, metrica="euclidiana"):
        self.k, self.metrica = k, metrica

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        return self

    def predict(self, X_te, bloco=128):
        X_te = np.asarray(X_te, float)
        saida = np.empty(len(X_te))
        for ini in range(0, len(X_te), bloco):
            d = X_te[ini:ini + bloco, None, :] - self.X[None, :, :]
            if self.metrica == "euclidiana":
                dist = np.sqrt(np.sum(d ** 2, axis=2))
            else:
                dist = np.sum(np.abs(d), axis=2)
            viz = np.argpartition(dist, self.k, axis=1)[:, :self.k]
            saida[ini:ini + bloco] = self.y[viz].mean(axis=1)
        return saida


class RegressaoLinear:
    """Regressao linear multipla (minimos quadrados, solucao fechada)."""

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.coef


# ----------------------------------------------------------------------------
# Validacao cruzada generica
# ----------------------------------------------------------------------------
CHAVES = ["MSE", "RMSE", "MAE", "R2", "R2_aj", "R2_treino", "t_treino", "t_teste"]


def validacao_cruzada(fabrica, X, y, norm="zscore"):
    """fabrica() devolve um modelo novo (fit/predict). Retorna dict com
    media e desvio de cada metrica + n_params + folds_divergentes."""
    folds = criar_folds(np.arange(len(y)))
    valores = {c: [] for c in CHAVES}
    n_params, divergentes = 0, 0

    for i in range(FOLDS):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(FOLDS) if j != i])
        X_tr, X_te = normalizar(X[tr], X[te], norm)

        modelo = fabrica()
        t0 = time.perf_counter()
        modelo.fit(X_tr, y[tr])
        t_treino = time.perf_counter() - t0

        t0 = time.perf_counter()
        prev = modelo.predict(X_te)
        t_teste = time.perf_counter() - t0

        prev_tr = modelo.predict(X_tr)                  # (fora dos tempos)
        if not (np.isfinite(prev).all() and np.isfinite(prev_tr).all()):
            divergentes += 1
            for c in CHAVES[:6]:
                valores[c].append(np.nan)
        else:
            m = calcular_metricas(y[te], prev, X.shape[1])
            r2_tr = calcular_metricas(y[tr], prev_tr, X.shape[1])["R2"]
            for c in ["MSE", "RMSE", "MAE", "R2", "R2_aj"]:
                valores[c].append(m[c])
            valores["R2_treino"].append(r2_tr)
        valores["t_treino"].append(t_treino)
        valores["t_teste"].append(t_teste)
        if hasattr(modelo, "n_parametros"):
            n_params = modelo.n_parametros

    saida = {"n_params": n_params, "folds_divergentes": divergentes}
    for c in CHAVES:
        v = np.asarray(valores[c], float)
        saida[c + "_media"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        saida[c + "_desvio"] = float(np.nanstd(v)) if np.isfinite(v).any() else np.nan
    saida["gap_R2"] = saida["R2_treino_media"] - saida["R2_media"]
    return saida


def fabrica_mlp(cfg):
    return lambda: MLP(cfg["camadas"], cfg["ativacao"], cfg["lr"], cfg["epocas"],
                       semente=cfg.get("semente", SEMENTE))


def rodar(lista, nome_exp, cfg, X, y, norm="zscore", rotulo=None):
    r = validacao_cruzada(fabrica_mlp(cfg), X, y, norm)
    linha = {
        "experimento": nome_exp,
        "rotulo": rotulo or "",
        "camadas": "-".join(map(str, cfg["camadas"])),
        "n_camadas_ocultas": len(cfg["camadas"]),
        "ativacao": cfg["ativacao"],
        "lr": cfg["lr"],
        "epocas": cfg["epocas"],
        "normalizacao": norm,
        "semente": cfg.get("semente", SEMENTE),
    }
    linha.update(r)
    lista.append(linha)
    print(f"[{nome_exp}] {linha['camadas']:>12} {cfg['ativacao']:<7} lr={cfg['lr']:<7} "
          f"ep={cfg['epocas']:<4} norm={norm:<7} RMSE={r['RMSE_media']:.3f} "
          f"R2={r['R2_media']:.4f} treino={r['t_treino_media']:.2f}s"
          + (f"  DIVERGIU em {r['folds_divergentes']} folds" if r["folds_divergentes"] else ""),
          flush=True)
    return linha


# ----------------------------------------------------------------------------
# Saidas: CSV e tabelas em Markdown
# ----------------------------------------------------------------------------
def salvar_csv(linhas, nome):
    if not linhas:
        return
    campos = []
    for l in linhas:
        for c in l:
            if c not in campos:
                campos.append(c)
    with open(os.path.join(PASTA_RES, nome), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def pm(l, chave, casas=4):
    m, d = l[chave + "_media"], l[chave + "_desvio"]
    if np.isnan(m):
        return "diverge"
    return f"{m:.{casas}f} ± {d:.{casas}f}"


def tabela_md(linhas, colunas_cfg, titulo):
    cab = colunas_cfg + ["MSE", "RMSE", "MAE", "R²", "R² aj.", "R² treino",
                         "Treino (s)", "Teste (s)", "Parâmetros"]
    out = [f"### {titulo}", "", "| " + " | ".join(cab) + " |",
           "|" + "---|" * len(cab)]
    for l in linhas:
        cfg = [str(l[c]) for c in colunas_cfg]
        out.append("| " + " | ".join(cfg + [
            pm(l, "MSE", 2), pm(l, "RMSE", 3), pm(l, "MAE", 3), pm(l, "R2"),
            pm(l, "R2_aj"), (f"{l['R2_treino_media']:.4f}" if not np.isnan(l['R2_treino_media']) else "—"),
            pm(l, "t_treino", 2), pm(l, "t_teste", 4), str(l["n_params"])]) + " |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    os.makedirs(PASTA_RES, exist_ok=True)
    os.makedirs(PASTA_FIG, exist_ok=True)
    np.seterr(all="ignore")

    X, y, atributos = carregar_dados()
    print(f"Dados: {X.shape[0]} amostras, {X.shape[1]} atributos, alvo = {COLUNA_ALVO}")
    print("Atributos:", atributos, flush=True)
    md = []
    todas = []

    # ---- E1: topologias ----------------------------------------------------
    topologias = [[8], [32], [128], [16, 8], [32, 16], [64, 32],
                  [32, 32, 16], [64, 32, 16], [64, 64, 32, 16]]
    e1 = [rodar(todas, "E1", dict(BASE, camadas=t), X, y) for t in topologias]
    md.append(tabela_md(e1, ["camadas", "n_camadas_ocultas"],
                        "E1 – Topologias (ReLU, lr=0.05, 100 épocas)"))

    # E1b: a rede larga (128) diverge com lr=0.05; repete com lr menor
    e1b = [rodar(todas, "E1b", dict(BASE, camadas=[128], lr=lr), X, y) for lr in [0.01, 0.02]]
    md.append(tabela_md(e1b, ["camadas", "lr"],
                        "E1b – Rede larga [128] com taxas de aprendizado menores (ReLU, 100 épocas)"))

    # ---- E2: taxa de aprendizado ------------------------------------------
    e2 = [rodar(todas, "E2", dict(BASE, lr=lr), X, y)
          for lr in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]]
    md.append(tabela_md(e2, ["lr"], "E2 – Taxa de aprendizado ([32,16], ReLU, 100 épocas)"))

    # ---- E3: numero de epocas ---------------------------------------------
    e3 = [rodar(todas, "E3", dict(BASE, epocas=ep), X, y)
          for ep in [10, 25, 50, 100, 200, 400]]
    md.append(tabela_md(e3, ["epocas"], "E3 – Número de épocas ([32,16], ReLU, lr=0.05)"))

    # ---- E4: funcao de ativacao x taxa de aprendizado ---------------------
    e4 = [rodar(todas, "E4", dict(BASE, ativacao=a, lr=lr), X, y)
          for a in ["relu", "tanh", "sigmoid"] for lr in [0.01, 0.05, 0.1]]
    md.append(tabela_md(e4, ["ativacao", "lr"],
                        "E4 – Função de ativação × taxa de aprendizado ([32,16], 100 épocas)"))

    # ---- E5: combinacao dos melhores fatores ------------------------------
    melhor = lambda L: min([l for l in L if l["folds_divergentes"] == 0],
                          key=lambda l: l["RMSE_media"])
    b1, b3, b4 = melhor(e1), melhor(e3), melhor(e4)
    ingenua = dict(camadas=[int(c) for c in b1["camadas"].split("-")],
                   ativacao=b4["ativacao"], lr=b4["lr"], epocas=int(b3["epocas"]))
    e5 = [rodar(todas, "E5", ingenua, X, y, rotulo="melhor de cada fator (combinação direta)")]
    for t in [[16, 8], [32, 16], [64, 32, 16]]:
        for lr in [0.05, 0.1]:
            for ep in [100, 200]:
                e5.append(rodar(todas, "E5", dict(BASE, camadas=t, lr=lr, epocas=ep), X, y,
                                rotulo="grade"))
    md.append(tabela_md(e5, ["camadas", "lr", "epocas"],
                        "E5 – Combinações (ReLU): topologia × lr × épocas"))

    # selecao: melhor desempenho e melhor equilibrio
    # custo ~ parametros x epocas (proxy deterministico do numero de atualizacoes)
    candidatas = [l for l in todas if l["folds_divergentes"] == 0]
    unicas = {}
    for l in candidatas:
        chave = (l["camadas"], l["ativacao"], l["lr"], l["epocas"])
        unicas.setdefault(chave, l)
    candidatas = list(unicas.values())
    melhor_desemp = melhor(candidatas)
    limite = 1.05 * melhor_desemp["RMSE_media"]
    equilibrio = min([l for l in candidatas if l["RMSE_media"] <= limite],
                     key=lambda l: (l["n_params"] * int(l["epocas"]), l["RMSE_media"]))

    def cfg_de(l, **extra):
        d = dict(camadas=[int(c) for c in l["camadas"].split("-")],
                 ativacao=l["ativacao"], lr=l["lr"], epocas=int(l["epocas"]))
        d.update(extra)
        return d

    resumo = [melhor_desemp] + ([equilibrio] if equilibrio is not melhor_desemp else [])
    md.append("### Seleção de configurações\n\n"
              f"- **Melhor desempenho** (menor RMSE): `{melhor_desemp['camadas']}`, "
              f"{melhor_desemp['ativacao']}, lr={melhor_desemp['lr']}, "
              f"{melhor_desemp['epocas']} épocas → RMSE {melhor_desemp['RMSE_media']:.3f}, "
              f"treino {melhor_desemp['t_treino_media']:.2f}s\n"
              f"- **Melhor equilíbrio** (RMSE ≤ 1,05 × o melhor e menor custo = parâmetros × épocas): "
              f"`{equilibrio['camadas']}`, {equilibrio['ativacao']}, lr={equilibrio['lr']}, "
              f"{equilibrio['epocas']} épocas → RMSE {equilibrio['RMSE_media']:.3f}, "
              f"treino {equilibrio['t_treino_media']:.2f}s\n")

    # ---- E7: robustez a inicializacao (5 sementes) -------------------------
    def achar(camadas, ativ, lr, ep):
        for l in todas:
            if (l["camadas"], l["ativacao"], l["lr"], int(l["epocas"])) == (camadas, ativ, lr, ep) \
                    and l["experimento"] != "E7":
                return l
        return None

    alvos = [("melhor desempenho (1 semente)", melhor_desemp), ("melhor equilíbrio", equilibrio)]
    for c in [("16-8", "relu", 0.05, 200), ("32-16", "relu", 0.05, 100), ("64-32-16", "relu", 0.05, 200)]:
        l = achar(*c)
        if l is not None:
            alvos.append(("outra candidata", l))
    vistos, e7, linhas7 = set(), [], []
    for tag, l in alvos:
        chave = (l["camadas"], l["ativacao"], l["lr"], int(l["epocas"]))
        if chave in vistos:
            continue
        vistos.add(chave)
        grupo = [rodar([], "E7", cfg_de(l, semente=sem), X, y, rotulo=tag) for sem in range(5)]
        e7 += grupo
        rm = np.array([g["RMSE_media"] for g in grupo])
        r2 = np.array([g["R2_media"] for g in grupo])
        linhas7.append((np.nanmean(rm), l, tag, rm, r2))
    linhas = ["### E7 – Sensibilidade à inicialização dos pesos (5 sementes × 5 folds)", "",
              "| Configuração | RMSE por semente | RMSE médio ± desvio entre sementes | R² médio ± desvio | pior RMSE |",
              "|---|---|---|---|---|"]
    for _, l, tag, rm, r2 in sorted(linhas7, key=lambda t: t[0]):
        linhas.append(f"| {l['camadas']}, {l['ativacao']}, lr={l['lr']}, {l['epocas']} ép. ({tag}) | "
                      f"{', '.join(f'{v:.2f}' for v in rm)} | {rm.mean():.3f} ± {rm.std():.3f} | "
                      f"{r2.mean():.4f} ± {r2.std():.4f} | {rm.max():.2f} |")
    md.append("\n".join(linhas) + "\n")
    salvar_csv(e7, "mlp_sementes.csv")
    recomendada = min(linhas7, key=lambda t: t[0])[1]
    md.append(f"- **Configuração recomendada** (menor RMSE médio entre sementes): "
              f"`{recomendada['camadas']}`, {recomendada['ativacao']}, lr={recomendada['lr']}, "
              f"{recomendada['epocas']} épocas\n")

    # ---- estatisticas dos atributos (justificativa do Z-score) -------------
    mu, sd = X.mean(axis=0), X.std(axis=0)
    linhas = ["### Atributos do cpu_small (justificativa e análise do Z-score)", "",
              "| Atributo | mínimo | máximo | média | desvio | assimetria | curtose (excesso) |",
              "|---|---|---|---|---|---|---|"]
    for j, n in enumerate(atributos):
        z3 = ((X[:, j] - mu[j]) / sd[j])
        linhas.append(f"| {n} | {X[:, j].min():.4g} | {X[:, j].max():.6g} | {mu[j]:.6g} | "
                      f"{sd[j]:.6g} | {np.mean(z3 ** 3):.2f} | {np.mean(z3 ** 4) - 3:.1f} |")
    md.append("\n".join(linhas) + "\n")
    salvar_csv(todas, "mlp_todos_experimentos.csv")

    # ---- E6: impacto da normalizacao --------------------------------------
    z = []
    normas = ["zscore", "minmax", "robusta", "log_zscore", "nenhuma"]
    for ativ in ["relu", "tanh"]:
        for norm in normas:
            lrs = [0.05] if norm != "nenhuma" else [0.05, 1e-3, 1e-5, 1e-7]
            for lr in lrs:
                cfg6 = dict(BASE, camadas=[32, 16], ativacao=ativ, lr=lr)
                z.append(rodar([], "E6", cfg6, X, y, norm=norm, rotulo=f"MLP 32-16 {ativ}"))
    for nome, fab in [("KNN Euclidiana (K=5)", lambda: KNNVetorizado(K, "euclidiana")),
                      ("KNN Manhattan (K=5)", lambda: KNNVetorizado(K, "manhattan")),
                      ("Regressão linear múltipla", lambda: RegressaoLinear())]:
        for norm in normas:
            r = validacao_cruzada(fab, X, y, norm)
            r.update({"experimento": "E6", "rotulo": nome, "normalizacao": norm,
                      "camadas": "-", "ativacao": "-", "lr": "-", "epocas": "-"})
            z.append(r)
            print(f"[E6] {nome:<28} norm={norm:<10} RMSE={r['RMSE_media']:.3f} "
                  f"R2={r['R2_media']:.4f}", flush=True)
    salvar_csv(z, "zscore_impacto.csv")

    cab = ["Modelo", "Normalização", "lr", "MSE", "RMSE", "MAE", "R²", "R² aj.",
           "Treino (s)", "Teste (s)"]
    linhas = ["### E6 – Impacto da normalização (mesmos 5 folds; parâmetros estimados só no treino)",
              "", "| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for l in z:
        div = f" ⚠ divergiu em {l['folds_divergentes']}/5 folds" if l["folds_divergentes"] else ""
        linhas.append("| " + " | ".join([
            str(l["rotulo"]), l["normalizacao"], str(l["lr"]),
            pm(l, "MSE", 2), pm(l, "RMSE", 3), pm(l, "MAE", 3), pm(l, "R2"),
            pm(l, "R2_aj"), pm(l, "t_treino", 3), pm(l, "t_teste", 4)]) + div + " |")
    md.append("\n".join(linhas) + "\n")

    # ---- comparacao final entre modelos (Z-score, mesmos folds) -----------
    comp = []
    for nome, fab in [("KNN Euclidiana (K=5)", lambda: KNNVetorizado(K, "euclidiana")),
                      ("KNN Manhattan (K=5)", lambda: KNNVetorizado(K, "manhattan")),
                      ("Regressão linear múltipla", lambda: RegressaoLinear())]:
        r = validacao_cruzada(fab, X, y, "zscore")
        r.update({"rotulo": nome, "camadas": "-", "ativacao": "-", "lr": "-", "epocas": "-",
                  "normalizacao": "zscore"})
        comp.append(r)
    usados = set()
    for tag, l in [("MLP recomendada", recomendada), ("MLP melhor equilíbrio", equilibrio),
                   ("MLP melhor desempenho (semente 42)", melhor_desemp)]:
        chave = (l["camadas"], l["ativacao"], l["lr"], int(l["epocas"]))
        if chave in usados:
            continue
        usados.add(chave)
        d = dict(l)
        d["rotulo"] = f"{tag}: {l['camadas']}, {l['ativacao']}, lr={l['lr']}, {l['epocas']} ép."
        comp.append(d)
    salvar_csv(comp, "comparacao_modelos.csv")
    cab = ["Modelo", "MSE", "RMSE", "MAE", "R²", "R² aj.", "Treino (s)", "Teste (s)"]
    linhas = ["### Comparação entre modelos de regressão (5-fold, Z-score)", "",
              "| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for l in comp:
        linhas.append("| " + " | ".join([
            l["rotulo"], pm(l, "MSE", 2), pm(l, "RMSE", 3), pm(l, "MAE", 3),
            pm(l, "R2"), pm(l, "R2_aj"), pm(l, "t_treino", 3), pm(l, "t_teste", 4)]) + " |")
    md.append("\n".join(linhas) + "\n")

    with open(os.path.join(PASTA_RES, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write("# Tabelas geradas automaticamente (experimentos_mlp.py)\n\n" + "\n".join(md))

    # ---- Figuras -----------------------------------------------------------
    gerar_figuras(todas, equilibrio, melhor_desemp, X, y)
    print("\nConcluido. Veja as pastas resultados/ e figuras/.")


def gerar_figuras(todas, equilibrio, melhor_desemp, X, y):
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})
    folds = criar_folds(np.arange(len(y)))
    te = folds[0]
    tr = np.concatenate(folds[1:])
    X_tr, X_te = normalizar(X[tr], X[te], "zscore")

    def treinar(l, epocas=None):
        cfg = dict(camadas=[int(c) for c in l["camadas"].split("-")],
                   ativacao=l["ativacao"], lr=l["lr"], epocas=int(l["epocas"]))
        if epocas:
            cfg["epocas"] = epocas
        m = MLP(cfg["camadas"], cfg["ativacao"], cfg["lr"], cfg["epocas"])
        m.fit(X_tr, y[tr], X_te, y[te])
        return m

    # curvas de aprendizado: equilibrio, melhor desempenho e uma rede grande
    grande = {"camadas": "64-64-32-16", "ativacao": "relu", "lr": 0.01, "epocas": 400}
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (tit, l) in zip(eixos, [("Melhor equilíbrio", equilibrio),
                                    ("Melhor desempenho", melhor_desemp),
                                    ("Rede grande (64-64-32-16, 400 ép.)", grande)]):
        m = treinar(l)
        ax.plot(m.historico["treino"], label="treino")
        ax.plot(m.historico["validacao"], label="teste (fold 1)")
        ax.set_yscale("log")
        ax.set_title(f"{tit}\n{l['camadas']} · {l['ativacao']} · lr={l['lr']}", fontsize=9)
        ax.set_xlabel("época")
    eixos[0].set_ylabel("MSE (alvo padronizado)")
    eixos[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "curvas_aprendizado.png"))
    plt.close(fig)

    # previsto x real (melhor equilibrio)
    m = treinar(equilibrio)
    prev = m.predict(X_te)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y[te], prev, s=6, alpha=.4)
    lim = [0, 100]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("usr real")
    ax.set_ylabel("usr previsto")
    ax.set_title("Previsto × real (fold 1, melhor equilíbrio)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "previsto_vs_real.png"))
    plt.close(fig)

    # desempenho x custo computacional (todas as configuracoes)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cores = {"E1": "C0", "E1b": "C5", "E2": "C1", "E3": "C2", "E4": "C3", "E5": "C4"}
    for l in todas:
        if l["folds_divergentes"] == 0 and l["RMSE_media"] < 4.6:
            ax.scatter(l["t_treino_media"], l["RMSE_media"], c=cores[l["experimento"]],
                       s=25, alpha=.8)
    for e, c in cores.items():
        ax.scatter([], [], c=c, label=e)
    ax.set_xscale("log")
    ax.set_xlabel("tempo médio de treino por fold (s, escala log)")
    ax.set_ylabel("RMSE médio (5-fold)")
    ax.set_title("Desempenho × custo computacional (RMSE < 4,6)")
    ax.legend(title="experimento", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "desempenho_vs_custo.png"))
    plt.close(fig)

    # generalizacao: R2 treino x R2 teste nas topologias (E1)
    e1 = [l for l in todas if l["experimento"] == "E1" and l["folds_divergentes"] == 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(e1))
    ax.bar(x - .2, [l["R2_treino_media"] for l in e1], .4, label="R² treino")
    ax.bar(x + .2, [l["R2_media"] for l in e1], .4, label="R² teste")
    ax.set_xticks(x)
    ax.set_xticklabels([l["camadas"] for l in e1], rotation=30, fontsize=8)
    ax.set_ylim(.955, .985)
    ax.set_title("Generalização por topologia (E1, lr=0,05; [128] omitida: divergiu)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "generalizacao_topologias.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
