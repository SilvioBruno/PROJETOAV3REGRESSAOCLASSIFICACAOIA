
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experimentos import (
    criar_folds,
    avaliar_modelo,
    gerar_experimentos
)

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / "dataset_player-position.csv"
PASTA_RESULTADOS = PASTA / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

NOMES_CLASSES = [
    "Defensor",
    "Atacante",
    "Goleiro",
    "Meio-campista"
]


def carregar_dataset():
    with open(ARQUIVO, "r", encoding="utf-8-sig") as arquivo:
        primeira_linha = arquivo.readline()

    delimitador = ";" if primeira_linha.count(";") > primeira_linha.count(",") else ","

    dados = np.genfromtxt(
        ARQUIVO,
        delimiter=delimitador,
        names=True,
        dtype=None,
        encoding="utf-8-sig"
    )

    dados = np.atleast_1d(dados)
    nomes = dados.dtype.names

    coluna_classe = next(
        (nome for nome in nomes if nome.lower().strip() == "class"),
        None
    )

    if coluna_classe is None:
        raise ValueError("Não foi encontrada a coluna 'class'.")

    y = np.asarray(dados[coluna_classe], dtype=int)
    colunas = [nome for nome in nomes if nome != coluna_classe]

    atributos = []

    for coluna in colunas:
        try:
            atributos.append(np.asarray(dados[coluna], dtype=float))
        except (ValueError, TypeError):
            raise ValueError(
                f"A coluna '{coluna}' não é numérica. "
                "É necessário codificar os atributos categóricos."
            )

    X = np.column_stack(atributos)

    if not np.array_equal(np.unique(y), np.array([0, 1, 2, 3])):
        raise ValueError(
            f"Esperadas classes 0, 1, 2 e 3. "
            f"Encontradas: {np.unique(y).tolist()}"
        )

    return X, y


def salvar_tabela(resultados):
    campos = [
        "modelo",
        "configuracao",
        "acuracia_media",
        "acuracia_desvio",
        "precisao_media",
        "precisao_desvio",
        "recall_media",
        "recall_desvio",
        "especificidade_media",
        "especificidade_desvio",
        "f1_media",
        "f1_desvio",
        "tempo_treino_media",
        "tempo_treino_desvio",
        "tempo_teste_media",
        "tempo_teste_desvio"
    ]

    with open(
        PASTA_RESULTADOS / "comparacao_classificadores.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()

        for resultado in resultados:
            escritor.writerow({
                campo: resultado[campo]
                for campo in campos
            })


def salvar_matriz(matriz, nome_arquivo, titulo):
    fig, ax = plt.subplots(figsize=(8, 6))
    imagem = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagem, ax=ax)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(NOMES_CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(NOMES_CLASSES)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)

    limite = matriz.max() / 2 if matriz.max() > 0 else 0

    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                color="white" if matriz[i, j] > limite else "black"
            )

    fig.tight_layout()
    fig.savefig(
        PASTA_RESULTADOS / nome_arquivo,
        dpi=300
    )
    plt.close(fig)


def executar():
    X, y = carregar_dataset()
    folds = criar_folds(y)
    experimentos = gerar_experimentos()
    resultados = []

    print("DATASET CARREGADO")
    print(f"Amostras: {len(X)}")
    print(f"Atributos: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print("Validação cruzada: 5 folds\n")

    for i, (nome, configuracao, criar_modelo) in enumerate(experimentos, 1):
        print(
            f"Executando {i}/{len(experimentos)}: "
            f"{nome} | {configuracao}"
        )

        resultado = avaliar_modelo(
            nome,
            configuracao,
            criar_modelo,
            X,
            y,
            folds
        )

        resultados.append(resultado)

    resultados.sort(
        key=lambda item: item["f1_media"],
        reverse=True
    )

    salvar_tabela(resultados)

    melhor_geral = resultados[0]

    perceptrons = [
        item for item in resultados
        if item["modelo"] == "Perceptron"
    ]

    melhor_perceptron = max(
        perceptrons,
        key=lambda item: item["f1_media"]
    )

    salvar_matriz(
        melhor_geral["matriz"],
        "matriz_confusao_melhor_modelo.png",
        f"Melhor modelo: {melhor_geral['modelo']}"
    )

    salvar_matriz(
        melhor_perceptron["matriz"],
        "matriz_confusao_perceptron.png",
        "Melhor configuração do Perceptron"
    )

    print("\nRANKING DOS CLASSIFICADORES")
    print("-" * 90)

    for posicao, item in enumerate(resultados, 1):
        print(
            f"{posicao}. {item['modelo']} | {item['configuracao']} | "
            f"Acurácia: {item['acuracia_media']:.4f} ± "
            f"{item['acuracia_desvio']:.4f} | "
            f"F1: {item['f1_media']:.4f} ± "
            f"{item['f1_desvio']:.4f} | "
            f"Treino: {item['tempo_treino_media']:.4f}s | "
            f"Teste: {item['tempo_teste_media']:.4f}s"
        )

    print("\nMELHOR MODELO GERAL")
    print(melhor_geral["modelo"])
    print(melhor_geral["configuracao"])
    print(f"F1 médio: {melhor_geral['f1_media']:.4f}")
    print(f"Desvio padrão: {melhor_geral['f1_desvio']:.4f}")

    print("\nMELHOR PERCEPTRON")
    print(melhor_perceptron["configuracao"])
    print(f"F1 médio: {melhor_perceptron['f1_media']:.4f}")
    print(f"Desvio padrão: {melhor_perceptron['f1_desvio']:.4f}")

    print("\nArquivos salvos em:", PASTA_RESULTADOS)


if __name__ == "__main__":
    executar()
