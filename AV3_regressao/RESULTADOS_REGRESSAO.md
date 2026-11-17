# AV3 – Regressão com MLP (cpu_small, alvo `usr`)

Dataset: 8192 amostras, 12 atributos, alvo `usr` (% de CPU em modo usuário). Validação cruzada de 5 folds com os **mesmos folds da AV2** (semente 42) e Z-score calculado só com o treino de cada fold. Todos os números abaixo são média ± desvio entre os 5 folds.

## 1. Como rodar

```
pip install numpy matplotlib
python experimentos_mlp.py        # ~15 min; gera resultados/ e figuras/
python extra_knn_outliers.py      # apoio à análise do Z-score
```

- `MLP.py` – MLP manual (NumPy puro): ativações ReLU/tanh/sigmoid, saída linear, perda MSE, backpropagation, gradiente descendente em mini-batches (64). Gradiente conferido numericamente (erro relativo < 1e-5).
- `experimentos_mlp.py` – todos os experimentos, tabelas (`resultados/tabelas.md`, CSVs) e figuras.
- **Tempos:** foram medidos neste ambiente (1 thread). Rode na sua máquina antes de montar os slides e use os seus tempos.

## 2. Resultados

### Comparação entre modelos (Z-score)

| Modelo | MSE | RMSE | MAE | R² | R² aj. | Treino (s) | Teste (s) |
|---|---|---|---|---|---|---|---|
| Regressão linear múltipla* | 97,08 ± 1,22 | 9,852 ± 0,062 | 6,171 ± 0,112 | 0,7131 ± 0,0064 | 0,7110 | 0,001 | 0,0001 |
| KNN Euclidiana (K=5) | 14,24 ± 4,30 | 3,735 ± 0,546 | 2,291 ± 0,097 | 0,9580 ± 0,0124 | 0,9577 | ≈0 | 0,94 |
| KNN Manhattan (K=5) | 12,89 ± 3,35 | 3,560 ± 0,459 | 2,202 ± 0,085 | 0,9619 ± 0,0099 | 0,9616 | ≈0 | 1,08 |
| **MLP 16-8, ReLU, lr 0,05, 200 ép. (recomendada)** | 9,33 ± 0,58 | 3,053 ± 0,097 | 2,187 ± 0,082 | 0,9724 ± 0,0020 | 0,9722 | 0,93 | 0,0002 |
| MLP 8, ReLU, lr 0,05, 100 ép. (mais barata) | 9,52 ± 0,70 | 3,084 ± 0,114 | 2,248 ± 0,072 | 0,9719 ± 0,0019 | 0,9717 | 0,30 | 0,0001 |

\* Referência minha (mínimos quadrados); se a AV2 de vocês já tem a regressão linear, use a de vocês. KNN = versão vetorizada, resultado idêntico ao das classes da AV2 (conferido: diferença 0,0), só que muito mais rápida.

### Experimentos de hiperparâmetros (base: 32-16, ReLU, lr 0,05, 100 ép.)

| Fator | Resultado (RMSE / R²) |
|---|---|
| **Taxa de aprendizado** | 0,0005 → 6,00 / 0,894 · 0,001 → 5,03 / 0,925 · 0,01 → 3,54 / 0,963 · 0,05 → 3,16 / 0,971 · 0,1 → 3,03 / 0,973 · 0,2 → 3,02 / 0,973 · **0,3 → diverge (RMSE 14,0 ± 8,9)** |
| **Épocas** (tempo treino) | 10 → 3,88 (0,04 s) · 50 → 3,28 (0,29 s) · 100 → 3,16 (0,63 s) · 200 → 3,10 (1,17 s) · 400 → 3,07 (2,34 s) |
| **Ativação** (lr 0,1) | ReLU 3,03 · tanh 3,04 · sigmoid 3,18 (sigmoid com lr 0,01: 4,42 — converge bem mais devagar e é ~1,5× mais lenta) |

### Topologias (ReLU, lr 0,05, 100 ép.)

| Camadas ocultas | Parâmetros | RMSE | R² | R² treino | Treino (s) | Teste (s) |
|---|---|---|---|---|---|---|
| 8 | 113 | 3,084 ± 0,114 | 0,9719 | 0,9728 | 0,30 | 0,0001 |
| 32 | 449 | 3,202 ± 0,166 | 0,9696 | 0,9715 | 0,46 | 0,0002 |
| 128 | 1793 | **diverge** (lr 0,05); com lr 0,02 → 3,379 | – | – | – | 0,0013 |
| 16-8 | 353 | 3,098 ± 0,093 | 0,9716 | 0,9745 | 0,50 | 0,0002 |
| 32-16 | 961 | 3,160 ± 0,097 | 0,9705 | 0,9754 | 0,63 | 0,0004 |
| 64-32 | 2945 | 3,039 ± 0,083 | 0,9727 | 0,9777 | 0,96 | 0,0009 |
| 32-32-16 | 2017 | 3,131 ± 0,177 | 0,9710 | 0,9762 | 0,88 | 0,0006 |
| 64-32-16 | 3457 | 3,112 ± 0,144 | 0,9714 | 0,9776 | 1,13 | 0,0010 |
| 64-64-32-16 | 7617 | 3,062 ± 0,152 | 0,9722 | 0,9811 | 1,91 | 0,0017 |

### Robustez à inicialização (5 sementes × 5 folds)

| Configuração | RMSE médio ± desvio entre sementes | Pior semente |
|---|---|---|
| 16-8, lr 0,05, 200 ép. | 3,043 ± 0,016 | 3,06 |
| 32-16, lr 0,05, 100 ép. | 3,057 ± 0,019 | 3,09 |
| 8, lr 0,05, 100 ép. | 3,129 ± 0,040 | 3,21 |
| 32-16, **lr 0,1**, 200 ép. | 3,740 ± 1,330 | **6,40** |

A configuração "campeã" da semente 42 (32-16, lr 0,1, RMSE 3,016) é frágil: em 1 de 5 sementes a rede instabilizou (RMSE 6,40). Por isso **não** a recomendo; a 16-8 com lr 0,05 é praticamente tão boa e estável.

## 3. Discussão (para os slides)

- **Desempenho:** MLP (R² ≈ 0,972) > KNN (0,958–0,962) >> regressão linear (0,713). O MLP também tem menor desvio entre folds (RMSE ± 0,10 contra ± 0,46–0,55 do KNN).
- **Hiperparâmetros:** a **taxa de aprendizado é o fator dominante** (RMSE de 6,0 a 3,0). Épocas têm retorno decrescente (de 100 para 400 épocas o RMSE cai só 3%, com custo 4×). Capacidade (topologia) importa pouco: de 113 a 7617 parâmetros o RMSE fica entre 3,04 e 3,20, diferença da ordem do desvio entre folds. Redes mais largas exigem lr menor (a de 128 neurons diverge com 0,05). ReLU e tanh empatam; sigmoid converge mais lentamente.
- **Generalização:** a lacuna R² treino − R² teste cresce com a capacidade (0,001 na rede de 8 neurônios → 0,009 na 64-64-32-16), mas o erro de teste não piora (curvas de aprendizado em `figuras/curvas_aprendizado.png`): há leve sobreajuste, sem perda de generalização. Como mais capacidade não melhora o teste, o limite parece vir dos dados (ruído do alvo) e não da rede; o alvo é bimodal: 283 amostras com `usr`=0 e o resto concentrado em valores altos (mediana 89).
- **Custo computacional:** o MLP paga no treino (0,3–1,9 s por fold) e prediz em **0,1–0,4 ms**; o KNN não treina, mas classificar o fold de teste (≈1638 amostras) leva ~1 s (**≈ 2000–10000× mais lento** que o MLP na inferência) e cresce com o tamanho da base. Regressão linear é instantânea, mas erra muito.
- **Conclusão:** melhor equilíbrio = **MLP 16-8, ReLU, lr 0,05, 200 épocas** (R² 0,972, 353 parâmetros, ~0,9 s de treino, estável entre sementes). Se o custo de treino for prioridade, a **rede com 8 neurônios** custa ⅓ do tempo (0,30 s) e perde só ~1% de RMSE.

## 4. Seção 3 do PDF – Impacto do Z-score (regressores da AV2)

No código da AV2, o Z-score foi usado nos regressores (KNN) e calculado só com o treino de cada fold (sem vazamento). Os classificadores (Naive Bayes) não usaram normalização, então a análise vale para os regressores e para o MLP.

| Normalização | KNN Eucl. (RMSE) | KNN Manh. (RMSE) | Regr. linear (RMSE) | MLP ReLU (RMSE) | MLP tanh (RMSE) |
|---|---|---|---|---|---|
| Nenhuma | 6,696 | 6,459 | 9,852 | **diverge** (todos os folds, lr 0,05 a 1e-7) | 7,278 (R² 0,843) |
| Z-score | 3,735 ± 0,546 | 3,560 ± 0,459 | 9,852 | 3,160 | 3,176 |
| Min-Max | 3,169 ± 0,194 | 3,050 ± 0,163 | 9,852 | 3,142 | 3,216 |
| Robusta (mediana/IQR) | 3,432 | 3,322 | 9,852 | diverge em 2/5 folds | 3,066 |
| log(1+x) + Z-score | 3,194 | 3,151 | **4,179 (R² 0,948)** | 3,072 | 3,033 |

- **Justificativa:** os atributos estão em escalas muito diferentes (`rchar` até 2,5·10⁶; `fork` até 20). KNN depende de distância e o MLP de gradiente, então ambos são sensíveis à escala.
- **Impacto:** sem normalização o erro do KNN quase dobra (RMSE 6,7 vs 3,7), o MLP ReLU diverge em qualquer lr testado e o tanh perde 0,13 de R². Na regressão linear (solução fechada) o Z-score não muda nada (resultado idêntico), só reescala os coeficientes.
- **Limite do Z-score aqui:** vários atributos têm caudas muito pesadas (curtose: `lread` 314, `swrite` 234, `runqsz` 174; ver `resultados/tabelas.md`). O Z-score não limita os outliers, e o KNN com Z-score tem RMSE pior e bem mais instável que com Min-Max (± 0,55 vs ± 0,19): o MAE é parecido (2,29 vs 2,17), mas há **9 previsões com erro > 30 pontos contra 1** (`extra_knn_outliers.py`).
- **Conclusão crítica:** o Z-score foi adequado e necessário (evita divergência e reduz o erro em ~45% no KNN), mas não foi a melhor escolha. **log(1+x) + Z-score** foi a melhor ou ficou perto da melhor opção em todos os modelos e fez a regressão linear passar de R² 0,71 para 0,95 (a relação com os atributos é não linear e muito assimétrica). Cuidado ao afirmar: para o MLP a diferença para o Z-score simples é pequena (~0,09 de RMSE, na ordem de 1 desvio entre folds).

## 5. Perguntas prováveis na apresentação

1. *Por que padronizar o alvo no treino do MLP?* Mantém os gradientes em escala moderada; a previsão é devolvida à escala original antes de calcular as métricas.
2. *Por que ReLU/tanh/sigmoid se comportam diferente?* Sigmoid tem derivada máx. 0,25 e satura, então aprende mais devagar (precisa de lr maior); ReLU não satura no lado positivo; tanh é centrada em zero.
3. *Por que a rede de 128 neurônios divergiu?* Explicação provável (não testada isoladamente): com mais pesos somando na saída, o gradiente e o passo efetivo crescem; com lr 0,02 convergiu (RMSE 3,38).
4. *Por que R² ajustado ≈ R²?* n ≈ 1638 por fold de teste e só p = 12 atributos: a penalização (n−1)/(n−p−1) é mínima.
5. *Por que o MLP é bem mais rápido que o KNN no teste?* O KNN calcula distância para todas as ≈6550 amostras de treino a cada consulta; o MLP faz poucas multiplicações de matrizes.

## 6. Limitações

- Uma divisão em folds (semente 42); a variação por semente de inicialização foi medida só nas configurações da tabela de robustez.
- Mini-batch fixo em 64 e gradiente descendente simples (sem momentum/Adam); batch e otimizador não foram variados.
- Escolha de hiperparâmetros feita com os mesmos 5 folds usados para reportar (sem conjunto de validação separado), então os números são levemente otimistas.
