# Algoritmo Genético para seleção de características (Random Forest)

Código do artigo **"Algoritmos Genéticos aplicados à seleção de características em Machine Learning"**
(X EDEPA, UNIUBE). Um Algoritmo Genético (AG) escolhe subconjuntos de atributos do conjunto
*Breast Cancer Wisconsin (Diagnostic)* (569 instâncias, 30 atributos), avaliados com Random Forest.

## O que o código faz

- **AG (Wrapper):** cromossomo binário de 30 genes; aptidão `f(x) = α·Acc(x) + (1 − α)·(1 − |S_x|/N)`,
  com `Acc` em [0, 1] (validação cruzada estratificada interna, k = 5) e `α = 0,9`.
- **Operadores:** torneio (tamanho 3), cruzamento de ponto único (80%), mutação bit-flip (2% por gene),
  elitismo de 1 indivíduo, parada por número fixo de gerações.
- **Decisões de implementação:** população inicial uniforme; cromossomo vazio é reparado (liga um gene
  sorteado); cache de aptidões; empate resolvido a favor do subconjunto com menos atributos.
- **Avaliação:** validação cruzada **aninhada** e estratificada (5 dobras externas). O AG roda só no
  treinamento de cada dobra externa; a dobra externa só mede o desempenho.
- **Comparadores, nas mesmas partições:** modelo completo (C1), AG (C2), Filter `SelectKBest`/ANOVA F com
  o mesmo número de atributos do AG (C3) e busca aleatória com o mesmo número de avaliações únicas (C4).
- **Saídas:** métricas por dobra, tempos (busca, treino, inferência), atributos selecionados e
  frequência de seleção, ambiente e argumentos usados. A classe positiva é *Maligno*.

## Como executar

```bash
pip install -r requirements.txt

# Teste rápido (alguns segundos)
python ga_selecao_caracteristicas.py --pop 8 --gens 3 --arvores 20 --sem-aleatoria --saida teste

# Execução relatada no artigo (orçamento reduzido, ~15 min com 2 núcleos)
python ga_selecao_caracteristicas.py --pop 24 --gens 15 --n-jobs 2 --seeds 0 --saida resultados

# Configuração original do projeto (população 50, 100 gerações), várias sementes: muito mais lenta
python ga_selecao_caracteristicas.py --seeds 0 1 2 3 4
```

O dataset vem do scikit-learn (não precisa de internet). `python ga_selecao_caracteristicas.py --help`
lista todos os parâmetros.

## Resultados da execução relatada no artigo

População 24, 15 gerações, semente 0, 5 dobras externas (média ± desvio-padrão entre dobras):

| Configuração | Atributos | Acurácia (%) | F1 (%) |
|---|---|---|---|
| C1 completo | 30 | 96,49 ± 1,64 | 95,23 ± 2,18 |
| C2 AG | 5,4 | 96,66 ± 1,90 | 95,41 ± 2,60 |
| C3 Filter (mesmo k) | 5,4 | 94,21 ± 2,81 | 92,09 ± 3,89 |
| C4 busca aleatória | 5,0 | 94,90 ± 2,09 | 93,00 ± 2,94 |

Arquivos completos em [`resultados/execucao_reduzida/`](resultados/execucao_reduzida/)
(`resumo.txt`, `por_dobra.csv`, `resultados.json`).

## Limitações

- O orçamento do AG foi **reduzido** (população 24 e 15 gerações, em vez de 50 e 100), por limite de tempo.
- Uma única semente e 5 dobras: as diferenças de décimos de ponto percentual estão dentro da variação entre dobras.
- A seleção foi instável entre as dobras (só `worst texture` apareceu nas 5).
- Um único conjunto de dados, de dimensão moderada; os resultados não permitem inferir validade clínica.

## Ambiente de teste

Python 3.13, numpy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0; 2 núcleos de CPU.
