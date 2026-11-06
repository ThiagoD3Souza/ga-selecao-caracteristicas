#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seleção de características com Algoritmo Genético (AG) + Random Forest.

Conjunto de dados: Breast Cancer Wisconsin (Diagnostic), 569 instâncias, 30 atributos
(carregado do scikit-learn; não precisa de internet).

Protocolo
---------
* Avaliação EXTERNA: validação cruzada estratificada (k_externo = 5). Em cada dobra externa
  o AG roda APENAS no treinamento externo; a dobra externa só serve para medir o desempenho.
* Aptidão (fitness), dentro do AG:  f(x) = alfa * Acc(x) + (1 - alfa) * (1 - |S_x| / N),
  com Acc em [0, 1] = acurácia média de uma validação cruzada estratificada interna (k = 5)
  de um Random Forest (100 árvores) treinado só com os atributos de x.
* Configurações comparadas nas MESMAS partições externas:
    C1 completo   : Random Forest com os 30 atributos
    C2 AG         : atributos escolhidos pelo AG
    C3 Filter     : SelectKBest (ANOVA F) com k = nº de atributos escolhido pelo AG na dobra
    C4 aleatória  : busca aleatória com o MESMO número de avaliações (únicas) do AG
* Modo "ingênuo": AG sobre todos os dados e relato da acurácia de validação cruzada usada na
  própria busca. Serve para mostrar o viés otimista de avaliar nas dobras usadas para escolher.

Classe positiva: Maligno (1). No scikit-learn, target 0 = maligno, 1 = benigno; aqui recodificamos.

Hiperparâmetros padrão = os do artigo (população 50, 100 gerações, torneio 3, cruzamento de ponto
único 80%, mutação bit-flip 2% por gene, alfa 0,9). Decisões não especificadas no artigo e adotadas
aqui: inicialização uniforme (cada gene 0/1 com prob. 0,5), elitismo de 1 indivíduo, parada por
número fixo de gerações, reparo de cromossomo vazio (liga um gene sorteado), empate resolvido a
favor do subconjunto com menos atributos, cache de aptidões (avaliação determinística por semente).

Uso rápido (teste):   python ga_selecao_caracteristicas.py --pop 8 --gens 3 --sem-aleatoria
Uso completo (artigo): python ga_selecao_caracteristicas.py --seeds 0 1 2 3 4
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# --------------------------------------------------------------------------- dados
def carregar_dados():
    d = load_breast_cancer()
    X = d.data.astype(float)
    y = (d.target == 0).astype(int)  # 1 = Maligno (positiva), 0 = Benigno
    return X, y, list(d.feature_names)


# --------------------------------------------------------------------------- aptidão
def avaliar_mascara(X, y, mascara, alfa, n_arvores, k_interno, semente):
    """Retorna (aptidão, acurácia interna) de um subconjunto de atributos."""
    idx = np.flatnonzero(mascara)
    cv = StratifiedKFold(k_interno, shuffle=True, random_state=semente)
    rf = RandomForestClassifier(n_estimators=n_arvores, random_state=semente, n_jobs=1)
    acc = float(cross_val_score(rf, X[:, idx], y, cv=cv, scoring="accuracy").mean())
    n = X.shape[1]
    apt = alfa * acc + (1.0 - alfa) * (1.0 - len(idx) / n)
    return apt, acc


def reparar(m, rng):
    """Cromossomo vazio é inadmissível: liga um gene sorteado."""
    if m.sum() == 0:
        m[rng.integers(len(m))] = 1
    return m


# --------------------------------------------------------------------------- AG
class AvaliadorComCache:
    """Avalia indivíduos (em paralelo) guardando as aptidões já calculadas."""

    def __init__(self, X, y, args, semente, paralelo):
        self.X, self.y, self.args, self.semente, self.par = X, y, args, semente, paralelo
        self.cache = {}
        self.pedidas = 0

    def __call__(self, pop):
        self.pedidas += len(pop)
        novos = {}
        for m in pop:
            k = m.tobytes()
            if k not in self.cache and k not in novos:
                novos[k] = m
        if novos:
            a = self.args
            res = self.par(
                delayed(avaliar_mascara)(self.X, self.y, m, a.alfa, a.arvores, a.k_interno, self.semente)
                for m in novos.values()
            )
            for k, r in zip(novos.keys(), res):
                self.cache[k] = r
        fit = np.array([self.cache[m.tobytes()][0] for m in pop])
        acc = np.array([self.cache[m.tobytes()][1] for m in pop])
        return fit, acc

    @property
    def unicas(self):
        return len(self.cache)


def ordenar(fit, nfeat):
    """Índices do melhor para o pior: maior aptidão; empate -> menos atributos."""
    return np.lexsort((nfeat, -fit))


def torneio(pop, fit, nfeat, rng, t):
    cand = rng.integers(len(pop), size=t)
    melhor = cand[ordenar(fit[cand], nfeat[cand])[0]]
    return pop[melhor]


def executar_ag(X, y, args, semente, par):
    rng = np.random.default_rng(semente)
    n = X.shape[1]
    aval = AvaliadorComCache(X, y, args, semente, par)

    pop = np.array([reparar(rng.integers(0, 2, size=n).astype(np.uint8), rng) for _ in range(args.pop)])
    fit, acc = aval(pop)
    historico = []
    for g in range(args.gens + 1):
        nfeat = pop.sum(axis=1)
        ordem = ordenar(fit, nfeat)
        historico.append({"geracao": g, "melhor_aptidao": float(fit[ordem[0]]),
                          "aptidao_media": float(fit.mean()), "atributos_do_melhor": int(nfeat[ordem[0]])})
        if g == args.gens:
            break
        elite = pop[ordem[: args.elite]].copy()
        filhos = []
        while len(filhos) < args.pop - args.elite:
            p1 = torneio(pop, fit, nfeat, rng, args.torneio)
            p2 = torneio(pop, fit, nfeat, rng, args.torneio)
            if rng.random() < args.pc:
                ponto = int(rng.integers(1, n))
                c1 = np.concatenate([p1[:ponto], p2[ponto:]])
                c2 = np.concatenate([p2[:ponto], p1[ponto:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                c ^= (rng.random(n) < args.pm).astype(np.uint8)  # mutação bit-flip por gene
                filhos.append(reparar(c, rng))
        pop = np.vstack([elite] + filhos[: args.pop - args.elite])
        fit, acc = aval(pop)

    # melhor de todas as gerações (o elitismo garante que está na última, mas não dependemos disso)
    chaves = list(aval.cache.keys())
    todos = np.array([np.frombuffer(k, dtype=np.uint8) for k in chaves])
    f_all = np.array([aval.cache[k][0] for k in chaves])
    a_all = np.array([aval.cache[k][1] for k in chaves])
    i = ordenar(f_all, todos.sum(axis=1))[0]
    return {"mascara": todos[i].copy(), "aptidao": float(f_all[i]), "acc_interna": float(a_all[i]),
            "avaliacoes_pedidas": aval.pedidas, "avaliacoes_unicas": aval.unicas, "historico": historico}


def busca_aleatoria(X, y, args, semente, par, orcamento):
    """Amostra `orcamento` subconjuntos distintos (tamanho uniforme em 1..N) e escolhe o de maior aptidão."""
    rng = np.random.default_rng(semente + 10_000)
    n = X.shape[1]
    vistos, pop = set(), []
    while len(pop) < orcamento:
        k = int(rng.integers(1, n + 1))
        m = np.zeros(n, dtype=np.uint8)
        m[rng.choice(n, size=k, replace=False)] = 1
        if m.tobytes() not in vistos:
            vistos.add(m.tobytes())
            pop.append(m)
    pop = np.array(pop)
    aval = AvaliadorComCache(X, y, args, semente, par)
    fit, _ = aval(pop)
    i = ordenar(fit, pop.sum(axis=1))[0]
    return {"mascara": pop[i].copy(), "aptidao": float(fit[i]), "avaliacoes_unicas": aval.unicas}


# --------------------------------------------------------------------------- avaliação final
def metricas(yv, pred):
    return {"acuracia": accuracy_score(yv, pred), "precisao": precision_score(yv, pred, zero_division=0),
            "sensibilidade": recall_score(yv, pred, zero_division=0), "f1": f1_score(yv, pred, zero_division=0)}


def treinar_e_medir(Xtr, ytr, Xte, yte, idx, args, semente):
    rf = RandomForestClassifier(n_estimators=args.arvores, random_state=semente, n_jobs=1)
    t0 = time.perf_counter()
    rf.fit(Xtr[:, idx], ytr)
    t_treino = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = rf.predict(Xte[:, idx])
    t_inf = time.perf_counter() - t0
    m = metricas(yte, pred)
    m.update({"n_atributos": int(len(idx)), "t_treino_s": t_treino, "t_inferencia_s": t_inf})
    return m


# --------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--arvores", type=int, default=100)
    ap.add_argument("--k-interno", dest="k_interno", type=int, default=5)
    ap.add_argument("--k-externo", dest="k_externo", type=int, default=5)
    ap.add_argument("--alfa", type=float, default=0.9)
    ap.add_argument("--torneio", type=int, default=3)
    ap.add_argument("--pc", type=float, default=0.8)
    ap.add_argument("--pm", type=float, default=0.02)
    ap.add_argument("--elite", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=-1)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--sem-aleatoria", action="store_true", help="pula a busca aleatória (C4)")
    ap.add_argument("--sem-ingenuo", action="store_true", help="pula o modo ingênuo")
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    X, y, nomes = carregar_dados()
    print(f"Dados: {X.shape[0]} instâncias, {X.shape[1]} atributos, malignos={int(y.sum())}, benignos={int((1-y).sum())}", flush=True)

    linhas, selecoes, ingenuos, ag_hist = [], [], [], []
    t_global = time.perf_counter()
    with Parallel(n_jobs=args.n_jobs) as par:
        for semente in args.seeds:
            kf = StratifiedKFold(args.k_externo, shuffle=True, random_state=semente)
            for d, (itr, ite) in enumerate(kf.split(X, y), start=1):
                Xtr, ytr, Xte, yte = X[itr], y[itr], X[ite], y[ite]
                # C1: modelo completo
                todos = np.arange(X.shape[1])
                r = treinar_e_medir(Xtr, ytr, Xte, yte, todos, args, semente)
                r.update({"semente": semente, "dobra": d, "config": "C1_completo", "t_busca_s": 0.0, "avaliacoes": 0})
                linhas.append(r)
                # C2: AG
                t0 = time.perf_counter()
                ag = executar_ag(Xtr, ytr, args, semente + d, par)
                t_busca = time.perf_counter() - t0
                idx = np.flatnonzero(ag["mascara"])
                r = treinar_e_medir(Xtr, ytr, Xte, yte, idx, args, semente)
                r.update({"semente": semente, "dobra": d, "config": "C2_AG", "t_busca_s": t_busca,
                          "avaliacoes": ag["avaliacoes_unicas"]})
                linhas.append(r)
                selecoes.append({"semente": semente, "dobra": d, "atributos": [nomes[i] for i in idx],
                                 "aptidao_interna": ag["aptidao"], "acc_interna": ag["acc_interna"]})
                ag_hist.append({"semente": semente, "dobra": d, "historico": ag["historico"],
                                "avaliacoes_pedidas": ag["avaliacoes_pedidas"], "avaliacoes_unicas": ag["avaliacoes_unicas"]})
                # C3: Filter com o mesmo número de atributos do AG
                t0 = time.perf_counter()
                sel = SelectKBest(f_classif, k=len(idx)).fit(Xtr, ytr)
                idx_f = np.flatnonzero(sel.get_support())
                t_f = time.perf_counter() - t0
                r = treinar_e_medir(Xtr, ytr, Xte, yte, idx_f, args, semente)
                r.update({"semente": semente, "dobra": d, "config": "C3_Filter", "t_busca_s": t_f, "avaliacoes": 0})
                linhas.append(r)
                # C4: busca aleatória com o mesmo orçamento de avaliações únicas
                if not args.sem_aleatoria:
                    t0 = time.perf_counter()
                    ra = busca_aleatoria(Xtr, ytr, args, semente + d, par, ag["avaliacoes_unicas"])
                    t_r = time.perf_counter() - t0
                    idx_r = np.flatnonzero(ra["mascara"])
                    r = treinar_e_medir(Xtr, ytr, Xte, yte, idx_r, args, semente)
                    r.update({"semente": semente, "dobra": d, "config": "C4_aleatoria", "t_busca_s": t_r,
                              "avaliacoes": ra["avaliacoes_unicas"]})
                    linhas.append(r)
                print(f"[semente {semente} dobra {d}] AG: {len(idx)} atributos, busca {t_busca:.0f}s, "
                      f"{ag['avaliacoes_unicas']} avaliações únicas ({ag['avaliacoes_pedidas']} pedidas) | "
                      + " | ".join(f"{l['config']}: acc={l['acuracia']:.3f}" for l in linhas[-(4 if not args.sem_aleatoria else 3):]),
                      flush=True)
            # modo ingênuo
            if not args.sem_ingenuo:
                t0 = time.perf_counter()
                ag = executar_ag(X, y, args, semente + 99, par)
                ingenuos.append({"semente": semente, "n_atributos": int(ag["mascara"].sum()),
                                 "acc_cv_da_busca": ag["acc_interna"], "tempo_s": time.perf_counter() - t0})
                print(f"[semente {semente}] modo ingênuo: {int(ag['mascara'].sum())} atributos, "
                      f"acurácia de CV usada na busca = {ag['acc_interna']:.4f}", flush=True)

    # ------------------------------------------------------------------ saídas
    campos = ["semente", "dobra", "config", "n_atributos", "acuracia", "precisao", "sensibilidade", "f1",
              "t_busca_s", "t_treino_s", "t_inferencia_s", "avaliacoes"]
    with open(os.path.join(args.saida, "por_dobra.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for l in linhas:
            w.writerow({k: (round(l[k], 6) if isinstance(l[k], float) else l[k]) for k in campos})

    configs = sorted({l["config"] for l in linhas})
    resumo = {}
    for c in configs:
        L = [l for l in linhas if l["config"] == c]
        resumo[c] = {k: {"media": float(np.mean([l[k] for l in L])), "dp": float(np.std([l[k] for l in L], ddof=1)) if len(L) > 1 else 0.0}
                     for k in ["n_atributos", "acuracia", "precisao", "sensibilidade", "f1", "t_busca_s", "t_treino_s", "t_inferencia_s", "avaliacoes"]}
        resumo[c]["n_dobras"] = len(L)

    freq = {}
    for s in selecoes:
        for a in s["atributos"]:
            freq[a] = freq.get(a, 0) + 1
    freq = dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))

    # diferença pareada AG - completo (mesmas dobras)
    def serie(c, m):
        return np.array([l[m] for l in linhas if l["config"] == c])
    dif = serie("C2_AG", "acuracia") - serie("C1_completo", "acuracia")

    ambiente = {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "numpy": np.__version__,
                "plataforma": platform.platform(), "cpus": os.cpu_count(), "argumentos": vars(args),
                "tempo_total_s": time.perf_counter() - t_global}
    with open(os.path.join(args.saida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "selecao_por_dobra": selecoes, "frequencia_atributos": freq,
                   "ingenuo": ingenuos, "diferenca_acuracia_AG_menos_completo": dif.tolist(),
                   "historico_ag": ag_hist, "ambiente": ambiente}, f, ensure_ascii=False, indent=2)

    # relatório em texto
    nome = {"C1_completo": "C1 completo (30)", "C2_AG": "C2 AG", "C3_Filter": "C3 Filter (mesmo k)", "C4_aleatoria": "C4 aleatória"}
    out = []
    out.append("RESUMO (média ± desvio-padrão entre as dobras externas; classe positiva = Maligno)\n")
    out.append(f"{'Configuração':22s} {'Atrib.':>8s} {'Acurácia %':>14s} {'Precisão %':>14s} {'Sensib. %':>14s} {'F1 %':>14s}")
    for c in configs:
        r = resumo[c]
        f = lambda k, p=100: f"{r[k]['media']*p:6.2f}±{r[k]['dp']*p:4.2f}"
        out.append(f"{nome[c]:22s} {r['n_atributos']['media']:8.1f} {f('acuracia'):>14s} {f('precisao'):>14s} {f('sensibilidade'):>14s} {f('f1'):>14s}")
    out.append("\nTEMPOS (média por dobra, segundos)")
    out.append(f"{'Configuração':22s} {'Busca':>10s} {'Treino final':>14s} {'Inferência':>12s} {'Avaliações':>11s}")
    for c in configs:
        r = resumo[c]
        out.append(f"{nome[c]:22s} {r['t_busca_s']['media']:10.2f} {r['t_treino_s']['media']:14.4f} {r['t_inferencia_s']['media']:12.4f} {r['avaliacoes']['media']:11.0f}")
    out.append(f"\nDiferença de acurácia AG - completo por dobra (p.p.): {np.round(dif*100, 2).tolist()}  média = {dif.mean()*100:.2f}")
    out.append(f"Dobras em que o AG foi >= completo: {int((dif >= 0).sum())} de {len(dif)}")
    out.append("\nFREQUÊNCIA DE SELEÇÃO DOS ATRIBUTOS PELO AG (nº de dobras, de " + str(len(selecoes)) + ")")
    for a, n in freq.items():
        out.append(f"  {n}  {a}")
    if ingenuos:
        out.append("\nMODO INGÊNUO (AG sobre todos os dados; acurácia de CV usada na própria busca)")
        for g in ingenuos:
            out.append(f"  semente {g['semente']}: {g['n_atributos']} atributos, acurácia de CV da busca = {g['acc_cv_da_busca']*100:.2f}%")
    out.append(f"\nTempo total: {ambiente['tempo_total_s']:.0f} s")
    texto = "\n".join(out)
    with open(os.path.join(args.saida, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\n" + texto, flush=True)


if __name__ == "__main__":
    main()
