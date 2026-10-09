"""Métricas padronizadas do trabalho.

É exatamente este código que o professor roda na correção (via
``scripts/avaliar.py``). Três níveis de avaliação:

1. **Escopo** - o 1º classificador sozinho, em todas as mensagens do teste
   (dominio × chitchat × fora_dominio).
2. **Tema** - o 2º classificador sozinho, apenas nas mensagens que *de fato* são do
   domínio (como se o 1º estágio fosse perfeito).
3. **Ponta a ponta** - a mensagem atravessa o grafo inteiro e comparamos a ``rota``
   final com o rótulo verdadeiro. Erros dos dois estágios se acumulam aqui; é a
   métrica que mais se parece com a experiência do usuário.

Duas taxas de erro merecem destaque num chatbot:

* **falso aceite**  - mensagem chitchat/fora do domínio que o bot respondeu como se
  fosse uma intenção do domínio (o bot "inventa" uma resposta);
* **falsa recusa**  - mensagem do domínio que o bot recusou ou tratou como conversa
  social (o usuário fica sem resposta que o bot tinha).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import Config
from .dados import rotulo_escopo
from .rotulos import CHITCHAT, CLASSES_ESCOPO, ESCLARECER, FORA_DOMINIO


def _metricas(y_true, y_pred, rotulos) -> dict:
    # O F1 macro considera só as classes presentes no conjunto avaliado: um teste
    # sem nenhuma mensagem de chitchat, por exemplo, não deve ganhar um "F1 = 0"
    # artificial para essa classe. Previsões erradas *para* essas classes continuam
    # contando como erro (derrubam a revocação da classe verdadeira).
    presentes = [r for r in rotulos if r in set(y_true)]
    return {
        "n": int(len(y_true)),
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=presentes, average="macro", zero_division=0)),
        "por_classe": classification_report(
            y_true, y_pred, labels=rotulos, output_dict=True, zero_division=0
        ),
        "rotulos": list(rotulos),
        "matriz_confusao": confusion_matrix(y_true, y_pred, labels=rotulos).tolist(),
    }


def avaliar_escopo(clf_escopo, textos, intencoes, config: Config) -> dict:
    y_true = rotulo_escopo(intencoes, config)
    y_pred = [str(r) for r in clf_escopo.predict(list(textos))]
    return _metricas(y_true, y_pred, CLASSES_ESCOPO)


def avaliar_tema(clf_tema, textos, intencoes, config: Config) -> dict:
    dominio = set(config.nomes_intencoes)
    pares = [(t, r) for t, r in zip(textos, intencoes) if r in dominio]
    if not pares:
        return {"n": 0}
    t, y_true = zip(*pares)
    y_pred = [str(r) for r in clf_tema.predict(list(t))]
    return _metricas(list(y_true), y_pred, config.nomes_intencoes)


def avaliar_ponta_a_ponta(grafo, textos, intencoes, config: Config) -> tuple[dict, pd.DataFrame]:
    """Passa cada mensagem pelo grafo. Devolve as métricas e a tabela de previsões."""
    linhas, tempos = [], []
    for texto, real in zip(textos, intencoes):
        inicio = time.perf_counter()
        estado = grafo.invoke({"mensagem": texto})
        tempos.append(time.perf_counter() - inicio)
        linhas.append({
            "texto": texto,
            "real": real,
            "rota": estado.get("rota", "<sem rota>"),
            "escopo": estado.get("escopo"),
            "confianca_escopo": estado.get("confianca_escopo"),
            "intencao": estado.get("intencao"),
            "confianca_intencao": estado.get("confianca_intencao"),
            "resposta": estado.get("resposta", ""),
        })
    tab = pd.DataFrame(linhas)
    tab["acertou"] = tab["real"] == tab["rota"]

    rotulos = config.nomes_intencoes + [CHITCHAT, FORA_DOMINIO]
    met = _metricas(tab["real"], tab["rota"], rotulos)
    # a matriz de confusão ponta a ponta ganha uma coluna extra: "esclarecer"
    met["rotulos_previstos"] = rotulos + [ESCLARECER]
    met["matriz_confusao"] = pd.crosstab(
        pd.Categorical(tab["real"], categories=rotulos),
        pd.Categorical(tab["rota"], categories=rotulos + [ESCLARECER]),
        dropna=False,
    ).values.tolist()

    dominio = tab["real"].isin(config.nomes_intencoes)
    rota_dominio = tab["rota"].isin(config.nomes_intencoes)
    met["taxa_falso_aceite"] = float(rota_dominio[~dominio].mean()) if (~dominio).any() else None
    met["taxa_falsa_recusa"] = (
        float(tab.loc[dominio, "rota"].isin([CHITCHAT, FORA_DOMINIO]).mean()) if dominio.any() else None
    )
    met["taxa_esclarecimento"] = float((tab["rota"] == ESCLARECER).mean())
    met["latencia_media_ms"] = float(1000 * np.mean(tempos)) if tempos else None
    return met, tab


def avaliar(config: Config, clf_escopo, clf_tema, grafo, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Os três níveis de avaliação em um único dicionário (formato do ``avaliacao.json``)."""
    textos, intencoes = df["texto"].tolist(), df["intencao"].tolist()
    ponta, tabela = avaliar_ponta_a_ponta(grafo, textos, intencoes, config)
    return {
        "grupo": config.grupo,
        "tema": config.tema,
        "intencoes": config.nomes_intencoes,
        "escopo": avaliar_escopo(clf_escopo, textos, intencoes, config),
        "tema_oraculo": avaliar_tema(clf_tema, textos, intencoes, config),
        "ponta_a_ponta": ponta,
    }, tabela


def tabela_resumo(resultado: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Tabela (uma linha por nível) com acurácia e F1 macro, e uma série com as taxas do chatbot."""
    pa = resultado["ponta_a_ponta"]
    linhas = [
        ("escopo (1º estágio)", resultado["escopo"]),
        ("tema (2º estágio, oráculo)", resultado["tema_oraculo"]),
        ("ponta a ponta (grafo)", pa),
    ]
    tab = pd.DataFrame(
        [{"nível": n, "n": m.get("n"), "acurácia": m.get("acuracia"), "F1 macro": m.get("f1_macro")}
         for n, m in linhas]
    ).set_index("nível")
    extras = pd.Series({
        "falso aceite": pa["taxa_falso_aceite"],
        "falsa recusa": pa["taxa_falsa_recusa"],
        "esclarecimentos": pa["taxa_esclarecimento"],
        "latência média (ms)": pa["latencia_media_ms"],
    })
    return tab, extras
