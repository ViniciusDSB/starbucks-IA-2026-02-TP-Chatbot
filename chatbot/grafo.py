"""O chatbot como um grafo de estados (LangGraph).

                      ┌──────────────────────┐
     mensagem ──────▶ │  classificar_escopo  │  (1º classificador)
                      └──────────┬───────────┘
          chitchat ┌─────────────┼──────────────┐ fora_dominio
                   ▼             │ dominio      ▼
        responder_chitchat       ▼            recusar
                      ┌──────────────────────┐
                      │   classificar_tema   │  (2º classificador)
                      └──────────┬───────────┘
          confiança < limiar ┌───┴──────┬─────────── ... ──┐
                             ▼          ▼                  ▼
                        esclarecer  intencao_A   ...   intencao_N
                             │          │                  │
                             └──────────┴──────▶ END ◀─────┘

Cada caixa é um **nó** (uma função Python que recebe o estado e devolve as
alterações). As setas que saem dos classificadores são **arestas condicionais**:
uma função lê o estado e escolhe o próximo nó. Os nós das intenções são criados
automaticamente a partir do ``grupo.yaml``.

**Onde entram o LLM e os documentos (RAG).** Se ``construir_grafo`` receber um
``llm`` (``chatbot.llm.ClienteLLM``) e uma ``base`` (``chatbot.conhecimento.
BaseConhecimento``), o nó de cada intenção (1) busca na base os trechos dos
documentos mais parecidos com a pergunta - só entre os da intenção prevista e os
gerais - e (2) pede ao LLM que redija a resposta usando apenas esses trechos. Sem
``llm`` (ou se a chamada falhar), o nó usa a resposta fixa do ``grupo.yaml``. A
ROTA é sempre decidida pelos classificadores do grupo - o LLM só escreve o texto.

**Contrato com o script de avaliação** - não quebre estes pontos:

* ``construir_grafo(config, clf_escopo, clf_tema, checkpointer=None, llm=None)``
  devolve o grafo compilado (a avaliação automática chama SEM ``llm``);
* ``grafo.invoke({"mensagem": texto})`` termina com o estado contendo ``rota``
  (o nome da intenção, ``chitchat``, ``fora_dominio`` ou ``esclarecer``) e
  ``resposta`` (o texto que o usuário veria).

Fora isso, o grupo pode e deve melhorar o grafo: respostas mais ricas, novos nós
(ex.: pedir um dado que falta), memória da conversa, um 3º classificador...
"""
from __future__ import annotations

import operator
import re
import uuid
from typing import Annotated, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from .config import Config
from .rotulos import CHITCHAT, DOMINIO, ESCLARECER, FORA_DOMINIO
from .texto import normalizar


class EstadoChat(TypedDict, total=False):
    """Tudo o que o grafo sabe durante o processamento de uma mensagem."""

    mensagem: str                  # entrada do usuário
    escopo: str                    # saída do 1º classificador
    confianca_escopo: float
    intencao: str | None           # saída do 2º classificador
    confianca_intencao: float
    candidatas: list               # [(intenção, probabilidade), ...] em ordem decrescente
    rota: str                      # nó final pelo qual a mensagem saiu (usado na avaliação)
    resposta: str                  # texto devolvido ao usuário
    fonte_resposta: str            # "fixa", "llm" ou "fixa (falha no LLM: ...)"
    trechos: list                  # trechos recuperados dos documentos (fonte, similaridade, texto)
    # Histórico da conversa. O ``operator.add`` é um *reducer*: em vez de
    # sobrescrever, cada nó que devolve "historico" ACRESCENTA itens à lista.
    historico: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def prever_com_confianca(clf, texto: str) -> tuple[str, float, list]:
    """Rótulo previsto, sua probabilidade e a lista ordenada de candidatas.

    Se o classificador não tiver ``predict_proba`` (ex.: SVM linear sem calibração),
    a confiança é 1,0 e só a classe prevista aparece entre as candidatas.
    """
    if hasattr(clf, "predict_proba"):
        try:
            proba = np.asarray(clf.predict_proba([texto]))[0]
            ordem = np.argsort(proba)[::-1]
            classes = [str(c) for c in clf.classes_]
            candidatas = [(classes[i], float(proba[i])) for i in ordem]
            return candidatas[0][0], candidatas[0][1], candidatas
        except (AttributeError, NotImplementedError):
            pass
    rotulo = str(clf.predict([texto])[0])
    return rotulo, 1.0, [(rotulo, 1.0)]


# Expressões regulares aplicadas ao texto normalizado (sem acento, minúsculo).
# A ordem importa: "oi, obrigado!" é agradecimento, não saudação.
_GATILHOS_CHITCHAT = {
    "agradecimento": r"\b(obrigad\w*|brigad\w*|valeu|vlw|agradec\w*|thanks)\b",
    "despedida": r"\b(tchau|ate logo|ate mais|ate breve|falou|flw|adeus|fui)\b",
    "saudacao": r"\b(oi+|ola|bom dia|boa tarde|boa noite|e ai|eai|opa|hello|salve)\b",
}


def tipo_chitchat(texto: str) -> str:
    """Regra simples (palavras-gatilho) para escolher a resposta social.

    É de propósito uma regra ingênua: trocá-la por um 3º classificador é uma das
    melhorias possíveis do bot.
    """
    t = normalizar(texto)
    for tipo, padrao in _GATILHOS_CHITCHAT.items():
        if re.search(padrao, t):
            return tipo
    return "padrao"


def _registro(estado: EstadoChat, rota: str, resposta: str, fonte: str = "fixa") -> dict:
    """Atualização padrão de um nó terminal: rota, resposta e uma linha no histórico."""
    return {
        "rota": rota,
        "resposta": resposta,
        "fonte_resposta": fonte,
        "historico": [{"usuario": estado["mensagem"], "bot": resposta, "rota": rota}],
    }


# ---------------------------------------------------------------------------
# Construção do grafo
# ---------------------------------------------------------------------------

def nome_no(intencao: str) -> str:
    """Nome do nó do grafo que responde a uma intenção."""
    return f"intencao_{intencao}"


def construir_grafo(config: Config, clf_escopo, clf_tema, checkpointer=None, llm=None, base=None):
    """Monta e compila o grafo do chatbot.

    ``checkpointer`` (ex.: ``langgraph.checkpoint.memory.InMemorySaver()``) guarda o
    estado entre mensagens de uma mesma conversa (``thread_id``); sem ele, cada
    ``invoke`` começa do zero - que é como a avaliação automática roda.

    ``llm`` (um ``chatbot.llm.ClienteLLM``) é opcional: com ele, as respostas dos nós
    listados em ``llm.usar_em`` no ``grupo.yaml`` são redigidas pelo LLM.

    ``base`` (uma ``chatbot.conhecimento.BaseConhecimento``) é opcional: com ela, as
    respostas das intenções são baseadas nos trechos recuperados dos documentos.
    """
    from .llm import ErroLLM, prompt_chitchat, prompt_fora_dominio, prompt_intencao

    assuntos = config.assuntos()
    respostas = config.respostas
    usar_em = set(config.llm.get("usar_em", [])) if llm is not None else set()

    def redigir(estado: EstadoChat, rota: str, onde: str, montar_prompt, resposta_fixa: str,
                extras: dict | None = None) -> dict:
        """Tenta o LLM; se não houver LLM ou a chamada falhar, usa a resposta fixa."""
        extras = extras or {}
        if onde not in usar_em:
            return {**_registro(estado, rota, resposta_fixa), **extras}
        try:
            texto = llm.conversar(montar_prompt(estado["mensagem"], estado.get("historico", [])))
            return {**_registro(estado, rota, texto, fonte="llm"), **extras}
        except ErroLLM as e:
            return {**_registro(estado, rota, resposta_fixa, fonte=f"fixa (falha no LLM: {e})"), **extras}

    # --- nós dos classificadores -------------------------------------------
    def classificar_escopo(estado: EstadoChat) -> dict:
        rotulo, conf, _ = prever_com_confianca(clf_escopo, estado["mensagem"])
        # limpa campos que podem ter sobrado da mensagem anterior (com checkpointer)
        return {"escopo": rotulo, "confianca_escopo": conf, "intencao": None,
                "confianca_intencao": 0.0, "candidatas": [], "trechos": []}

    def classificar_tema(estado: EstadoChat) -> dict:
        rotulo, conf, candidatas = prever_com_confianca(clf_tema, estado["mensagem"])
        return {"intencao": rotulo, "confianca_intencao": conf, "candidatas": candidatas}

    # --- arestas condicionais ----------------------------------------------
    def rotear_escopo(estado: EstadoChat) -> str:
        return estado["escopo"]  # "dominio", "chitchat" ou "fora_dominio"

    def rotear_tema(estado: EstadoChat) -> str:
        if estado["confianca_intencao"] < config.limiar_tema:
            return ESCLARECER
        return estado["intencao"]

    # --- nós terminais ------------------------------------------------------
    def responder_chitchat(estado: EstadoChat) -> dict:
        modelos = respostas[CHITCHAT]
        fixa = modelos.get(tipo_chitchat(estado["mensagem"]), modelos["padrao"]).format(assuntos=assuntos)
        return redigir(estado, CHITCHAT, "chitchat",
                       lambda m, h: prompt_chitchat(config, m, h), fixa)

    def recusar(estado: EstadoChat) -> dict:
        fixa = respostas[FORA_DOMINIO].format(assuntos=assuntos)
        return redigir(estado, FORA_DOMINIO, "fora_dominio",
                       lambda m, h: prompt_fora_dominio(config, m, h), fixa)

    def esclarecer(estado: EstadoChat) -> dict:
        cands = [c for c, _ in estado.get("candidatas", [])] + config.nomes_intencoes
        op1, op2 = cands[0], next(c for c in cands if c != cands[0])
        texto = respostas[ESCLARECER].format(
            opcao1=config.intencoes[op1].descricao, opcao2=config.intencoes[op2].descricao
        )
        return _registro(estado, ESCLARECER, texto)

    def criar_no_intencao(nome: str):
        resposta = config.intencoes[nome].resposta

        def responder(estado: EstadoChat) -> dict:
            # RAG: recupera os trechos mais parecidos (só da intenção prevista + gerais)
            trechos = base.buscar(estado["mensagem"], intencao=nome) if base is not None and usar_em else []
            registro = [{"fonte": t.fonte, "pagina": t.pagina, "intencao": t.intencao,
                         "similaridade": round(sim, 3), "texto": t.texto} for t, sim in trechos]
            return redigir(estado, nome, "intencoes",
                           lambda m, h: prompt_intencao(config, nome, m, h, trechos), resposta,
                           extras={"trechos": registro})

        responder.__name__ = nome_no(nome)
        return responder

    # --- montagem -----------------------------------------------------------
    g = StateGraph(EstadoChat)
    g.add_node("classificar_escopo", classificar_escopo)
    g.add_node("classificar_tema", classificar_tema)
    g.add_node("responder_chitchat", responder_chitchat)
    g.add_node("recusar", recusar)
    g.add_node("esclarecer", esclarecer)
    for nome in config.nomes_intencoes:
        g.add_node(nome_no(nome), criar_no_intencao(nome))

    g.add_edge(START, "classificar_escopo")
    g.add_conditional_edges(
        "classificar_escopo",
        rotear_escopo,
        {DOMINIO: "classificar_tema", CHITCHAT: "responder_chitchat", FORA_DOMINIO: "recusar"},
    )
    mapa_tema = {nome: nome_no(nome) for nome in config.nomes_intencoes}
    mapa_tema[ESCLARECER] = "esclarecer"
    g.add_conditional_edges("classificar_tema", rotear_tema, mapa_tema)

    for terminal in ["responder_chitchat", "recusar", "esclarecer", *mapa_tema.values()]:
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Conversa
# ---------------------------------------------------------------------------

class Conversa:
    """Atalho para conversar com o grafo mantendo o histórico (um ``thread_id``).

    ``usar_llm=True`` tenta criar o cliente do OpenRouter com a chave do grupo e
    carregar a base de conhecimento (criando o índice na primeira vez); se faltar a
    chave ou os documentos, avisa e segue com as respostas fixas.
    """

    def __init__(self, config: Config, clf_escopo, clf_tema, usar_llm: bool = False, llm=None, base=None):
        from langgraph.checkpoint.memory import InMemorySaver

        if usar_llm and llm is None:
            from .llm import ClienteLLM, ErroLLM

            try:
                llm = ClienteLLM.da_config(config)
            except ErroLLM as e:
                print(f"[aviso] LLM desligado: {e}")
        if llm is not None and base is None:
            from .conhecimento import BaseConhecimento

            try:
                base = BaseConhecimento.carregar_ou_criar(config)
            except (FileNotFoundError, ValueError) as e:
                print(f"[aviso] sem base de conhecimento: {e}")
        self.llm, self.base = llm, base
        self.grafo = construir_grafo(config, clf_escopo, clf_tema,
                                     checkpointer=InMemorySaver(), llm=llm, base=base)
        self.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def enviar(self, mensagem: str) -> EstadoChat:
        return self.grafo.invoke({"mensagem": mensagem}, self.thread)

    @property
    def historico(self) -> list:
        return self.grafo.get_state(self.thread).values.get("historico", [])
