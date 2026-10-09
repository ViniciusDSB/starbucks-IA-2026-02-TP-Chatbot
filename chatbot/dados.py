"""Leitura, validação e resumo dos arquivos de dados (``treino.csv`` e ``teste.csv``).

Formato (CSV com cabeçalho, UTF-8, separador vírgula):

    texto,intencao,origem
    "posso renovar meu livro pelo celular?",renovacao,humano
    "bom dia!",chitchat,humano
    "qual a previsão do tempo amanhã?",fora_dominio,massive

* ``texto``    - a mensagem do usuário, exatamente como seria digitada;
* ``intencao`` - uma intenção do domínio, ``chitchat`` ou ``fora_dominio``;
* ``origem``   - de onde veio a frase (ver ``chatbot.rotulos.ORIGENS``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .rotulos import CHITCHAT, DOMINIO, FORA_DOMINIO, ORIGENS

COLUNAS = ["texto", "intencao", "origem"]


def carregar_dados(caminho: str | Path) -> pd.DataFrame:
    """Lê um CSV de dados e normaliza espaços. Não valida rótulos (ver ``validar_dados``)."""
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"{caminho}: colunas ausentes {faltando}. Esperado: {COLUNAS}")
    df = df[COLUNAS].copy()
    for c in COLUNAS:
        df[c] = df[c].str.strip()
    return df


def rotulo_escopo(intencoes, config: Config) -> list[str]:
    """Converte rótulos de intenção em rótulos do 1º estágio (dominio/chitchat/fora_dominio)."""
    dominio = set(config.nomes_intencoes)
    saida = []
    for r in intencoes:
        if r in dominio:
            saida.append(DOMINIO)
        elif r in (CHITCHAT, FORA_DOMINIO):
            saida.append(r)
        else:
            raise ValueError(f"Rótulo desconhecido: '{r}'")
    return saida


def adicionar_escopo(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Devolve uma cópia do DataFrame com a coluna ``escopo`` derivada de ``intencao``."""
    df = df.copy()
    df["escopo"] = rotulo_escopo(df["intencao"], config)
    return df


def validar_dados(df: pd.DataFrame, config: Config) -> list[str]:
    """Lista de problemas encontrados (lista vazia = arquivo válido)."""
    problemas = []
    validos = set(config.nomes_intencoes) | {CHITCHAT, FORA_DOMINIO}
    desconhecidos = sorted(set(df["intencao"]) - validos)
    if desconhecidos:
        problemas.append(f"rótulos que não estão no grupo.yaml: {desconhecidos}")
    origens = sorted(set(df["origem"]) - ORIGENS)
    if origens:
        problemas.append(f"valores inválidos na coluna 'origem': {origens} (aceitos: {sorted(ORIGENS)})")
    vazios = int((df["texto"] == "").sum())
    if vazios:
        problemas.append(f"{vazios} linha(s) com texto vazio")
    dup = int(df["texto"].str.lower().duplicated().sum())
    if dup:
        problemas.append(f"{dup} texto(s) duplicado(s) dentro do arquivo")
    return problemas


def vazamento(treino: pd.DataFrame, teste: pd.DataFrame) -> list[str]:
    """Frases do teste que aparecem (ignorando maiúsculas/espaços) também no treino."""
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    vistos = set(treino["texto"].map(norm))
    return [t for t in teste["texto"] if norm(t) in vistos]


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela intenção × origem com totais - útil para o relatório."""
    tab = pd.crosstab(df["intencao"], df["origem"], margins=True, margins_name="total")
    return tab
