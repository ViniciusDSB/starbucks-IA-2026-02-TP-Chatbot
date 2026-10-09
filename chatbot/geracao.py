"""Geração de frases de TREINO com um LLM (OpenRouter).

Gerar dados com LLM é permitido **apenas para o treino** e exige três cuidados:

1. **Revisar tudo.** O LLM erra o rótulo, repete frases, escreve de um jeito que
   nenhum estudante escreveria. Leia cada frase antes de passá-la ao ``treino.csv``.
2. **Declarar.** Toda geração fica registrada em ``dados/prompts_llm.md`` (modelo,
   data, prompt completo e quantas frases saíram). Esse registro é a base da seção
   de uso de IA do artigo - não o apague.
3. **Variar.** Um único prompt produz frases com o mesmo "sotaque". Mude as
   instruções (informal, com erros de digitação, pergunta longa, uma palavra só...)
   e compare, no relatório, modelos treinados com e sem esses dados.

Uso pelo script: ``python scripts/gerar_dados_llm.py --help``.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config
from .llm import ClienteLLM

ESTILOS = [
    "perguntas completas e educadas",
    "mensagens curtas e informais, como no WhatsApp, algumas sem acento ou com abreviações (vc, pq, q)",
    "mensagens com erros de digitação comuns e sem pontuação",
    "mensagens de estudantes ansiosos ou com pressa, explicando a situação em uma ou duas frases",
    "mensagens bem curtas, de 1 a 4 palavras",
]


def _prompt_intencao(config: Config, nome: str, n: int, estilo: str) -> str:
    i = config.intencoes[nome]
    outras = "; ".join(f"{o.descricao}" for k, o in config.intencoes.items() if k != nome)
    return (
        f"Contexto: um chatbot de atendimento sobre \"{config.tema}\".\n"
        f"Gere {n} mensagens DIFERENTES que um estudante universitário brasileiro enviaria "
        f"ao chatbot sobre o assunto: \"{i.descricao}\".\n"
        f"Estilo: {estilo}.\n"
        f"As mensagens NÃO podem ser sobre estes outros assuntos: {outras}.\n"
        "Varie o vocabulário e a estrutura. Não numere, não use aspas, não explique: "
        "escreva uma mensagem por linha."
    )


def _prompt_chitchat(config: Config, n: int, estilo: str) -> str:
    return (
        f"Gere {n} mensagens DIFERENTES de conversa social que alguém enviaria a um chatbot "
        f"(cumprimentos, agradecimentos, despedidas, elogios, reclamações genéricas, perguntas "
        f"sobre o próprio bot, risadas, 'ok', 'entendi'). Estilo: {estilo}. "
        "Não inclua nenhum pedido de informação. Uma mensagem por linha, sem numeração."
    )


def _prompt_fora(config: Config, n: int, estilo: str) -> str:
    return (
        f"Um chatbot atende SOMENTE sobre \"{config.tema}\" ({config.assuntos()}).\n"
        f"Gere {n} mensagens DIFERENTES que um estudante universitário enviaria e que o chatbot "
        "NÃO deve responder, porque são sobre OUTROS assuntos. Metade delas deve ser sobre "
        "assuntos VIZINHOS da vida universitária (outros setores, outros serviços, outras dúvidas "
        "acadêmicas) e usar palavras parecidas com as do tema, para confundir um classificador. "
        f"A outra metade, sobre assuntos do dia a dia. Estilo: {estilo}. "
        "Uma mensagem por linha, sem numeração."
    )


def _limpar_linhas(texto: str) -> list[str]:
    linhas = []
    for linha in texto.splitlines():
        linha = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", linha).strip().strip('"').strip()
        if 2 <= len(linha) <= 300:
            linhas.append(linha)
    return linhas


def gerar(config: Config, cliente: ClienteLLM, rotulos: list[str], n_por_estilo: int = 10,
          estilos: list[str] | None = None, registro: str | Path | None = None) -> pd.DataFrame:
    """Gera frases para cada rótulo em ``rotulos`` (intenções, 'chitchat', 'fora_dominio').

    Devolve um DataFrame ``texto,intencao,origem`` (origem = ``llm``) SEM duplicatas e
    acrescenta cada prompt usado ao arquivo de ``registro`` (Markdown).
    """
    estilos = estilos or ESTILOS
    linhas, log = [], []
    for rotulo in rotulos:
        for estilo in estilos:
            if rotulo == "chitchat":
                prompt = _prompt_chitchat(config, n_por_estilo, estilo)
            elif rotulo == "fora_dominio":
                prompt = _prompt_fora(config, n_por_estilo, estilo)
            else:
                prompt = _prompt_intencao(config, rotulo, n_por_estilo, estilo)
            resposta = cliente.conversar([{"role": "user", "content": prompt}],
                                         temperatura=0.9, max_tokens=60 * n_por_estilo)
            frases = _limpar_linhas(resposta)
            linhas += [{"texto": f, "intencao": rotulo, "origem": "llm"} for f in frases]
            log.append((rotulo, estilo, prompt, len(frases)))
            print(f"  {rotulo:20s} | {len(frases):3d} frases | {estilo[:50]}")

    df = pd.DataFrame(linhas, columns=["texto", "intencao", "origem"])
    df = df[~df["texto"].str.lower().duplicated()].reset_index(drop=True)

    if registro:
        registro = Path(registro)
        registro.parent.mkdir(parents=True, exist_ok=True)
        novo = not registro.exists()
        with open(registro, "a", encoding="utf-8") as f:
            if novo:
                f.write("# Registro de geração de dados com LLM\n\n"
                        "Gerado automaticamente por `scripts/gerar_dados_llm.py`. Use este arquivo "
                        "na seção de uso de IA do artigo. **Não apague.**\n")
            f.write(f"\n## {datetime.now():%d/%m/%Y %H:%M} - modelo `{cliente.modelo}`\n\n")
            f.write(f"Frases geradas (sem duplicatas): {len(df)}\n")
            for rotulo, estilo, prompt, n in log:
                f.write(f"\n### `{rotulo}` - {n} frases\n\n```text\n{prompt}\n```\n")
    return df
