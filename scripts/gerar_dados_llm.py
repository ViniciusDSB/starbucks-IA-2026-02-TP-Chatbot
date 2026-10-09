#!/usr/bin/env python3
"""Gera frases de TREINO com um LLM do OpenRouter (precisa da chave do grupo).

Uso (a partir da raiz do repositório):

    # 10 frases x 5 estilos = ~50 frases para CADA intenção do grupo.yaml
    python scripts/gerar_dados_llm.py

    # só algumas intenções, com 6 frases por estilo
    python scripts/gerar_dados_llm.py --rotulos renovacao multa --n 6

    # negativos: conversa social e fora do domínio (inclui negativos difíceis)
    python scripts/gerar_dados_llm.py --rotulos chitchat fora_dominio

Saídas:
    dados/gerados_llm.csv  - as frases geradas (texto,intencao,origem=llm), ACRESCENTADAS
                             ao arquivo se ele já existir
    dados/prompts_llm.md   - registro de todos os prompts usados (para o artigo)

As frases NÃO vão direto para o treino. Abra dados/gerados_llm.csv, revise linha a
linha (apague as ruins, corrija rótulos errados) e só então copie as boas para
dados/treino.csv. NUNCA use frases geradas no teste.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chatbot.config import carregar_config  # noqa: E402
from chatbot.geracao import gerar  # noqa: E402
from chatbot.llm import ClienteLLM, ErroLLM  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grupo.yaml")
    ap.add_argument("--rotulos", nargs="*", default=None,
                    help="intenções (ou chitchat/fora_dominio) a gerar; padrão: todas as intenções")
    ap.add_argument("--n", type=int, default=10, help="frases por estilo (há 5 estilos)")
    ap.add_argument("--modelo", default=None, help="modelo do OpenRouter (padrão: o do grupo.yaml)")
    ap.add_argument("--saida", default="dados/gerados_llm.csv")
    args = ap.parse_args(argv)

    config = carregar_config(args.config)
    rotulos = args.rotulos or config.nomes_intencoes
    validos = set(config.nomes_intencoes) | {"chitchat", "fora_dominio"}
    invalidos = [r for r in rotulos if r not in validos]
    if invalidos:
        print(f"Rótulos desconhecidos: {invalidos}. Válidos: {sorted(validos)}")
        return 2
    try:
        cliente = ClienteLLM.da_config(config)
        if args.modelo:
            cliente.modelo = args.modelo
    except ErroLLM as e:
        print(f"[ERRO] {e}")
        return 1

    print(f"Gerando com {cliente.modelo} (chave {cliente.chave_mascarada})...")
    try:
        novos = gerar(config, cliente, rotulos, n_por_estilo=args.n,
                      registro=config.pasta / "dados" / "prompts_llm.md")
    except ErroLLM as e:
        print(f"[ERRO] {e}")
        return 1

    saida = config.pasta / args.saida
    if saida.exists():
        antigos = pd.read_csv(saida, dtype=str, keep_default_na=False)
        novos = pd.concat([antigos, novos], ignore_index=True)
        novos = novos[~novos["texto"].str.lower().duplicated()]
    novos.to_csv(saida, index=False)
    print(f"\n{len(novos)} frases em {saida} (uso: {cliente.uso})")
    print("PRÓXIMO PASSO: revise o arquivo antes de copiar as frases para dados/treino.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
