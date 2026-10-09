#!/usr/bin/env python3
"""Converse com o chatbot no terminal.

Uso (a partir da raiz do repositório):

    python scripts/conversar.py                                    # o bot do grupo
    python scripts/conversar.py --config exemplos/biblioteca/grupo.yaml
    python scripts/conversar.py --depurar                          # mostra rotas e confianças
    python scripts/conversar.py --sem-llm                          # respostas fixas, sem OpenRouter

Com a chave do OpenRouter configurada (arquivo .env), as respostas das intenções
são redigidas pelo LLM a partir dos documentos da pasta documentos/ (o índice
vetorial é criado na primeira vez); sem a chave, o bot usa as respostas fixas.

Digite 'sair' (ou Ctrl+D) para encerrar.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chatbot.config import carregar_config  # noqa: E402
from chatbot.grafo import Conversa  # noqa: E402
from chatbot.modelos import carregar_classificadores  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grupo.yaml")
    ap.add_argument("--depurar", action="store_true", help="mostra o caminho percorrido no grafo")
    ap.add_argument("--sem-llm", action="store_true", help="não usa o OpenRouter (respostas fixas)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    config = carregar_config(args.config)
    conversa = Conversa(config, *carregar_classificadores(config), usar_llm=not args.sem_llm)
    modo = f"LLM: {conversa.llm.modelo}" if conversa.llm else "respostas fixas"
    print(f"Chatbot: {config.tema}  [{modo}]  (digite 'sair' para encerrar)\n")
    while True:
        try:
            msg = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"sair", "exit", "quit"}:
            break
        if not msg:
            continue
        estado = conversa.enviar(msg)
        if args.depurar:
            linha = f"   [escopo={estado['escopo']} ({estado['confianca_escopo']:.2f})"
            if estado.get("intencao"):
                linha += f" -> intenção={estado['intencao']} ({estado['confianca_intencao']:.2f})"
            print(linha + f" -> rota={estado['rota']} | resposta {estado.get('fonte_resposta', 'fixa')}]")
            for t in estado.get("trechos") or []:
                pag = f", p. {t['pagina']}" if t.get("pagina") else ""
                print(f"     trecho {t['similaridade']:.2f}  {t['fonte']}{pag}")
        print(f"bot > {estado['resposta']}\n")
    if conversa.llm:
        print(f"Uso do LLM nesta conversa: {conversa.llm.uso}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
