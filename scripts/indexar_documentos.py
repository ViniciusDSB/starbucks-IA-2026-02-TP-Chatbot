#!/usr/bin/env python3
"""Cria (ou recria) o índice vetorial da base de conhecimento e testa buscas.

O índice é criado sozinho na primeira conversa com o LLM; este script serve para
(1) ver quantos trechos saíram de cada documento e (2) testar a BUSCA sem gastar
créditos do LLM - o melhor jeito de descobrir se faltam documentos.

Uso (a partir da raiz do repositório):

    python scripts/indexar_documentos.py                        # cria/atualiza e mostra o resumo
    python scripts/indexar_documentos.py --forcar               # recria do zero
    python scripts/indexar_documentos.py --buscar "posso renovar pelo celular?"
    python scripts/indexar_documentos.py --buscar "abre domingo?" --intencao horario
    python scripts/indexar_documentos.py --config exemplos/biblioteca/grupo.yaml --buscar "..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chatbot.config import carregar_config  # noqa: E402
from chatbot.conhecimento import BaseConhecimento  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grupo.yaml")
    ap.add_argument("--forcar", action="store_true", help="recria o índice mesmo sem mudanças")
    ap.add_argument("--buscar", default=None, help="pergunta de teste")
    ap.add_argument("--intencao", default=None, help="restringe a busca a uma intenção (+ geral)")
    ap.add_argument("-k", type=int, default=None, help="quantos trechos mostrar")
    args = ap.parse_args(argv)

    config = carregar_config(args.config)
    try:
        base = BaseConhecimento.carregar_ou_criar(config, forcar=args.forcar)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERRO] {e}")
        return 1
    if not args.buscar:
        print(base.resumo().to_string())
        sem_docs = [i for i in config.nomes_intencoes if i not in {t.intencao for t in base.trechos}]
        if sem_docs:
            print(f"\nIntenções sem documento próprio (só contam com geral/): {sem_docs}")
        return 0
    print(f"Pergunta: {args.buscar!r}" + (f"  (intenção: {args.intencao})" if args.intencao else ""))
    for i, (t, sim) in enumerate(base.buscar(args.buscar, intencao=args.intencao, k=args.k), 1):
        pag = f", p. {t.pagina}" if t.pagina else ""
        print(f"\n[{i}] similaridade {sim:.3f} | {t.fonte}{pag} | intenção: {t.intencao}")
        print("    " + t.texto.replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
