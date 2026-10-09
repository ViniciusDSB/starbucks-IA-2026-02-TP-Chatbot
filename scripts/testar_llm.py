#!/usr/bin/env python3
"""Testa a chave do OpenRouter: faz UMA pergunta curta ao modelo do grupo.yaml.

Uso (a partir da raiz do repositório):

    python scripts/testar_llm.py
    python scripts/testar_llm.py --config exemplos/biblioteca/grupo.yaml

Se der certo, aparece a resposta do modelo e quantos tokens foram gastos. Se der
errado, a mensagem explica o motivo mais provável (sem chave, chave inválida,
créditos esgotados, modelo inexistente, sem internet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chatbot.config import MODELO_PADRAO, carregar_config  # noqa: E402
from chatbot.llm import ClienteLLM, ErroLLM  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grupo.yaml")
    args = ap.parse_args(argv)
    try:
        config = carregar_config(args.config)
        modelo = config.llm["modelo"]
    except Exception:  # noqa: BLE001 - grupo.yaml ainda não preenchido
        config, modelo = None, MODELO_PADRAO
    try:
        cliente = (ClienteLLM.da_config(config) if config else ClienteLLM(modelo=modelo))
        print(f"Chave encontrada: {cliente.chave_mascarada}")
        print(f"Modelo: {cliente.modelo}\nPerguntando...")
        texto = cliente.conversar([
            {"role": "system", "content": "Responda em português, em uma frase curta."},
            {"role": "user", "content": "Diga olá para a turma de Inteligência Artificial da UFMS."},
        ])
    except ErroLLM as e:
        print(f"[ERRO] {e}")
        return 1
    print(f"\nResposta: {texto}\nUso: {cliente.uso}\n\nTudo certo com o OpenRouter!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
