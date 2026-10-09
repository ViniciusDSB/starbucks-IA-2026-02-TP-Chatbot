#!/usr/bin/env python3
"""Avaliação padronizada do chatbot - o MESMO script usado na correção.

Uso (a partir da raiz do repositório):

    python scripts/avaliar.py                                   # grupo.yaml + dados/teste.csv
    python scripts/avaliar.py --config exemplos/biblioteca/grupo.yaml
    python scripts/avaliar.py --teste outro_teste.csv --saida resultados/outro.json

Gera:
    resultados/avaliacao.json   - todas as métricas (formato lido pelo professor)
                                  (a pasta resultados/ fica ao lado do grupo.yaml usado)
    resultados/avaliacao_previsoes.csv - uma linha por mensagem do teste, com a rota escolhida

Na correção, o professor roda este script com o teste do grupo e com conjuntos de
teste próprios (não divulgados). Por isso, **não altere este arquivo**: melhorias
vão em ``chatbot/`` e no notebook.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chatbot.avaliacao import avaliar, tabela_resumo  # noqa: E402
from chatbot.config import carregar_config  # noqa: E402
from chatbot.dados import carregar_dados, validar_dados  # noqa: E402
from chatbot.grafo import construir_grafo  # noqa: E402
from chatbot.modelos import carregar_classificadores  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grupo.yaml", help="arquivo de configuração (padrão: grupo.yaml)")
    ap.add_argument("--teste", default=None, help="CSV de teste (padrão: o do grupo.yaml)")
    ap.add_argument("--saida", default=None, help="JSON de saída (padrão: resultados/avaliacao.json)")
    ap.add_argument("--rotulos-desconhecidos", choices=["erro", "fora_dominio"], default="erro",
                    help="o que fazer com rótulos de teste que não são intenções do grupo "
                         "(o professor usa 'fora_dominio' ao testar com mensagens de outros grupos)")
    ap.add_argument("--quieto", action="store_true", help="não imprime a tabela de erros")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    config = carregar_config(args.config)
    caminho_teste = Path(args.teste) if args.teste else config.teste
    df = carregar_dados(caminho_teste)

    if args.rotulos_desconhecidos == "fora_dominio":
        conhecidos = set(config.nomes_intencoes) | {"chitchat", "fora_dominio"}
        df.loc[~df["intencao"].isin(conhecidos), "intencao"] = "fora_dominio"
    problemas = [p for p in validar_dados(df, config) if "duplicado" not in p]
    if problemas:
        print(f"[ERRO] {caminho_teste}: " + "; ".join(problemas))
        return 2

    clf_escopo, clf_tema = carregar_classificadores(config)
    grafo = construir_grafo(config, clf_escopo, clf_tema)
    resultado, previsoes = avaliar(config, clf_escopo, clf_tema, grafo, df)
    resultado["arquivo_teste"] = str(caminho_teste)

    saida = Path(args.saida) if args.saida else config.pasta / "resultados" / "avaliacao.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    previsoes.to_csv(saida.parent / f"{saida.stem}_previsoes.csv", index=False)

    tab, extras = tabela_resumo(resultado)
    print(f"\nGrupo: {config.grupo}  |  Tema: {config.tema}  |  Teste: {caminho_teste} ({len(df)} mensagens)\n")
    print(tab.round(3).to_string())
    print()
    print(extras.round(3).to_string())
    if not args.quieto:
        erros = previsoes[~previsoes["acertou"]]
        if len(erros):
            print(f"\nErros ponta a ponta ({len(erros)}):")
            print(erros[["texto", "real", "rota"]].to_string(index=False, max_colwidth=60))
    print(f"\nResultados salvos em {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
