#!/usr/bin/env python3
"""Baixa a parte em português da base MASSIVE (Amazon) para usar como exemplos
NEGATIVOS (chitchat e fora_dominio) no treino.

MASSIVE (FitzGerald et al., 2022) tem ~16 mil comandos de assistente virtual
(alarmes, música, clima, culinária, piadas...) traduzidos para 51 idiomas. Aqui
usamos a versão em português (pt-PT, português europeu) publicada pelo projeto
SetFit no Hugging Face. Licença: CC BY 4.0 - cite no relatório:

    FitzGerald, J. et al. MASSIVE: A 1M-Example Multilingual Natural Language
    Understanding Dataset with 51 Typologically-Diverse Languages. ACL 2023.

Uso (precisa de internet; funciona no Colab):

    python scripts/baixar_massive.py

Gera em dados/geral/:
    massive_chitchat.csv     - intenções general_greet, general_joke, general_quirky
    massive_fora_dominio.csv - todas as demais intenções (assistente doméstico)

Atenção, para o relatório:
* são negativos FÁCEIS - quase nada ali se parece com o seu domínio. Os negativos
  DIFÍCEIS (assuntos vizinhos ao seu tema) precisam ser escritos pelo grupo;
* é português de Portugal ("autocarro", "telemóvel"), o que muda a distribuição
  das palavras em relação ao que os usuários do seu bot escreveriam;
* não use a MASSIVE no TESTE: o teste deve ser escrito por pessoas (ver enunciado).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
URL = "https://huggingface.co/datasets/SetFit/amazon_massive_intent_pt-PT/resolve/main/{parte}.jsonl"
CHITCHAT = {"general_greet", "general_joke", "general_quirky"}


def baixar(parte: str) -> pd.DataFrame:
    print(f"baixando {parte}...", flush=True)
    with urllib.request.urlopen(URL.format(parte=parte), timeout=60) as resp:
        linhas = [json.loads(l) for l in resp.read().decode("utf-8").splitlines() if l.strip()]
    return pd.DataFrame(linhas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-fora", type=int, default=1500,
                    help="quantas frases fora_dominio manter (amostra estratificada por intenção)")
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args(argv)

    try:
        df = pd.concat([baixar(p) for p in ("train", "validation", "test")], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        print(f"Falha no download ({e}). Verifique a conexão ou rode no Colab.")
        return 1

    df = df.rename(columns={"text": "texto", "label_text": "massive_intencao"})
    df = df.drop_duplicates("texto")
    chit = df[df["massive_intencao"].isin(CHITCHAT)].copy()
    fora = df[~df["massive_intencao"].isin(CHITCHAT)].copy()
    frac = min(1.0, args.max_fora / max(len(fora), 1))
    fora = fora.groupby("massive_intencao", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=args.semente))

    destino = RAIZ / "dados" / "geral"
    destino.mkdir(parents=True, exist_ok=True)
    for nome, parte, rotulo in (("massive_chitchat.csv", chit, "chitchat"),
                                ("massive_fora_dominio.csv", fora, "fora_dominio")):
        saida = parte.assign(intencao=rotulo, origem="massive")[["texto", "intencao", "origem", "massive_intencao"]]
        saida.to_csv(destino / nome, index=False)
        print(f"{destino / nome}: {len(saida)} frases")
    print("\nDica: a coluna extra 'massive_intencao' ajuda a inspecionar os exemplos; "
          "remova-a ao juntar com o seu treino.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
