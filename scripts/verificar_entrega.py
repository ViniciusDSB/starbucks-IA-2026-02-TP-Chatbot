#!/usr/bin/env python3
"""Confere se o repositório do grupo está pronto para a correção automática.

Uso (a partir da raiz do repositório):

    python scripts/verificar_entrega.py

Cada item sai como OK, AVISO ou ERRO. Uma entrega com qualquer ERRO não passa
pela avaliação automática (ver critérios C6 e C10 do enunciado). Rode antes de
cada commit importante - e obrigatoriamente antes de entregar.

Os mínimos verificados aqui são os do enunciado; se houver divergência, vale o
enunciado (e avise o professor).
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from chatbot.config import ErroConfig, carregar_config  # noqa: E402
from chatbot.dados import carregar_dados, validar_dados, vazamento  # noqa: E402
from chatbot.rotulos import CHITCHAT, CLASSES_ESCOPO, ESCLARECER, FORA_DOMINIO  # noqa: E402

# ---- mínimos do enunciado --------------------------------------------------
INTENCOES_MIN, INTENCOES_MAX = 5, 10
INTEGRANTES_MIN, INTEGRANTES_MAX = 3, 4
TREINO_POR_INTENCAO = 50
TREINO_CHITCHAT = 100
TREINO_FORA = 150
TREINO_FRACAO_HUMANA = 0.20          # origem "humano" ou "coletado"
TESTE_POR_INTENCAO = 15
TESTE_CHITCHAT = 30
TESTE_FORA = 40
TESTE_ORIGENS = {"humano", "coletado"}
DOCUMENTOS_MIN = 5


class Relatorio:
    def __init__(self):
        self.itens = []

    def ok(self, msg):
        self.itens.append(("OK", msg))

    def aviso(self, msg):
        self.itens.append(("AVISO", msg))

    def erro(self, msg):
        self.itens.append(("ERRO", msg))

    def imprimir(self):
        simbolo = {"OK": "[ OK  ]", "AVISO": "[AVISO]", "ERRO": "[ERRO ]"}
        for nivel, msg in self.itens:
            print(f"{simbolo[nivel]} {msg}")
        n_erros = sum(1 for n, _ in self.itens if n == "ERRO")
        n_avisos = sum(1 for n, _ in self.itens if n == "AVISO")
        print(f"\n{n_erros} erro(s), {n_avisos} aviso(s).")
        print("Entrega PRONTA para a avaliação automática." if n_erros == 0
              else "Corrija os ERROS antes de entregar.")
        return n_erros


def contar(df, rotulo):
    return int((df["intencao"] == rotulo).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grupo.yaml")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    r = Relatorio()

    # 1. configuração ---------------------------------------------------------
    try:
        config = carregar_config(args.config)
        r.ok(f"grupo.yaml lido: grupo '{config.grupo}', tema '{config.tema}'")
    except (ErroConfig, OSError) as e:
        r.erro(f"grupo.yaml: {e}")
        return r.imprimir()

    n = len(config.intencoes)
    (r.ok if INTENCOES_MIN <= n <= INTENCOES_MAX else r.erro)(
        f"{n} intenções definidas (exigido: {INTENCOES_MIN} a {INTENCOES_MAX})")

    integrantes = [i for i in config.integrantes if isinstance(i, dict) and i.get("nome")]
    (r.ok if INTEGRANTES_MIN <= len(integrantes) <= INTEGRANTES_MAX else r.erro)(
        f"{len(integrantes)} integrante(s) com nome (exigido: {INTEGRANTES_MIN} a {INTEGRANTES_MAX})")
    if any(not str(i.get("rga", "")).strip() for i in integrantes):
        r.aviso("há integrante sem RGA no grupo.yaml")
    if "TODO" in config.arquivo.read_text(encoding="utf-8"):
        r.erro("o grupo.yaml ainda contém marcações TODO")

    # 2. dados ----------------------------------------------------------------
    dados = {}
    for papel, caminho in (("treino", config.treino), ("teste", config.teste)):
        try:
            df = carregar_dados(caminho)
        except Exception as e:  # noqa: BLE001
            r.erro(f"{papel}: não foi possível ler {caminho}: {e}")
            continue
        problemas = validar_dados(df, config)
        for p in problemas:
            (r.aviso if "duplicado" in p else r.erro)(f"{papel}.csv: {p}")
        if not problemas:
            r.ok(f"{papel}.csv: {len(df)} mensagens, formato válido")
        dados[papel] = df

    if "treino" in dados:
        df = dados["treino"]
        poucos = {i: contar(df, i) for i in config.nomes_intencoes if contar(df, i) < TREINO_POR_INTENCAO}
        (r.erro if poucos else r.ok)(
            f"treino: mínimo de {TREINO_POR_INTENCAO} exemplos por intenção"
            + (f" - abaixo: {poucos}" if poucos else ""))
        for rot, minimo in ((CHITCHAT, TREINO_CHITCHAT), (FORA_DOMINIO, TREINO_FORA)):
            c = contar(df, rot)
            (r.ok if c >= minimo else r.erro)(f"treino: {c} exemplos de {rot} (mínimo {minimo})")
        frac = df["origem"].isin(["humano", "coletado"]).mean() if len(df) else 0
        (r.ok if frac >= TREINO_FRACAO_HUMANA else r.erro)(
            f"treino: {frac:.0%} das frases escritas por pessoas ou coletadas "
            f"(mínimo {TREINO_FRACAO_HUMANA:.0%})")

    if "teste" in dados:
        df = dados["teste"]
        poucos = {i: contar(df, i) for i in config.nomes_intencoes if contar(df, i) < TESTE_POR_INTENCAO}
        (r.erro if poucos else r.ok)(
            f"teste: mínimo de {TESTE_POR_INTENCAO} exemplos por intenção"
            + (f" - abaixo: {poucos}" if poucos else ""))
        for rot, minimo in ((CHITCHAT, TESTE_CHITCHAT), (FORA_DOMINIO, TESTE_FORA)):
            c = contar(df, rot)
            (r.ok if c >= minimo else r.erro)(f"teste: {c} exemplos de {rot} (mínimo {minimo})")
        ruins = sorted(set(df["origem"]) - TESTE_ORIGENS)
        (r.erro if ruins else r.ok)(
            "teste: todas as frases escritas por pessoas ou coletadas"
            + (f" - origens não permitidas no teste: {ruins}" if ruins else ""))

    if "treino" in dados and "teste" in dados:
        vaz = vazamento(dados["treino"], dados["teste"])
        (r.erro if vaz else r.ok)(
            "nenhuma frase do teste repetida no treino"
            + (f" - {len(vaz)} repetida(s), ex.: {vaz[:3]}" if vaz else ""))

    # 3. modelos e grafo ------------------------------------------------------
    try:
        from chatbot.grafo import construir_grafo
        from chatbot.modelos import carregar_classificadores, checar_interface

        clf_escopo, clf_tema = carregar_classificadores(config)
        checar_interface(clf_escopo, CLASSES_ESCOPO, "escopo")
        checar_interface(clf_tema, config.nomes_intencoes, "tema")
        r.ok("classificadores carregados e com as classes corretas")
        for nome, clf in (("escopo", clf_escopo), ("tema", clf_tema)):
            if not hasattr(clf, "predict_proba"):
                r.aviso(f"classificador de {nome} sem predict_proba: limiar e esclarecimento não funcionam")
        versoes = config.modelo_escopo.parent / "versoes.json"
        if not config.carregador and not versoes.exists():
            r.aviso("modelos/versoes.json ausente (salve com chatbot.modelos.salvar_classificadores)")

        grafo = construir_grafo(config, clf_escopo, clf_tema)
        validas = set(config.nomes_intencoes) | {CHITCHAT, FORA_DOMINIO, ESCLARECER}
        amostra = (dados["teste"]["texto"].head(20).tolist() if "teste" in dados else ["olá"])
        rotas = [grafo.invoke({"mensagem": t}).get("rota") for t in amostra]
        invalidas = sorted({str(x) for x in rotas if x not in validas})
        (r.erro if invalidas else r.ok)(
            f"grafo responde e devolve 'rota' válida ({len(amostra)} mensagens testadas)"
            + (f" - rotas inválidas: {invalidas}" if invalidas else ""))
    except Exception as e:  # noqa: BLE001
        r.erro(f"modelos/grafo: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)

    # 4. base de conhecimento -------------------------------------------------
    try:
        from chatbot.conhecimento import BaseConhecimento, listar_documentos

        pasta_docs = BaseConhecimento.pasta_documentos(config)
        docs = listar_documentos(pasta_docs)
        (r.ok if len(docs) >= DOCUMENTOS_MIN else r.erro)(
            f"{len(docs)} documento(s) na base de conhecimento (mínimo {DOCUMENTOS_MIN})")
        if docs:
            base = BaseConhecimento.construir(config, verbose=False)
            etiquetas = {t.intencao for t in base.trechos}
            sem = [i for i in config.nomes_intencoes if i not in etiquetas]
            (r.aviso if sem else r.ok)(
                "todas as intenções têm documentos na própria subpasta"
                + (f" - sem documento próprio: {sem}" if sem else ""))
        fontes = pasta_docs / "FONTES.md"
        fontes_ok = fontes.exists() and "TODO" not in fontes.read_text(encoding="utf-8")
        (r.ok if fontes_ok else r.erro)("documentos/FONTES.md preenchido com a origem de cada documento")
    except Exception as e:  # noqa: BLE001
        r.erro(f"base de conhecimento: {type(e).__name__}: {e}")

    # 5. segurança da chave -----------------------------------------------------
    padrao_chave = re.compile(r"sk-or-[A-Za-z0-9_-]{16,}")
    vazados = []
    for arq in RAIZ.rglob("*"):
        partes = set(arq.relative_to(RAIZ).parts)
        if (not arq.is_file() or arq.name == ".env" or partes & {".git", ".venv", "venv", ".venv-correcao"}
                or arq.suffix in {".joblib", ".pdf", ".png", ".jpg", ".zip"} or arq.stat().st_size > 5_000_000):
            continue
        try:
            if padrao_chave.search(arq.read_text(encoding="utf-8", errors="ignore")):
                vazados.append(str(arq.relative_to(RAIZ)))
        except OSError:
            pass
    (r.erro if vazados else r.ok)(
        "nenhuma chave do OpenRouter escrita em arquivos do repositório"
        + (f" - REMOVA a chave de: {vazados} (e peça uma nova ao professor)" if vazados else ""))

    # 6. entregáveis ----------------------------------------------------------
    nb = RAIZ / "notebooks" / "01_trabalho_grupo.ipynb"
    (r.ok if nb.exists() else r.erro)("notebook notebooks/01_trabalho_grupo.ipynb presente")
    pdf = RAIZ / "relatorio" / "relatorio.pdf"
    (r.ok if pdf.exists() else r.aviso)("relatorio/relatorio.pdf presente")
    links = RAIZ / "LINKS.txt"
    texto_links = links.read_text(encoding="utf-8") if links.exists() else ""
    tem_video = re.search(r"^\s*video\s*:\s*https?://\S+", texto_links, re.I | re.M)
    (r.ok if tem_video else r.aviso)("LINKS.txt com o link do vídeo (linha 'video: https://...')")
    uso_ia = RAIZ / "USO_DE_IA.md"
    texto_ia = uso_ia.read_text(encoding="utf-8") if uso_ia.exists() else "TODO"
    (r.ok if "TODO" not in texto_ia else r.aviso)(
        "USO_DE_IA.md preenchido (base da seção obrigatória de uso de IA do artigo e do vídeo)")

    return 1 if r.imprimir() else 0


if __name__ == "__main__":
    sys.exit(main())
