"""Salvar e carregar os dois classificadores do chatbot.

**Contrato** (é isto que o script de avaliação espera de cada classificador):

* ``predict(lista_de_textos) -> lista/array de rótulos (str)``;
* ``classes_`` - os rótulos que o classificador conhece;
* ``predict_proba(lista_de_textos) -> matriz (n_textos × n_classes)`` - opcional,
  mas necessário para usar o limiar de confiança e a pergunta de esclarecimento.

Qualquer ``Pipeline`` do scikit-learn cumpre o contrato. O classificador de escopo
deve conhecer exatamente as classes ``dominio``, ``chitchat`` e ``fora_dominio``; o
de tema, exatamente as intenções do ``grupo.yaml``.

Por padrão os modelos são arquivos ``.joblib``. Se o grupo precisar de outra forma
de carregar (por exemplo, um modelo que não é do scikit-learn), aponte em
``grupo.yaml`` -> ``modelos.carregador`` uma função ``"modulo:funcao"`` que recebe a
``Config`` e devolve a tupla ``(clf_escopo, clf_tema)``.
"""
from __future__ import annotations

import importlib
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import sklearn

from .config import Config
from .rotulos import CLASSES_ESCOPO


def salvar_modelo(modelo, caminho: str | Path) -> Path:
    """Salva com joblib e registra as versões das bibliotecas ao lado (``versoes.json``)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho)
    versoes = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "salvo_em": datetime.now().isoformat(timespec="seconds"),
    }
    with open(caminho.parent / "versoes.json", "w", encoding="utf-8") as f:
        json.dump(versoes, f, indent=2, ensure_ascii=False)
    return caminho


def salvar_classificadores(clf_escopo, clf_tema, config: Config) -> None:
    """Confere o contrato e salva os dois modelos nos caminhos do ``grupo.yaml``."""
    checar_interface(clf_escopo, CLASSES_ESCOPO, "escopo")
    checar_interface(clf_tema, config.nomes_intencoes, "tema")
    salvar_modelo(clf_escopo, config.modelo_escopo)
    salvar_modelo(clf_tema, config.modelo_tema)


def carregar_classificadores(config: Config):
    """Devolve ``(clf_escopo, clf_tema)`` prontos para uso no grafo."""
    if config.carregador:
        modulo, _, funcao = config.carregador.partition(":")
        fn = getattr(importlib.import_module(modulo), funcao)
        return fn(config)
    for p in (config.modelo_escopo, config.modelo_tema):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {p}. Treine e salve os classificadores antes "
                "(veja a seção 'Salvar os modelos' do notebook)."
            )
    return joblib.load(config.modelo_escopo), joblib.load(config.modelo_tema)


def checar_interface(clf, classes_esperadas, nome: str) -> None:
    """Lança ``TypeError``/``ValueError`` se ``clf`` não cumprir o contrato."""
    if not hasattr(clf, "predict"):
        raise TypeError(f"Classificador de {nome} não tem o método predict")
    if not hasattr(clf, "classes_"):
        raise TypeError(f"Classificador de {nome} não tem o atributo classes_ (foi treinado?)")
    obtidas, esperadas = set(map(str, clf.classes_)), set(classes_esperadas)
    if obtidas != esperadas:
        raise ValueError(
            f"Classificador de {nome}: classes {sorted(obtidas)} diferentes das esperadas "
            f"{sorted(esperadas)}"
        )
    saida = list(clf.predict(["teste de sanidade"]))
    if len(saida) != 1 or str(saida[0]) not in esperadas:
        raise ValueError(f"Classificador de {nome}: predict devolveu {saida!r}")
