"""Formas de transformar texto em vetores numéricos.

Classificadores (regressão logística, redes neurais, árvores...) recebem vetores,
não frases. Este módulo oferece três representações prontas:

* ``tfidf_palavras``   - saco de palavras (e pares de palavras) ponderado por TF-IDF;
* ``tfidf_caracteres`` - pedaços de 2 a 5 letras; robusto a erros de digitação
                          ("renovaçao", "renvar") e a variações ("renovo", "renovar");
* ``EmbeddingsSentenca`` - vetores densos de um modelo pré-treinado
                          (opcional, exige ``pip install -r requirements-extra.txt``).

Todas seguem a API do scikit-learn (``fit``/``transform``) e podem ser o primeiro
passo de um ``Pipeline``.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .texto import normalizar


def tfidf_palavras(ngram_range=(1, 2), min_df=1, sublinear_tf=True, **kwargs) -> TfidfVectorizer:
    """TF-IDF sobre palavras (unigramas e bigramas por padrão), com ``normalizar``."""
    return TfidfVectorizer(
        preprocessor=normalizar,
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        **kwargs,
    )


def tfidf_caracteres(ngram_range=(2, 5), min_df=1, sublinear_tf=True, **kwargs) -> TfidfVectorizer:
    """TF-IDF sobre n-gramas de caracteres *dentro das palavras* (``char_wb``)."""
    return TfidfVectorizer(
        preprocessor=normalizar,
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        **kwargs,
    )


class EmbeddingsSentenca(BaseEstimator, TransformerMixin):
    """Transforma frases em *embeddings* com um modelo da biblioteca sentence-transformers.

    O modelo pré-treinado NÃO é salvo dentro do ``.joblib`` (ele tem centenas de MB):
    só o seu nome. Ao carregar o pipeline, o modelo é baixado/aberto de novo na
    primeira chamada a ``transform``.

    Modelos multilíngues que funcionam bem em português:
    ``paraphrase-multilingual-MiniLM-L12-v2`` (leve) e
    ``paraphrase-multilingual-mpnet-base-v2`` (melhor, mais lento).
    """

    def __init__(self, modelo: str = "paraphrase-multilingual-MiniLM-L12-v2", normalizar_vetores: bool = True):
        self.modelo = modelo
        self.normalizar_vetores = normalizar_vetores

    def _carregar(self):
        if getattr(self, "_st", None) is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - depende do ambiente
                raise ImportError(
                    "EmbeddingsSentenca precisa do pacote sentence-transformers: "
                    "pip install -r requirements-extra.txt"
                ) from e
            self._st = SentenceTransformer(self.modelo)
        return self._st

    def fit(self, X, y=None):
        self._carregar()
        return self

    def transform(self, X):
        vetores = self._carregar().encode(
            list(X), normalize_embeddings=self.normalizar_vetores, show_progress_bar=False
        )
        return np.asarray(vetores)

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado.pop("_st", None)  # não salva o modelo pesado no joblib
        return estado
