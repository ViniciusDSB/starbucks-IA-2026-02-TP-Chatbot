"""Pré-processamento de texto.

As funções daqui ficam em um módulo do pacote (e não dentro do notebook) por um
motivo prático: o ``joblib`` só consegue salvar e recarregar um ``Pipeline`` que
usa uma função se essa função puder ser *importada* depois. Função definida no
notebook não pode. Se o grupo criar o próprio pré-processamento, coloque-o aqui
(ou em outro módulo dentro de ``chatbot/``).
"""
from __future__ import annotations

import re
import unicodedata

_STOPWORDS_BRUTAS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para",
    "pra", "pro", "e", "ou", "que", "se", "me", "te", "lhe", "eu", "tu", "ele", "ela",
    "nós", "vos", "eles", "elas", "meu", "minha", "meus", "minhas", "seu", "sua",
    "seus", "suas", "ao", "aos", "à", "às", "é", "ser", "ter", "tem", "tô", "estou",
    "esta", "está", "isso", "isto", "esse", "essa", "este", "aquele", "aquela", "já",
    "também", "mais", "muito", "bem", "aqui", "lá", "com", "sem",
}


def remover_acentos(texto: str) -> str:
    """'Renovação' -> 'Renovacao'. Usuários de chat frequentemente omitem acentos."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos, sem pontuação e com espaços simples.

    Use como ``preprocessor`` do ``TfidfVectorizer``:

    >>> normalizar("Posso RENOVAR o livro??  Obrigado!")
    'posso renovar o livro obrigado'
    """
    texto = remover_acentos(str(texto).lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return " ".join(texto.split())


#: Lista curta de palavras vazias (*stopwords*) do português, já sem acentos (para
#: combinar com ``normalizar``). Uso: ``TfidfVectorizer(stop_words=STOPWORDS, ...)``.
#: Atenção: em classificação de intenções, remover palavras como "não", "como",
#: "quando" e "onde" pode APAGAR justamente a pista da intenção. Por isso elas
#: ficaram de fora desta lista. Teste com e sem remoção e discuta no relatório.
STOPWORDS = sorted({remover_acentos(p) for p in _STOPWORDS_BRUTAS})
