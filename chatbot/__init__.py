"""Código base do Trabalho Prático de IA (FACOM/UFMS, 2026/2): chatbot com
classificação de intenções em dois estágios, orquestrado com LangGraph, com
respostas redigidas por um LLM (OpenRouter) a partir de uma base de documentos (RAG).

Módulos:

* ``rotulos``        - rótulos reservados (dominio, chitchat, fora_dominio, esclarecer);
* ``config``         - leitura do ``grupo.yaml``;
* ``dados``          - leitura e validação de ``treino.csv``/``teste.csv``;
* ``texto``          - pré-processamento (normalização, stopwords);
* ``representacoes`` - TF-IDF de palavras/caracteres e embeddings (opcional);
* ``modelos``        - salvar/carregar os classificadores (o "contrato");
* ``conhecimento``   - documentos -> trechos -> índice vetorial -> busca (RAG);
* ``llm``            - cliente do OpenRouter, prompts do bot e ClassificadorLLM (comparação);
* ``geracao``        - geração de frases de treino com LLM;
* ``grafo``          - o chatbot como grafo de estados (LangGraph);
* ``avaliacao``      - as métricas padronizadas usadas na correção.
"""
from .config import Config, carregar_config  # noqa: F401
from .rotulos import CHITCHAT, CLASSES_ESCOPO, DOMINIO, ESCLARECER, FORA_DOMINIO  # noqa: F401

__version__ = "2026.2.0"
