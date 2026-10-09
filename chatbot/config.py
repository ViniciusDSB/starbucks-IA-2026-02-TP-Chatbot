"""Leitura e validação do arquivo de configuração do grupo (``grupo.yaml``).

Todos os caminhos do YAML são relativos à pasta onde o próprio YAML está. Assim o
mesmo código serve para o exemplo (``exemplos/biblioteca/grupo.yaml``) e para o
trabalho do grupo (``grupo.yaml``, na raiz do repositório).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .rotulos import RESERVADOS

RAIZ_REPO = Path(__file__).resolve().parent.parent
CONFIG_PADRAO = RAIZ_REPO / "grupo.yaml"

_NOME_VALIDO = re.compile(r"^[a-z][a-z0-9_]*$")

RESPOSTAS_PADRAO = {
    "chitchat": {
        "saudacao": "Olá! Em que posso ajudar?",
        "agradecimento": "Por nada! Se precisar, é só chamar.",
        "despedida": "Até logo! Bons estudos.",
        "padrao": "Sou um assistente automático e posso ajudar com: {assuntos}.",
    },
    "fora_dominio": (
        "Desculpe, não sei responder sobre isso. "
        "Posso ajudar com: {assuntos}."
    ),
    "esclarecer": "Não tenho certeza se entendi. Sua dúvida é sobre {opcao1} ou sobre {opcao2}?",
}


#: Modelo usado quando o grupo.yaml não define ``llm.modelo``. A lista completa de
#: modelos (e preços) está em https://openrouter.ai/models - o professor pode
#: indicar outro modelo; basta trocar no grupo.yaml.
MODELO_PADRAO = "openai/gpt-6-luna"

LLM_PADRAO = {
    "modelo": MODELO_PADRAO,
    "temperatura": 0.3,        # 0 = sempre a mesma resposta; 1 = mais criativo/variado
    "max_tokens": 300,         # limite de tamanho da resposta (1 token ~ 3/4 de palavra)
    "usar_em": ["intencoes", "chitchat"],   # em quais nós o LLM redige a resposta
    "persona": (
        "Você é um assistente virtual de atendimento a estudantes universitários. "
        "Seja cordial, direto e use linguagem simples."
    ),
    "contato": "procurar o atendimento presencial do setor responsável",
    "max_frases": 3,
    "turnos_historico": 3,     # quantas trocas anteriores o LLM enxerga
}


CONHECIMENTO_PADRAO = {
    "pasta": "documentos",          # onde ficam os documentos oficiais (relativo ao grupo.yaml)
    "representacao": "tfidf",       # "tfidf" (padrão, sem downloads) ou "embeddings"
    "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
    "tamanho_trecho": 700,          # caracteres por trecho (chunk)
    "k": 4,                         # quantos trechos vão para o prompt
    "similaridade_minima": 0.05,    # trechos menos parecidos que isso são descartados
}


class ErroConfig(ValueError):
    """Erro de preenchimento do ``grupo.yaml``."""


@dataclass
class Intencao:
    nome: str
    descricao: str
    resposta: str              # resposta fixa (usada sem LLM e na avaliação automática)


@dataclass
class Config:
    arquivo: Path
    grupo: str
    integrantes: list[dict]
    tema: str
    descricao_tema: str
    intencoes: dict[str, Intencao]
    treino: Path
    teste: Path
    modelo_escopo: Path
    modelo_tema: Path
    carregador: str | None = None
    limiar_tema: float = 0.0
    respostas: dict = field(default_factory=dict)
    llm: dict = field(default_factory=lambda: dict(LLM_PADRAO))
    conhecimento: dict = field(default_factory=lambda: dict(CONHECIMENTO_PADRAO))

    @property
    def pasta(self) -> Path:
        return self.arquivo.parent

    @property
    def nomes_intencoes(self) -> list[str]:
        return list(self.intencoes)

    def assuntos(self) -> str:
        """Lista legível das intenções, usada nas respostas de recusa."""
        descr = [i.descricao for i in self.intencoes.values()]
        if len(descr) <= 1:
            return "".join(descr)
        return ", ".join(descr[:-1]) + " e " + descr[-1]


def _exigir(dic: dict, chave: str, onde: str):
    if not isinstance(dic, dict) or chave not in dic or dic[chave] in (None, ""):
        raise ErroConfig(f"Campo obrigatório ausente em {onde}: '{chave}'")
    return dic[chave]


def carregar_config(caminho: str | Path | None = None) -> Config:
    """Lê e valida o ``grupo.yaml``. Lança :class:`ErroConfig` com mensagem clara."""
    caminho = Path(caminho) if caminho else CONFIG_PADRAO
    if not caminho.is_absolute():
        # tenta relativo ao diretório atual e, depois, à raiz do repositório
        caminho = caminho if caminho.exists() else RAIZ_REPO / caminho
    if not caminho.exists():
        raise ErroConfig(f"Arquivo de configuração não encontrado: {caminho}")
    caminho = caminho.resolve()
    with open(caminho, encoding="utf-8") as f:
        bruto = yaml.safe_load(f) or {}

    pasta = caminho.parent

    def resolver(p: str) -> Path:
        return (pasta / p).resolve()

    grupo = _exigir(bruto, "grupo", "raiz")
    tema = _exigir(bruto, "tema", "raiz")
    dados = _exigir(bruto, "dados", "raiz")
    modelos = _exigir(bruto, "modelos", "raiz")
    brutas = _exigir(bruto, "intencoes", "raiz")

    if not isinstance(brutas, dict):
        raise ErroConfig("'intencoes' deve ser um mapeamento nome -> {descricao, resposta}")

    intencoes = {}
    for nome, info in brutas.items():
        nome = str(nome)
        if not _NOME_VALIDO.match(nome):
            raise ErroConfig(
                f"Nome de intenção inválido: '{nome}'. Use só letras minúsculas sem acento, "
                "dígitos e '_', começando por letra (ex.: 'renovacao_emprestimo')."
            )
        if nome in RESERVADOS:
            raise ErroConfig(f"'{nome}' é um rótulo reservado e não pode ser intenção do domínio.")
        intencoes[nome] = Intencao(
            nome=nome,
            descricao=str(_exigir(info, "descricao", f"intencoes.{nome}")).strip(),
            resposta=str(_exigir(info, "resposta", f"intencoes.{nome}")).strip(),
        )

    respostas = {k: (dict(v) if isinstance(v, dict) else v) for k, v in RESPOSTAS_PADRAO.items()}
    for chave, valor in (bruto.get("respostas") or {}).items():
        if isinstance(valor, dict) and isinstance(respostas.get(chave), dict):
            respostas[chave].update(valor)
        else:
            respostas[chave] = valor

    roteamento = bruto.get("roteamento") or {}
    limiar = float(roteamento.get("limiar_tema", 0.0))
    if not 0.0 <= limiar <= 1.0:
        raise ErroConfig("roteamento.limiar_tema deve estar entre 0 e 1")

    llm = dict(LLM_PADRAO)
    llm.update({k: v for k, v in (bruto.get("llm") or {}).items() if v is not None})
    llm["usar_em"] = list(llm.get("usar_em") or [])
    invalidos = set(llm["usar_em"]) - {"intencoes", "chitchat", "fora_dominio"}
    if invalidos:
        raise ErroConfig(f"llm.usar_em aceita intencoes, chitchat e fora_dominio; recebi {sorted(invalidos)}")

    conhecimento = dict(CONHECIMENTO_PADRAO)
    conhecimento.update({k: v for k, v in (bruto.get("conhecimento") or {}).items() if v is not None})
    if conhecimento["representacao"] not in ("tfidf", "embeddings"):
        raise ErroConfig("conhecimento.representacao deve ser 'tfidf' ou 'embeddings'")

    return Config(
        arquivo=caminho,
        grupo=str(_exigir(grupo, "nome", "grupo")),
        integrantes=list(grupo.get("integrantes") or []),
        tema=str(_exigir(tema, "nome", "tema")),
        descricao_tema=str(tema.get("descricao", "")).strip(),
        intencoes=intencoes,
        treino=resolver(_exigir(dados, "treino", "dados")),
        teste=resolver(_exigir(dados, "teste", "dados")),
        modelo_escopo=resolver(_exigir(modelos, "escopo", "modelos")),
        modelo_tema=resolver(_exigir(modelos, "tema", "modelos")),
        carregador=modelos.get("carregador"),
        limiar_tema=limiar,
        respostas=respostas,
        llm=llm,
        conhecimento=conhecimento,
    )
