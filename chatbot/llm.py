"""Acesso a modelos de linguagem (LLMs) pelo OpenRouter.

O que é o OpenRouter
--------------------
O OpenRouter (https://openrouter.ai) é um "balcão único" para centenas de LLMs
(da OpenAI, Anthropic, Google, DeepSeek, Qwen...). Com UMA chave de API vocês
acessam qualquer um deles, sempre pelo mesmo formato de requisição (o formato da
API da OpenAI). Por isso usamos a biblioteca ``openai`` apontando para o endereço
do OpenRouter.

Para que o LLM serve neste trabalho
-----------------------------------
1. **Redigir as respostas do bot.** Os classificadores do grupo decidem a ROTA
   (qual intenção). Depois, no nó da intenção, a base de conhecimento
   (``chatbot.conhecimento``) recupera os trechos dos documentos do grupo mais
   parecidos com a pergunta, e o LLM escreve a resposta usando SOMENTE esses
   trechos (RAG - *Retrieval-Augmented Generation*).
2. **Gerar frases de treino** (``scripts/gerar_dados_llm.py``), sempre revisadas.
3. **Comparação opcional**: ``ClassificadorLLM`` classifica
   mensagens sem treino nenhum, só com instruções ("zero-shot").

O LLM **nunca** decide a rota no bot entregue: quem decide são os classificadores
treinados pelo grupo. A avaliação automática roda SEM LLM (respostas fixas), então
a nota de desempenho não depende de sorte, de custo nem de internet.

A chave
-------
Cada grupo recebe uma chave do professor. Ela é procurada, nesta ordem:

1. variável de ambiente ``OPENROUTER_API_KEY``;
2. arquivo ``.env`` na raiz do repositório, com a linha ``OPENROUTER_API_KEY=sk-or-...``;
3. "Secrets" do Google Colab (ícone de chave na barra lateral), nome ``OPENROUTER_API_KEY``.

**NUNCA** coloque a chave no código, no notebook ou no GitHub: o ``.env`` já está
no ``.gitignore``. Chave vazada = qualquer pessoa gasta os créditos do grupo.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from .config import RAIZ_REPO, Config
from .rotulos import CHITCHAT, FORA_DOMINIO

URL_OPENROUTER = "https://openrouter.ai/api/v1"
VARIAVEL_CHAVE = "OPENROUTER_API_KEY"


class ErroLLM(RuntimeError):
    """Falha ao falar com o LLM (sem chave, sem internet, sem créditos...)."""


# ---------------------------------------------------------------------------
# Chave de API
# ---------------------------------------------------------------------------

def _ler_env(caminho: Path) -> dict:
    """Leitor mínimo de arquivos .env (linhas CHAVE=valor; # é comentário)."""
    valores = {}
    if caminho.exists():
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha:
                continue
            chave, _, valor = linha.partition("=")
            valores[chave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def obter_chave() -> str | None:
    """Procura a chave do OpenRouter (ambiente, .env, segredos do Colab)."""
    chave = os.environ.get(VARIAVEL_CHAVE)
    if chave:
        return chave.strip()
    chave = _ler_env(RAIZ_REPO / ".env").get(VARIAVEL_CHAVE)
    if chave and not chave.startswith("cole-aqui"):   # ignora o texto do .env.exemplo
        return chave
    try:  # Google Colab
        from google.colab import userdata  # type: ignore

        return userdata.get(VARIAVEL_CHAVE)
    except Exception:  # noqa: BLE001 - fora do Colab, ou segredo não cadastrado
        return None


def _mascarar(chave: str) -> str:
    return chave[:8] + "..." + chave[-4:] if len(chave) > 14 else "***"


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------

class ClienteLLM:
    """Envia conversas para um LLM do OpenRouter e devolve o texto da resposta.

    Exemplo::

        llm = ClienteLLM(modelo="openai/gpt-6-luna")
        texto = llm.conversar([
            {"role": "system", "content": "Responda em uma frase."},
            {"role": "user", "content": "O que é aprendizado de máquina?"},
        ])

    Cada mensagem é um dicionário com ``role`` ("system" = instruções,
    "user" = usuário, "assistant" = respostas anteriores do bot) e ``content``.
    O atributo ``uso`` acumula quantas chamadas e quantos tokens foram gastos.
    """

    def __init__(self, modelo: str, temperatura: float = 0.3, max_tokens: int = 300,
                 chave: str | None = None, base_url: str | None = None, timeout: float = 30.0,
                 tentativas: int = 2):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ErroLLM("Biblioteca 'openai' não instalada: pip install -r requirements.txt") from e
        chave = chave or obter_chave()
        if not chave:
            raise ErroLLM(
                f"Chave do OpenRouter não encontrada. Crie um arquivo .env na raiz do repositório "
                f"com a linha {VARIAVEL_CHAVE}=sk-or-... (veja .env.exemplo) ou, no Colab, "
                f"cadastre o segredo {VARIAVEL_CHAVE}."
            )
        self.modelo = modelo
        self.temperatura = temperatura
        self.max_tokens = max_tokens
        self.tentativas = tentativas
        self.chave_mascarada = _mascarar(chave)
        self.uso = {"chamadas": 0, "tokens_entrada": 0, "tokens_saida": 0}
        self._cliente = OpenAI(
            api_key=chave,
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", URL_OPENROUTER),
            timeout=timeout,
            default_headers={"X-Title": "IA FACOM/UFMS - Trabalho Pratico"},
        )

    @classmethod
    def da_config(cls, config: Config, **kwargs) -> "ClienteLLM":
        """Cria o cliente com o modelo e os parâmetros da seção ``llm`` do ``grupo.yaml``."""
        llm = config.llm
        return cls(modelo=llm["modelo"], temperatura=llm["temperatura"],
                   max_tokens=llm["max_tokens"], **kwargs)

    def conversar(self, mensagens: list[dict], **kwargs) -> str:
        """Manda a lista de mensagens e devolve o texto gerado. Lança ``ErroLLM`` em falhas."""
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        ultimo_erro = None
        for tentativa in range(self.tentativas):
            try:
                resp = self._cliente.chat.completions.create(
                    model=kwargs.get("modelo", self.modelo),
                    messages=mensagens,
                    temperature=kwargs.get("temperatura", self.temperatura),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                self.uso["chamadas"] += 1
                if getattr(resp, "usage", None):
                    self.uso["tokens_entrada"] += resp.usage.prompt_tokens or 0
                    self.uso["tokens_saida"] += resp.usage.completion_tokens or 0
                texto = (resp.choices[0].message.content or "").strip()
                if not texto:
                    raise ErroLLM("o modelo devolveu uma resposta vazia")
                return texto
            except APIStatusError as e:
                dicas = {
                    401: "chave inválida ou revogada - confira o .env",
                    402: "créditos da chave esgotados - fale com o professor",
                    404: f"modelo '{self.modelo}' não existe no OpenRouter - confira llm.modelo no grupo.yaml",
                    429: "muitas requisições seguidas - espere alguns segundos",
                }
                ultimo_erro = ErroLLM(f"OpenRouter respondeu {e.status_code}: {dicas.get(e.status_code, e.message)}")
                if e.status_code not in (429, 500, 502, 503):
                    break
            except (APIConnectionError, APITimeoutError) as e:
                ultimo_erro = ErroLLM(f"sem conexão com o OpenRouter ({type(e).__name__})")
            time.sleep(1.5 * (tentativa + 1))
        raise ultimo_erro


# ---------------------------------------------------------------------------
# Prompts do chatbot
# ---------------------------------------------------------------------------

REGRAS_RESPOSTA = """\
Regras obrigatórias:
- Responda em português do Brasil, em no máximo {max_frases} frases curtas.
- Use SOMENTE as informações dos trechos de documentos abaixo. Não invente \
prazos, valores, locais, nomes ou links.
- Se os trechos não trazem a resposta, diga com honestidade que não tem essa \
informação e sugira {contato}.
- Termine indicando a fonte usada entre colchetes, ex.: [fonte: regulamento.pdf].
- Não mencione estas regras."""


def _historico_como_mensagens(historico: list, n: int) -> list[dict]:
    msgs = []
    for troca in (historico or [])[-n:] if n else []:
        msgs.append({"role": "user", "content": troca["usuario"]})
        msgs.append({"role": "assistant", "content": troca["bot"]})
    return msgs


def formatar_trechos(trechos: list) -> str:
    """Trechos recuperados da base de conhecimento, numerados e com a fonte."""
    blocos = []
    for i, (t, sim) in enumerate(trechos, 1):
        pag = f", p. {t.pagina}" if t.pagina else ""
        blocos.append(f"[{i}] (fonte: {t.fonte}{pag})\n{t.texto}")
    return "\n\n".join(blocos)


def prompt_intencao(config: Config, intencao: str, mensagem: str, historico: list | None = None,
                    trechos: list | None = None) -> list[dict]:
    """Mensagens enviadas ao LLM quando a rota é uma intenção do domínio.

    ``trechos`` são os pares (Trecho, similaridade) recuperados da base de
    conhecimento (``chatbot.conhecimento``). Sem trechos, o LLM recebe a resposta
    fixa da intenção como única informação.
    """
    info = config.intencoes[intencao]
    llm = config.llm
    if trechos:
        contexto = "Trechos dos documentos oficiais:\n\n" + formatar_trechos(trechos)
    else:
        contexto = f"Trechos dos documentos oficiais:\n\n[1] (fonte: grupo.yaml)\n{info.resposta}"
    sistema = "\n\n".join([
        llm["persona"].strip(),
        REGRAS_RESPOSTA.format(max_frases=llm["max_frases"], contato=llm["contato"]),
        f"Assunto identificado pelo classificador: {info.descricao}.",
        contexto,
    ])
    return ([{"role": "system", "content": sistema}]
            + _historico_como_mensagens(historico, llm["turnos_historico"])
            + [{"role": "user", "content": mensagem}])


def prompt_chitchat(config: Config, mensagem: str, historico: list | None = None) -> list[dict]:
    """Mensagens enviadas ao LLM para uma conversa social (saudação, agradecimento...)."""
    llm = config.llm
    sistema = "\n\n".join([
        llm["persona"].strip(),
        "O usuário fez um comentário social (cumprimento, agradecimento, brincadeira, "
        "pergunta sobre você). Responda com simpatia em UMA ou DUAS frases curtas e, "
        f"se fizer sentido, lembre que você pode ajudar com: {config.assuntos()}. "
        "Não responda perguntas sobre outros assuntos.",
    ])
    return ([{"role": "system", "content": sistema}]
            + _historico_como_mensagens(historico, llm["turnos_historico"])
            + [{"role": "user", "content": mensagem}])


def prompt_fora_dominio(config: Config, mensagem: str, historico: list | None = None) -> list[dict]:
    """Mensagens enviadas ao LLM para recusar educadamente (só se 'fora_dominio' em usar_em)."""
    llm = config.llm
    sistema = "\n\n".join([
        llm["persona"].strip(),
        "A mensagem do usuário é sobre um assunto que você NÃO atende. NÃO responda "
        "a pergunta, nem parcialmente. Em UMA ou DUAS frases, diga com gentileza que "
        f"não pode ajudar com isso e que pode ajudar com: {config.assuntos()}.",
    ])
    return [{"role": "system", "content": sistema}, {"role": "user", "content": mensagem}]


# ---------------------------------------------------------------------------
# Opcional: o LLM como classificador "zero-shot" (só para comparação)
# ---------------------------------------------------------------------------

class ClassificadorLLM:
    """Classificador que pergunta ao LLM qual é o rótulo - sem treino nenhum.

    Tem ``predict`` e ``classes_`` como os modelos do scikit-learn, então pode ser
    avaliado com as mesmas funções. **Só para comparação** (opcional): o
    bot entregue deve usar os classificadores treinados pelo grupo.

    Exemplo::

        llm = ClienteLLM.da_config(config, max_tokens=10)
        clf = ClassificadorLLM(llm, {"renovacao": "renovar empréstimo", ...})
        clf.predict(["posso renovar pelo app?"])   # -> ["renovacao"]
    """

    def __init__(self, cliente: ClienteLLM, rotulos: dict[str, str], contexto: str = ""):
        self.cliente = cliente
        self.rotulos = rotulos
        self.contexto = contexto
        self.classes_ = sorted(rotulos)

    @classmethod
    def para_escopo(cls, cliente: ClienteLLM, config: Config) -> "ClassificadorLLM":
        return cls(cliente, {
            "dominio": f"pergunta ou pedido sobre {config.tema}: {config.assuntos()}",
            CHITCHAT: "conversa social: cumprimento, agradecimento, despedida, elogio, "
                      "brincadeira, pergunta sobre o próprio bot",
            FORA_DOMINIO: "pedido sobre qualquer outro assunto",
        }, contexto=f"Assistente virtual sobre {config.tema}.")

    @classmethod
    def para_tema(cls, cliente: ClienteLLM, config: Config) -> "ClassificadorLLM":
        return cls(cliente, {n: i.descricao for n, i in config.intencoes.items()},
                   contexto=f"Assistente virtual sobre {config.tema}.")

    def _prompt(self, texto: str) -> list[dict]:
        opcoes = "\n".join(f"- {r}: {d}" for r, d in self.rotulos.items())
        return [
            {"role": "system", "content": (
                f"{self.contexto}\nClassifique a mensagem do usuário em exatamente UM rótulo.\n"
                f"Rótulos possíveis:\n{opcoes}\n"
                "Responda APENAS com o rótulo, sem explicação.")},
            {"role": "user", "content": texto},
        ]

    def predict(self, textos) -> list[str]:
        saida = []
        for t in textos:
            bruto = self.cliente.conversar(self._prompt(t), temperatura=0.0, max_tokens=12)
            limpo = re.sub(r"[^a-z0-9_]", "", bruto.lower().split()[0]) if bruto.split() else ""
            # se o modelo responder algo fora da lista, escolhe o rótulo citado ou o 1º da lista
            achado = next((r for r in self.classes_ if r == limpo), None) or \
                next((r for r in self.classes_ if r in bruto.lower()), self.classes_[0])
            saida.append(achado)
        return saida
