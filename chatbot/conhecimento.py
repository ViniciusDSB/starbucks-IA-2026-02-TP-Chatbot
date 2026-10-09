"""Base de conhecimento: documentos do grupo -> trechos -> vetores -> busca.

É a parte "R" de um sistema RAG (*Retrieval-Augmented Generation*, geração
aumentada por recuperação). Em vez de o LLM responder "de cabeça" (e inventar), o
bot primeiro **recupera** os trechos dos documentos oficiais mais parecidos com a
pergunta e só então pede ao LLM que responda **com base nesses trechos**.

Como funciona
-------------

1. **Documentos.** O grupo coloca os arquivos oficiais do tema na pasta
   ``documentos/`` (``.pdf``, ``.md``, ``.txt``, ``.html``), organizados assim::

       documentos/
         geral/              <- vale para qualquer intenção (regimento, FAQ geral...)
         renovacao/          <- uma subpasta com o NOME de cada intenção (opcional)
         multa/
         ...

   Arquivos dentro da pasta de uma intenção recebem essa intenção como etiqueta;
   os demais ficam como ``geral``.
2. **Trechos (*chunks*).** Cada documento é quebrado em pedaços de ~700
   caracteres, respeitando parágrafos. Documento inteiro é grande demais para o
   prompt; frase solta perde o contexto.
3. **Vetores.** Cada trecho vira um vetor (TF-IDF, que vocês já conhecem, ou
   *embeddings* de um modelo pré-treinado - ver ``representacao`` no grupo.yaml).
4. **Índice local** (o "banco vetorial"). Vetores, trechos e etiquetas são
   salvos em ``.vectordb/indice.joblib``. **Na primeira vez** que o bot precisa da
   base, o índice é criado; depois é só carregado. Se algum documento mudar, a
   assinatura muda e o índice é recriado automaticamente.
5. **Busca.** A pergunta vira vetor do mesmo jeito e calculamos a **similaridade
   do cosseno** com todos os trechos. Os ``k`` mais parecidos vão para o prompt.
   Se o classificador de tema previu uma intenção, a busca fica restrita aos
   trechos daquela intenção + ``geral`` - o classificador ajuda a busca!

Em produção usa-se um banco vetorial dedicado (Chroma, FAISS, pgvector...). Aqui o
índice é um arquivo com uma matriz - pequeno, transparente e sem instalar nada.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import joblib
import numpy as np

from .config import Config

EXTENSOES = {".md", ".txt", ".pdf", ".html", ".htm"}
#: arquivos de controle da pasta, que NÃO entram na base
IGNORADOS = {"fontes.md", "leiame.md", "readme.md"}
GERAL = "geral"
VERSAO_INDICE = 1


@dataclass
class Trecho:
    texto: str
    fonte: str        # caminho do arquivo, relativo à pasta de documentos
    intencao: str     # nome da intenção (pela subpasta) ou "geral"
    pagina: int | None = None


# ---------------------------------------------------------------------------
# 1. Leitura dos documentos
# ---------------------------------------------------------------------------

class _ExtratorHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.partes, self._ignorar = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self._ignorar += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.partes.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer") and self._ignorar:
            self._ignorar -= 1

    def handle_data(self, data):
        if not self._ignorar:
            self.partes.append(data)


def ler_documento(caminho: Path) -> list[tuple[str, int | None]]:
    """Texto do arquivo como lista de (texto, página). Página só existe em PDF."""
    ext = caminho.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise ImportError("Ler PDF exige o pacote pypdf: pip install -r requirements.txt") from e
        leitor = PdfReader(str(caminho))
        return [(p.extract_text() or "", i + 1) for i, p in enumerate(leitor.pages)]
    texto = caminho.read_text(encoding="utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        ext_html = _ExtratorHTML()
        ext_html.feed(texto)
        texto = "".join(ext_html.partes)
    return [(texto, None)]


def listar_documentos(pasta: Path) -> list[Path]:
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSOES and not p.name.startswith(".")
                  and p.name.lower() not in IGNORADOS)


# ---------------------------------------------------------------------------
# 2. Quebra em trechos
# ---------------------------------------------------------------------------

def quebrar_em_trechos(texto: str, tamanho: int = 700) -> list[str]:
    """Junta parágrafos até ~``tamanho`` caracteres. Títulos Markdown (#) são
    repetidos no início dos trechos seguintes, para não perder o contexto."""
    texto = texto.replace("\r", "")
    paragrafos = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", texto)]
    paragrafos = [p for p in paragrafos if p]
    trechos, atual, titulo = [], "", ""
    for p in paragrafos:
        if p.startswith("#"):
            if atual and atual != titulo:   # não guarda trecho que é só um título
                trechos.append(atual)
            titulo = p.lstrip("#").strip()
            atual = titulo
            continue
        if atual and len(atual) + len(p) + 1 > tamanho:
            trechos.append(atual)
            atual = f"{titulo}\n{p}" if titulo else p
        else:
            atual = f"{atual}\n{p}" if atual else p
        while len(atual) > 1.6 * tamanho:  # parágrafo gigante (ex.: PDF sem quebras)
            corte = atual.rfind(" ", 0, tamanho) if " " in atual[:tamanho] else tamanho
            trechos.append(atual[:corte])
            atual = atual[corte:].strip()
    if atual and atual != titulo:
        trechos.append(atual)
    return trechos


# ---------------------------------------------------------------------------
# 3-5. Índice vetorial
# ---------------------------------------------------------------------------

def _criar_vetorizador(cfg: dict):
    if cfg["representacao"] == "embeddings":
        from .representacoes import EmbeddingsSentenca

        return EmbeddingsSentenca(modelo=cfg["modelo_embeddings"])
    from sklearn.feature_extraction.text import TfidfVectorizer

    from .texto import normalizar

    # n-gramas de caracteres: tolera erros de digitação e variações (renovo/renovar)
    return TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb", ngram_range=(3, 5),
                           sublinear_tf=True)


def _normalizar_linhas(m):
    """Divide cada linha pelo seu comprimento: o produto escalar vira cosseno."""
    from scipy import sparse

    if sparse.issparse(m):
        normas = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
        normas[normas == 0] = 1.0
        return sparse.diags(1.0 / normas) @ m
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return m / normas


class BaseConhecimento:
    """Índice vetorial dos documentos do grupo.

    Uso típico::

        base = BaseConhecimento.carregar_ou_criar(config)   # cria na 1ª vez
        for t, sim in base.buscar("posso renovar pelo celular?", intencao="renovacao"):
            print(f"{sim:.2f}  {t.fonte}: {t.texto[:80]}")
    """

    def __init__(self, trechos: list[Trecho], vetorizador, matriz, assinatura: str, cfg: dict):
        self.trechos = trechos
        self.vetorizador = vetorizador
        self.matriz = matriz
        self.assinatura = assinatura
        self.cfg = cfg

    # --- construção --------------------------------------------------------
    @staticmethod
    def pasta_documentos(config: Config) -> Path:
        return (config.pasta / config.conhecimento["pasta"]).resolve()

    @staticmethod
    def arquivo_indice(config: Config) -> Path:
        return config.pasta / ".vectordb" / "indice.joblib"

    @classmethod
    def assinatura_atual(cls, config: Config) -> str:
        """Muda quando qualquer documento (ou a configuração da base) muda."""
        pasta = cls.pasta_documentos(config)
        h = hashlib.sha256(json.dumps(config.conhecimento, sort_keys=True).encode())
        h.update(str(VERSAO_INDICE).encode())
        h.update(Path(__file__).read_bytes())   # mudou o código da base? recria também
        for arq in listar_documentos(pasta):
            h.update(str(arq.relative_to(pasta)).encode())
            h.update(hashlib.sha256(arq.read_bytes()).digest())
        h.update(",".join(config.nomes_intencoes).encode())
        return h.hexdigest()[:16]

    @classmethod
    def construir(cls, config: Config, verbose: bool = True) -> "BaseConhecimento":
        cfg = config.conhecimento
        pasta = cls.pasta_documentos(config)
        arquivos = listar_documentos(pasta)
        if not arquivos:
            raise FileNotFoundError(
                f"Nenhum documento em {pasta}. Coloque arquivos .pdf/.md/.txt/.html lá "
                "(ver documentos/LEIAME.md).")
        intencoes = set(config.nomes_intencoes)
        trechos = []
        for arq in arquivos:
            rel = arq.relative_to(pasta)
            pasta_pai = rel.parts[0] if len(rel.parts) > 1 else GERAL
            intencao = pasta_pai if pasta_pai in intencoes else GERAL
            for texto, pagina in ler_documento(arq):
                for t in quebrar_em_trechos(texto, cfg["tamanho_trecho"]):
                    if len(t) >= 30:
                        trechos.append(Trecho(t, str(rel), intencao, pagina))
        if not trechos:
            raise ValueError(f"Os documentos de {pasta} não têm texto extraível.")
        vetorizador = _criar_vetorizador(cfg)
        matriz = _normalizar_linhas(vetorizador.fit_transform([t.texto for t in trechos]))
        base = cls(trechos, vetorizador, matriz, cls.assinatura_atual(config), cfg)
        if verbose:
            print(f"[base de conhecimento] {len(arquivos)} documento(s) -> {len(trechos)} trechos "
                  f"({cfg['representacao']}); índice salvo em {cls.arquivo_indice(config)}")
        return base

    def salvar(self, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"versao": VERSAO_INDICE, "trechos": self.trechos, "vetorizador": self.vetorizador,
                     "matriz": self.matriz, "assinatura": self.assinatura, "cfg": self.cfg}, caminho)

    @classmethod
    def carregar_ou_criar(cls, config: Config, forcar: bool = False, verbose: bool = True) -> "BaseConhecimento":
        """Carrega o índice salvo; cria (ou recria) se não existir ou se os documentos mudaram."""
        arquivo = cls.arquivo_indice(config)
        assinatura = cls.assinatura_atual(config)
        if arquivo.exists() and not forcar:
            try:
                dados = joblib.load(arquivo)
                if dados.get("assinatura") == assinatura and dados.get("versao") == VERSAO_INDICE:
                    return cls(dados["trechos"], dados["vetorizador"], dados["matriz"],
                               dados["assinatura"], dados["cfg"])
                if verbose:
                    print("[base de conhecimento] documentos (ou configuração) mudaram - recriando o índice...")
            except Exception:  # noqa: BLE001 - índice corrompido/versão diferente: recria
                pass
        base = cls.construir(config, verbose=verbose)
        base.salvar(arquivo)
        return base

    # --- consulta -----------------------------------------------------------
    def buscar(self, pergunta: str, intencao: str | None = None, k: int | None = None,
               minimo: float | None = None) -> list[tuple[Trecho, float]]:
        """Os ``k`` trechos mais parecidos com a pergunta, com a similaridade (0 a 1).

        Com ``intencao``, só entram trechos daquela intenção ou de ``geral``; se o
        filtro não deixar nenhum trecho, a busca é refeita sem ele.
        """
        k = k or self.cfg["k"]
        minimo = self.cfg["similaridade_minima"] if minimo is None else minimo
        q = _normalizar_linhas(self.vetorizador.transform([pergunta]))
        produto = self.matriz @ q.T            # vetores normalizados: produto escalar = cosseno
        sims = produto.toarray().ravel() if hasattr(produto, "toarray") else np.asarray(produto).ravel()
        candidatos = np.arange(len(self.trechos))
        if intencao:
            filtrados = [i for i in candidatos if self.trechos[i].intencao in (intencao, GERAL)]
            if filtrados:
                candidatos = np.array(filtrados)
        ordem = candidatos[np.argsort(sims[candidatos])[::-1]]
        return [(self.trechos[i], float(sims[i])) for i in ordem[:k] if sims[i] >= minimo]

    def resumo(self):
        """Tabela (DataFrame) com quantos trechos há por documento e intenção."""
        import pandas as pd

        df = pd.DataFrame([{"fonte": t.fonte, "intencao": t.intencao, "caracteres": len(t.texto)}
                           for t in self.trechos])
        return df.groupby(["intencao", "fonte"]).agg(trechos=("caracteres", "size"),
                                                     caracteres=("caracteres", "sum"))
