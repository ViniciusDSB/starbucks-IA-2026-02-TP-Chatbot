"""
Crawler recursivo de editais (IC, monitoria, etc.).

Fluxo: páginas-semente -> páginas filtradas -> ... -> PDFs.
Todos os PDFs vão para ./pdfs e o índice (pdf, de onde veio, link anterior,
cadeia completa e texto/contexto do link) vai para ./pdfs_index.json.

Dependências: pip install requests beautifulsoup4
"""

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# CONFIGURAÇÃO (é aqui que você mexe pra ajustar os filtros)
# --------------------------------------------------------------------------
OUTPUT_DIR = Path("pdfs")
JSON_PATH = Path("pdfs_index.json")
MAX_DEPTH = 5          # quantos níveis de páginas seguir a partir da semente
DELAY = 0.5            # segundos entre requisições
TIMEOUT = 30
MAX_CONTEXT = 400      # tamanho máx. (caracteres) do texto do elemento que contém o link

# Só segue links desses domínios (None = qualquer um). Casa por sufixo.
ALLOWED_DOMAINS = ("ufms.br",)

# Tudo é comparado sem acento e em minúsculas.
INCLUDE_KEYWORDS = [
    "edital", "editais", "iniciacao cientifica", "pibic", "pivic", "pibiti", "piviti",
    "pibid", "monitoria", "selecao", "selecao de bolsistas", "bolsista", "bolsa",
    "boletim oficial", "relatorio"
]
EXCLUDE_KEYWORDS = [
    "ensino medio", "pibic-em", "pibic em", "pibic jr", "pibic-jr", "iniciacao cientifica junior",
    "resultado", "homologa", "classificacao", "convocacao", "taxas de publicacao", "taxa de publicacao",
    "embrapa", "fluxograma", "opinioes", "artigo", "manual", "minuta", "cgen", "tabela", "tabela qualis",
    "slides", "acesso ao patrimonio genetico", "carta"
]

# Extensões de mídia/arquivos que nunca interessam (PDF fica de fora de propósito)
MEDIA_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".ico", ".mp4", ".mp3",
    ".avi", ".mov", ".wav", ".zip", ".rar", ".7z", ".xls", ".xlsx",
    ".ppt", ".pptx", ".odt", ".ods", ".css", ".js", ".xml", ".rss",
}

SEED_LINKS = [
    *[f"https://propp.ufms.br/category/editais-publicados/editais-iniciacao-cientifica/page/{i}/"
      for i in range(1, 6)],
    *[f"https://prograd.ufms.br/category/editais/editais-monitoria/page/{i}/"
      for i in range(1, 6)],
    "https://prograd.ufms.br/edital-no-21-2026-prograd-ufms-cadastro-de-propostas-de-plano-de-trabalho-do-programa-institucional-de-monitoria-voluntaria-na-ufms-em-2026/",
    "https://boletimoficial.ufms.br/bse/publicacao?id=533961",
    "https://www.ufms.br/programas-de-bolsas-de-iniciacao-cientifica-e-inovacao-seguem-com-inscricoes-abertas-ate-17-de-junho/",
    "https://propp.ufms.br/edital-pibic-pibic-af-pibiti-2025/",
    "https://prograd.ufms.br/files/2023/01/Tutorial-Monitoria-Elaboracao-de-Relatorios-Finais-Itens-5-e-6.pdf",

]

# --------------------------------------------------------------------------
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (trabalho-academico-chatbot-editais)"
VISITED: set[str] = set()


@dataclass
class Link:
    url: str
    text: str      # texto do próprio link (ou title/alt)
    context: str   # texto do elemento HTML que contém o link
    kind: str      # "a" ou "embed" (iframe/embed/object)


@dataclass
class Fetched:
    is_pdf: bool
    response: requests.Response | None = None
    html: str | None = None


# ----------------------------- utilidades ---------------------------------
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def looks_like_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def context_of(tag) -> str:
    """Sobe na árvore HTML enquanto o texto do ancestral for curto; devolve o maior."""
    best = tag.get_text(" ", strip=True)
    for parent in tag.parents:
        if parent.name in ("body", "html", "[document]"):
            break
        text = parent.get_text(" ", strip=True)
        if len(text) > MAX_CONTEXT:
            break
        best = text
    return best


def extract_links(html: str, base_url: str) -> list[Link]:
    soup = BeautifulSoup(html, "html.parser")
    attr_by_tag = {"a": "href", "iframe": "src", "embed": "src", "object": "data", "source": "src"}
    links = []
    for tag in soup.find_all(list(attr_by_tag)):
        raw = tag.get(attr_by_tag[tag.name])
        if not raw or raw.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = urldefrag(urljoin(base_url, raw.strip()))[0]
        if not url.startswith("http"):
            continue
        text = tag.get_text(" ", strip=True) or tag.get("title", "") or tag.get("alt", "")
        links.append(Link(
            url=url,
            text=text,
            context=context_of(tag),
            kind="a" if tag.name == "a" else "embed",
        ))
    return links


# ------------------------------- filtros ----------------------------------
# Cada filtro recebe um Link e devolve True para MANTER. Pra adicionar um novo,
# escreva a função e coloque na lista FILTERS no final desta seção.
def not_media(link: Link) -> bool:
    return Path(urlparse(link.url).path).suffix.lower() not in MEDIA_EXT


def allowed_domain(link: Link) -> bool:
    if ALLOWED_DOMAINS is None:
        return True
    host = urlparse(link.url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def link_haystack(link: Link) -> str:
    return normalize(f"{link.text} {link.context} {unquote(link.url)}")


def not_excluded(link: Link) -> bool:
    hay = link_haystack(link)
    return not any(k in hay for k in EXCLUDE_KEYWORDS)


def is_relevant(link: Link) -> bool:
    # PDFs diretos e conteúdo embutido (iframe/embed) herdam a relevância da
    # página onde estão; páginas comuns precisam casar com alguma palavra-chave
    if looks_like_pdf(link.url) or link.kind == "embed":
        return True
    hay = link_haystack(link)
    return any(k in hay for k in INCLUDE_KEYWORDS)


FILTERS = [not_media, allowed_domain, not_excluded, is_relevant]


def filter_pdf_links(website_link: str, html: str) -> list[Link]:
    """Links da página que passam em todos os filtros (sem repetir os já visitados)."""
    kept, seen = [], set()
    for link in extract_links(html, website_link):
        if link.url in seen or link.url in VISITED:
            continue
        seen.add(link.url)
        if all(f(link) for f in FILTERS):
            kept.append(link)
    return kept


# ------------------------------- rede -------------------------------------
def fetch(url: str) -> Fetched | None:
    """Faz um único GET e descobre pelo Content-Type se é PDF ou HTML."""
    try:
        r = session.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [erro] {url}: {e}")
        return None
    finally:
        time.sleep(DELAY)

    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" in ctype:
        return Fetched(is_pdf=True, response=r)
    if "html" in ctype:
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding
        return Fetched(is_pdf=False, html=r.text)
    r.close()
    return None


def download_pdf_from(url: str, link: Link, chain: list[str], results: list[dict],
                      response: requests.Response | None = None) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        if response is None:
            response = session.get(url, timeout=TIMEOUT, stream=True)
            response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [erro download] {url}: {e}")
        return

    stem = re.sub(r"[^\w\-]+", "_", Path(unquote(urlparse(url).path)).stem)[:80] or "documento"
    path = OUTPUT_DIR / f"{hashlib.sha1(url.encode()).hexdigest()[:8]}_{stem}.pdf"

    first = True
    with open(path, "wb") as f:
        for chunk in response.iter_content(8192):
            if first:
                first = False
                if not chunk.startswith(b"%PDF"):
                    print(f"  [ignorado] não é PDF de verdade: {url}")
                    f.close()
                    path.unlink(missing_ok=True)
                    return
            f.write(chunk)
    time.sleep(DELAY)

    results.append({
        "pdf_path": path.as_posix(),
        "downloaded_from": url,
        "previous_link": chain[-2] if len(chain) > 1 else None,
        "link_text": link.text,
        "link_context": link.context,
        "chain": chain,
    })
    print(f"  [pdf] {path.name}  <-  {url}")


# ------------------------------ recursão ----------------------------------
def find_pdfs(website_links: list[Link], chain: list[str], depth: int,
              results: list[dict]) -> None:
    for link in website_links:
        url = link.url
        if url in VISITED:
            continue
        VISITED.add(url)
        new_chain = chain + [url]

        if looks_like_pdf(url):
            download_pdf_from(url, link, new_chain, results)
            continue
        if depth >= MAX_DEPTH:
            continue

        print(f"{'  ' * depth}> {url}")
        fetched = fetch(url)
        if fetched is None:
            continue
        if fetched.is_pdf:  # URL sem .pdf que na verdade serve um PDF (ex.: Boletim Oficial)
            download_pdf_from(url, link, new_chain, results, fetched.response)
            continue

        child_links = filter_pdf_links(url, fetched.html)
        find_pdfs(child_links, new_chain, depth + 1, results)


def main() -> None:
    results: list[dict] = []
    seeds = [Link(u, "", "", "a") for u in SEED_LINKS]
    try:
        find_pdfs(seeds, [], 0, results)
    finally:  # salva o índice mesmo se interromper com Ctrl+C
        JSON_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(results)} PDFs baixados. Índice em {JSON_PATH}")


if __name__ == "__main__":
    main()