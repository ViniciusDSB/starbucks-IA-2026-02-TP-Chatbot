# filtrar.py  (colocar na pasta documentos/ ao lado de FONTES.md e intencoes.yaml)
import argparse, csv, re, shutil
from pathlib import Path
import fitz                      # pip install pymupdf
import yaml                      # pip install pyyaml
from bs4 import BeautifulSoup    # pip install beautifulsoup4
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent
QUARENTENA = ROOT / "_quarentena"
FONTES = ROOT / "FONTES.md"
YAML = ROOT / "intencoes.yaml"
RELATORIO = ROOT / "relatorio_relevancia.csv"
EXT = {".pdf", ".html", ".htm"}
KW = re.compile(r"inicia[çc][ãa]o cient[íi]fica|monitoria|pibic|pibiti|elabora[çc][ãa]o de (relat[óo]rio|trabalhos)|volunt[aá]rio|remunerad[oa]", re.I)

# Exemplos do que NÃO interessa (ajuste/adicione conforme o que aparecer no relatório)
NEGATIVAS = [
    "Edital de incentivo à participação de mulheres na ciência",
    "Edital de solicitação e empréstimo de equipamentos de pesquisa e laboratório",
    " MANUTENÇÃO DE EQUIPAMENTOS DE LABORATÓRIOS DE PESQUISA",
    "RELATÓRIO DE GESTÃO",
    "REVALIDAÇÃO E RECONHECIMENTO DE DIPLOMAS",
    "Reconhecimento de Diplomas",
    "SELEÇÃO DE PROPOSTAS PARA APOIO NA INSTALAÇÃO E/OU MANUTENÇÃO DE EQUIPAMENTOS DE LABORATÓRIOS",
    "Cartão Pesquisa UFMS",
    "Dispõe sobre o acesso ao patrimônio genético, sobre a proteção e o acesso ao conhecimento",
    "Conselho de Gestão do Patrimônio Genético (CGen)",
    "CGen",
    "CADASTRO DE LIGAS ACADÊMICAS",
    "adastro de Ligas Acadêmicas na UFMS",
    "PROFESSOR SUBSTITUTO",
    "Bolsa Mérito concedida pelo Ministério das Relações Exteriores"
]

def extrair(p: Path, limite=2000) -> str:
    if p.suffix.lower() == ".pdf":
        with fitz.open(p) as d:
            txt = " ".join(pg.get_text() for pg in d[:3])
    else:
        soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        titulo = soup.title.get_text(" ") if soup.title else ""
        txt = titulo + " " + soup.get_text(" ")
    return " ".join(txt.split())[:limite]

def trechos(t, n=500):
    return [t[i:i + n] for i in range(0, len(t), n)]

def chave_fontes(f: Path) -> str:
    return "./" + f.relative_to(ROOT).as_posix()

def atualizar_fontes(removidos: set[str]):
    linhas = FONTES.read_text(encoding="utf-8").splitlines()
    novas = []
    for l in linhas:
        if l.startswith("|"):
            celula = l.split("|")[1].strip().strip("`")
            if celula in removidos:
                continue
        novas.append(l)
    FONTES.write_text("\n".join(novas) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limiar", type=float, default=0.35)
    ap.add_argument("--aplicar", action="store_true", help="move para _quarentena e atualiza FONTES.md")
    args = ap.parse_args()

    intencoes = yaml.safe_load(YAML.read_text(encoding="utf-8"))["intencoes"]
    rotulos = list(intencoes)
    refs_pos = [" ".join(intencoes[k]["descricao"].split()) for k in rotulos]

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb_pos = model.encode(refs_pos, convert_to_tensor=True)
    emb_neg = model.encode(NEGATIVAS, convert_to_tensor=True)

    arquivos = sorted(f for f in ROOT.rglob("*")
                      if f.suffix.lower() in EXT and QUARENTENA not in f.parents)
    docs = []
    for f in arquivos:
        try:
            t = extrair(f)
        except Exception as e:
            print("ERRO ao ler", f, e)
            t = ""
        docs.append((f, t, trechos(t)))

    todos = [c for _, _, cs in docs for c in cs]
    emb = model.encode(todos, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    sim_pos = util.cos_sim(emb, emb_pos)   # trechos x intenções
    sim_neg = util.cos_sim(emb, emb_neg)   # trechos x negativas

    linhas, i = [], 0
    for f, t, cs in docs:
        n = len(cs)
        if n == 0:
            linhas.append(dict(f=f, pos=0.0, neg=0.0, intencao="", kw=0, trecho="", sem_texto=True))
            continue
        sp, sn = sim_pos[i:i + n], sim_neg[i:i + n]
        i += n
        melhor = sp.max(dim=0).values            # melhor trecho por intenção
        linhas.append(dict(
            f=f, pos=melhor.max().item(), neg=sn.max().item(),
            intencao=rotulos[melhor.argmax().item()],
            kw=len(KW.findall(t)), trecho=t[:80], sem_texto=len(t) < 50))

    for l in linhas:
        l["manter"] = (not l["sem_texto"]) and l["pos"] >= args.limiar and l["pos"] > l["neg"]

    # Relatório ordenado do menos para o mais relevante
    linhas.sort(key=lambda l: l["pos"])
    with open(RELATORIO, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["score", "score_neg", "kw", "melhor_intencao", "manter", "arquivo", "trecho"])
        for l in linhas:
            w.writerow([f"{l['pos']:.3f}", f"{l['neg']:.3f}", l["kw"], l["intencao"],
                        "sim" if l["manter"] else "NAO", chave_fontes(l["f"]), l["trecho"]])
            print(f"{l['pos']:.2f} neg={l['neg']:.2f} kw={l['kw']:<3} {'ok ' if l['manter'] else 'XXX'} {chave_fontes(l['f'])}")
    print(f"\nRelatório: {RELATORIO}  |  limiar={args.limiar}  |  a remover: {sum(not l['manter'] for l in linhas)}")

    if not args.aplicar:
        print("(dry-run: nada foi movido; use --aplicar)")
        return

    removidos = set()
    for l in linhas:
        if l["manter"]:
            continue
        destino = QUARENTENA / l["f"].relative_to(ROOT)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(l["f"], destino)
        removidos.add(chave_fontes(l["f"]))
    atualizar_fontes(removidos)
    print(f"{len(removidos)} arquivos movidos para {QUARENTENA} e removidos do FONTES.md")

if __name__ == "__main__":
    main()