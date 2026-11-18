# restaurar.py  (na pasta documentos/, ao lado de filtrar.py)
import argparse, csv, shutil, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUARENTENA = ROOT / "_quarentena"
RELATORIOS = [ROOT / "relatorio_relevancia_1.csv", ROOT / "relatorio_relevancia_2.csv"]
MESCLADO = ROOT / "relatorio_mesclado.csv"

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))

def eh_sim(v: str) -> bool:
    return norm(v) in {"sim", "s", "yes", "y", "1", "true"}

def ler(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=";"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true", help="move de fato (padrão: dry-run)")
    args = ap.parse_args()

        # Mescla: chave = caminho em 'arquivo'. O relatório 2 sobrescreve o 1.
    mesclado, campos = {}, None
    for rel in RELATORIOS:                 # ordem importa: 1 primeiro, 2 por último
        if not rel.exists():
            print("AVISO: não encontrado:", rel.name)
            continue
        for row in ler(rel):
            campos = campos or list(row.keys())
            mesclado[row["arquivo"].strip()] = row   # o último a aparecer vence

    linhas = sorted(mesclado.values(), key=lambda r: float(r["score"].replace(",", ".")))
    with open(MESCLADO, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, delimiter=";")
        w.writeheader()
        w.writerows(linhas)
    print(f"Relatório mesclado: {MESCLADO.name} ({len(linhas)} arquivos)\n")

    restaurar = [r for r in linhas if eh_sim(r["manter"])]
    movidos = 0
    for r in restaurar:
        rel = r["arquivo"].strip().removeprefix("./")
        origem_q = QUARENTENA / rel
        destino = ROOT / rel
        if destino.exists():
            continue                                   # já está no lugar
        if not origem_q.exists():
            print("NÃO ENCONTRADO na quarentena:", rel)
            continue
        if ROOT.resolve() not in destino.resolve().parents:
            print("IGNORADO (fora da pasta):", rel)
            continue
        print(("RESTAURAR " if args.aplicar else "restauraria ") + rel)
        if args.aplicar:
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(origem_q, destino)
        movidos += 1

    print(f"\n{movidos} arquivo(s) {'restaurado(s)' if args.aplicar else 'a restaurar'}.")
    if not args.aplicar:
        print("(dry-run: nada foi movido; use --aplicar)")

if __name__ == "__main__":
    main()