import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Organiza arquivos em pastas conforme o JSON agrupado por intenção."
    )
    parser.add_argument(
        "--json", default="arquivos_por_intencao.json",
        help="Caminho do JSON agrupado por intenção."
    )
    parser.add_argument(
        "--source-root", default=".",
        help="Pasta-base onde estão os arquivos de origem (por exemplo, a pasta que contém files/)."
    )
    parser.add_argument(
        "--output-root", default=".",
        help="Pasta-base de destino. O valor do campo 'path' do JSON é relativo a ela."
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copia os arquivos (padrão). Esta opção existe para deixar explícita a operação."
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move os arquivos em vez de copiá-los."
    )
    args = parser.parse_args()

    if args.copy and args.move:
        parser.error("Use --copy ou --move, não ambos.")

    json_path = Path(args.json)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    with json_path.open("r", encoding="utf-8") as file:
        grouped_files = json.load(file)

    transfer = shutil.move if args.move else shutil.copy2
    copied = 0
    missing = []

    for intention, files in grouped_files.items():
        for item in files:
            source_key = "pdf_path" if "pdf_path" in item else "html_path" if "html_path" in item else None
            if source_key is None:
                print(f"[IGNORADO] Item sem pdf_path/html_path: {item}")
                continue

            source = source_root / item[source_key]
            destination = output_root / item["path"]

            if not source.is_file():
                missing.append(source)
                print(f"[AUSENTE] {source}")
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            transfer(str(source), str(destination))
            copied += 1
            print(f"[OK] {source} -> {destination}")

    print(f"\nConcluído: {copied} arquivo(s) {'movido(s)' if args.move else 'copiado(s)'}.")
    if missing:
        print(f"{len(missing)} arquivo(s) de origem não foram encontrados.")


if __name__ == "__main__":
    main()
