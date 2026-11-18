
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, json_path):
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def get_filename(item):
    for key in ("pdf_path", "html_path", "path"):
        value = item.get(key)
        if value:
            return Path(value).name
    return None


def flatten_entries(data):
    entries = []

    if isinstance(data, dict):
        for intention, items in data.items():
            if not isinstance(items, list):
                continue

            for item in items:
                if isinstance(item, dict):
                    entry = item.copy()
                    entry.setdefault("intention", intention)
                    entries.append(entry)

    elif isinstance(data, list):
        entries = [item.copy() for item in data if isinstance(item, dict)]

    return entries


def scan_directory(general_dir):
    files_by_path = {}

    for file_path in general_dir.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(general_dir.parent).as_posix()
        intention = file_path.relative_to(general_dir).parts[0]

        files_by_path[relative_path] = {
            "path": relative_path,
            "intention": intention,
            "filename": file_path.name,
        }

    return files_by_path


def synchronize(existing_data, files_by_path):
    old_entries = flatten_entries(existing_data)

    entries_by_filename = defaultdict(list)
    entries_by_path = {}

    for entry in old_entries:
        filename = get_filename(entry)
        old_path = entry.get("path")

        if filename:
            entries_by_filename[filename].append(entry)

        if old_path:
            entries_by_path[old_path.replace("\\", "/")] = entry

    updated_entries = []
    matched_ids = set()
    new_files = []
    updated_count = 0

    for current_path, current_file in sorted(files_by_path.items()):
        entry = entries_by_path.get(current_path)

        if entry is None:
            candidates = entries_by_filename.get(
                current_file["filename"], []
            )
            unmatched = [
                candidate for candidate in candidates
                if id(candidate) not in matched_ids
            ]

            if len(unmatched) == 1:
                entry = unmatched[0]
            elif len(unmatched) > 1:
                print(
                    f"[AMBIGUOUS] {current_path}: "
                    "multiple JSON entries have the same filename."
                )
                new_files.append(current_file)
                continue
            else:
                new_files.append(current_file)
                continue

        matched_ids.add(id(entry))

        updated_entry = entry.copy()
        updated_entry["path"] = current_path

        for key in ("pdf_path", "html_path"):
            if key in updated_entry:
                updated_entry[key] = current_path

        if updated_entry.get("intention") != current_file["intention"]:
            updated_entry["intention"] = current_file["intention"]

        if updated_entry != entry:
            updated_count += 1

        updated_entries.append(updated_entry)

    removed_count = len(old_entries) - len(matched_ids)

    grouped_entries = defaultdict(list)

    for entry in updated_entries:
        intention = entry.get("intention", "uncategorized")
        grouped_entries[intention].append(entry)

    result = {
        intention: sorted(
            items,
            key=lambda item: item.get("path", "")
        )
        for intention, items in sorted(grouped_entries.items())
    }

    return result, updated_count, removed_count, new_files


def main():
    parser = argparse.ArgumentParser(
        description="Synchronizes the JSON index with files in the general directory."
    )
    parser.add_argument(
        "--directory",
        default="../geral",
        help="Directory containing the organized files."
    )
    parser.add_argument(
        "--json",
        default="../arquivos_por_intencao.json",
        help="Path to the JSON index."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Save the changes. Without this flag, only preview the results."
    )
    args = parser.parse_args()

    general_dir = Path(args.directory)
    json_path = Path(args.json)

    if not general_dir.is_dir():
        parser.error(f"Directory not found: {general_dir}")

    if not json_path.is_file():
        parser.error(f"JSON file not found: {json_path}")

    existing_data = load_json(json_path)
    files_by_path = scan_directory(general_dir)

    result, updated_count, removed_count, new_files = synchronize(
        existing_data,
        files_by_path
    )

    retained_count = sum(len(items) for items in result.values())

    print(f"Files found in directory: {len(files_by_path)}")
    print(f"Entries retained: {retained_count}")
    print(f"Entries updated: {updated_count}")
    print(f"Entries removed: {removed_count}")
    print(f"New or ambiguous files: {len(new_files)}")

    for item in new_files:
        print(f"[UNMATCHED] {item['path']}")

    if args.apply:
        backup_path = json_path.with_suffix(".json.bak")
        backup_path.write_text(
            json_path.read_text(encoding="utf-8"),
            encoding="utf-8"
        )
        save_json(result, json_path)
        print(f"\nIndex updated: {json_path}")
        print(f"Backup created: {backup_path}")
    else:
        print("\nPreview only. Run with --apply to save the changes.")


if __name__ == "__main__":
    main()
