import argparse
import json
from pathlib import Path

from kilt_data_processing import clean_passage


FEVER_FILES = ("train.jsonl", "shared_task_dev.jsonl")


def preprocess_fever_file(input_path: Path, output_path: Path) -> int:
    count = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue

            example = json.loads(line)
            example["claim"] = clean_passage(example["claim"], clean_minimal=False)

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    return count


def preprocess_fever_dir(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in FEVER_FILES:
        input_path = input_dir / filename
        output_path = output_dir / filename
        count = preprocess_fever_file(input_path, output_path)
        print(f"Wrote {count} examples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply KILT passage preprocessing to FEVER claim text."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("CS410_project") / "fever_data",
        help="Folder containing train.jsonl and shared_task_dev.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CS410_project") / "fever_data_preprocessed",
        help="Folder for preprocessed FEVER JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_fever_dir(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
