import json
from collections import defaultdict
from pathlib import Path

PASSAGES_DIR = Path("../data/kilt/kilt_passages_minimal_fever_evidence")
OUTPUT_PATH = Path("../data/kilt/kilt_docs_fever_evidence.jsonl")


def build_index():
    docs: dict[str, dict] = {}

    files = sorted(PASSAGES_DIR.glob("kilt_passages_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No passage files found in {PASSAGES_DIR}")

    for path in files:
        print(f"Reading {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                doc_id = p["doc_id"]
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "title": p["title"],
                        "passages": [],
                    }
                docs[doc_id]["passages"].append({
                    "passage_id": p["passage_id"],
                    "text": p["text"],
                })

    for d in docs.values():
        d["passages"].sort(key=lambda x: int(x["passage_id"].split("_")[-1]))

    print(f"Writing {len(docs)} docs to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for doc in docs.values():
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_doc_index(path: str | Path = OUTPUT_PATH) -> dict[str, dict]:
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            index[doc["doc_id"]] = doc
    return index


if __name__ == "__main__":
    build_index()
