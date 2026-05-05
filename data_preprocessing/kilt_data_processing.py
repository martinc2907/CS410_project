import json
import os
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_passage(text: str, clean_minimal=True) -> str:
    text = text.strip()

    # Skip section headers entirely
    if text.startswith("Section::::"):
        return ""

    # Remove BULLET prefix
    if text.startswith("BULLET::::"):
        text = text[len("BULLET::::"):].strip()

    if clean_minimal:
        return text
    
    # Normalize whitespace + lowercase
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize + remove stopwords + stem
    tokens = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(tokens)


def main(stop_idx=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin:
        fout = None
        shard_idx = 0
        passages_in_shard = 0
        i = 0

        try:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                entry = json.loads(line)

                doc_id = entry["wikipedia_id"]
                title = entry["wikipedia_title"]
                passages = entry.get("text", [])

                for idx, p in enumerate(passages):
                    cleaned = clean_passage(p, clean_minimal=False)

                    if len(cleaned) < 40:
                        continue

                    if fout is None or passages_in_shard == shard_size:
                        if fout is not None:
                            fout.close()

                        output_path = os.path.join(
                            output_dir,
                            f"kilt_passages_{shard_idx:05d}.jsonl"
                        )
                        fout = open(output_path, "w", encoding="utf-8")
                        shard_idx += 1
                        passages_in_shard = 0

                    out = {
                        "passage_id": f"{doc_id}_{idx}",
                        "doc_id": doc_id,
                        "title": title,
                        "text": cleaned
                    }

                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    passages_in_shard += 1
                i += 1

                if i % 10000 == 0:
                    print(i)

                if i == stop_idx:
                    break


        finally:
            if fout is not None:
                fout.close()


if __name__ == "__main__":
    input_path = "kilt_knowledgesource.json"
    output_dir = "kilt_passages_preprocessed"
    shard_size = 500000

    main()
