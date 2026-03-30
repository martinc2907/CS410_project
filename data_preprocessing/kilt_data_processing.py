import json
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:

        i = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            doc_id = entry["wikipedia_id"]
            title = entry["wikipedia_title"]
            passages = entry.get("text", [])

            for idx, p in enumerate(passages):
                cleaned = clean_passage(p)

                if len(cleaned) < 40:
                    continue

                out = {
                    "passage_id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": cleaned
                }

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            i += 1

            if i == stop_idx:
                break


if __name__ == "__main__":
    input_path = "kilt_knowledgesource.json"
    output_path = "kilt_passages.jsonl"

    main(stop_idx=50)