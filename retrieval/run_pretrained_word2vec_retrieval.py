# %% [markdown]
# # Pure Pretrained Word2Vec Retrieval for FEVER
#
# This is a JupyterLab-friendly script using `# %%` cells.
#
# This version does **not** use TF-IDF.
#
# Method:
# 1. Load pretrained Google News Word2Vec.
# 2. Aggregate KILT passages into one text per `doc_id`.
# 3. Represent each document as a title-boosted, SIF-weighted average of Word2Vec vectors.
# 4. Represent each FEVER claim the same way.
# 5. Retrieve top-k docs by cosine similarity.
#
# SIF weighting uses corpus word frequency:
#
# `weight(w) = a / (a + p(w))`
#
# This is still a pure Word2Vec/static-embedding retrieval method, not TF-IDF.

# %%
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# In Jupyter, install if needed:
# !pip install gensim numpy tqdm
import gensim.downloader as api
from gensim.models import KeyedVectors
from tqdm.auto import tqdm


# %% [markdown]
# ## Config
#
# Set `PROJECT_DIR` to your project folder.
#
# `PRETRAINED_MODE`:
# - `"download"` downloads/caches `word2vec-google-news-300` through gensim.
# - `"local"` loads a local `GoogleNews-vectors-negative300.bin.gz`.
#
# Hardware:
# - GPU is not needed for classic Word2Vec lookup + cosine retrieval.
# - CPU is fine.
# - RAM matters most. Google News Word2Vec can need 4GB+ RAM by itself.

# %%
PROJECT_DIR = Path(".").resolve()

CLAIMS_PATH = PROJECT_DIR / "fever_data" / "train.jsonl"
CORPUS_DIR = PROJECT_DIR / "kilt_passages_minimal_fever_evidence"

OUTPUT_FULL = PROJECT_DIR / "fever_retrieved_docs_pretrained_word2vec_pure.jsonl"
OUTPUT_SLIM = PROJECT_DIR / "fever_retrieved_doc_ids_pretrained_word2vec_pure.jsonl"

PRETRAINED_MODE = "download"  # "download" or "local"
LOCAL_WORD2VEC_PATH = PROJECT_DIR / "GoogleNews-vectors-negative300.bin.gz"

TOP_DOCS = 5
BATCH_SIZE = 2048
TITLE_WEIGHT = 5
SIF_A = 1e-3

# For debugging, set a small number like 1000. Use None for full data.
MAX_CLAIMS = None


# %% [markdown]
# ## Helpers

# %%
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def gold_titles_from_evidence(evidence: list) -> set[str]:
    titles = set()
    for group in evidence or []:
        for item in group or []:
            if len(item) >= 3 and item[2]:
                titles.add(str(item[2]).replace("_", " "))
    return titles


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


# %% [markdown]
# ## Load Claims

# %%
claims = []
for row in iter_jsonl(CLAIMS_PATH):
    claims.append(row)
    if MAX_CLAIMS is not None and len(claims) >= MAX_CLAIMS:
        break

print(f"Loaded {len(claims):,} claims")


# %% [markdown]
# ## Load KILT Corpus and Aggregate by Document
#
# We retrieve document ids because Martin's eval expects doc ids.
# The title is repeated because FEVER claims are entity-heavy, and Wikipedia
# titles are very informative for document retrieval.

# %%
doc_tokens = defaultdict(list)
doc_titles = {}
word_counts = Counter()
total_tokens = 0

corpus_files = sorted(CORPUS_DIR.glob("*.jsonl"))
start = time.time()

for path in corpus_files:
    for row in tqdm(iter_jsonl(path), desc=f"Reading {path.name}"):
        doc_id = str(row.get("doc_id"))
        title = row.get("title") or ""
        text = row.get("text") or ""
        tokens = tokenize((" ".join([title] * TITLE_WEIGHT)) + " " + text)

        doc_titles.setdefault(doc_id, title)
        doc_tokens[doc_id].extend(tokens)
        word_counts.update(tokens)
        total_tokens += len(tokens)

print(f"Loaded {len(doc_tokens):,} docs")
print(f"Total tokens: {total_tokens:,}")
print(f"Corpus load time: {(time.time() - start) / 60:.1f} min")


# %% [markdown]
# ## Load Pretrained Word2Vec
#
# First run may download around 1.5GB. Later runs use the local gensim cache.

# %%
start = time.time()

if PRETRAINED_MODE == "download":
    wv = api.load("word2vec-google-news-300")
elif PRETRAINED_MODE == "local":
    wv = KeyedVectors.load_word2vec_format(str(LOCAL_WORD2VEC_PATH), binary=True)
else:
    raise ValueError(f"Unknown PRETRAINED_MODE: {PRETRAINED_MODE}")

print(f"Loaded pretrained vectors: {len(wv):,} tokens, dim={wv.vector_size}")
print(f"Load time: {(time.time() - start) / 60:.1f} min")


# %% [markdown]
# ## Build Pure Word2Vec Embeddings
#
# Uses SIF word weighting from corpus frequency, not TF-IDF.

# %%
def sif_weight(token: str) -> float:
    prob = word_counts.get(token, 0) / max(total_tokens, 1)
    return SIF_A / (SIF_A + prob)


def weighted_word2vec(tokens: list[str]) -> np.ndarray:
    vec_sum = np.zeros(wv.vector_size, dtype=np.float32)
    weight_sum = 0.0

    for token in tokens:
        if token not in wv:
            continue
        weight = sif_weight(token)
        vec_sum += wv[token].astype(np.float32) * weight
        weight_sum += weight

    if weight_sum == 0.0:
        return vec_sum
    return vec_sum / weight_sum


# %%
doc_ids = list(doc_tokens.keys())
doc_matrix = np.zeros((len(doc_ids), wv.vector_size), dtype=np.float32)

for i, doc_id in enumerate(tqdm(doc_ids, desc="Embedding docs")):
    doc_matrix[i] = weighted_word2vec(doc_tokens[doc_id])

doc_matrix = normalize_matrix(doc_matrix)
doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

print(f"Doc matrix shape: {doc_matrix.shape}")


# %% [markdown]
# ## Retrieve Top Docs for Claims

# %%
def claim_embedding(claim_text: str) -> np.ndarray:
    # Repeat claim tokens once only. Title boosting applies to docs, not claims.
    return weighted_word2vec(tokenize(claim_text))


start = time.time()
recall_hits = 0
precision_sum = 0.0
claims_with_gold = 0
relevant_retrieved = 0
retrieved_total = 0

with OUTPUT_FULL.open("w", encoding="utf-8") as full_out, OUTPUT_SLIM.open(
    "w", encoding="utf-8"
) as slim_out:
    for batch_start in tqdm(range(0, len(claims), BATCH_SIZE), desc="Retrieving"):
        batch = claims[batch_start : batch_start + BATCH_SIZE]
        claim_matrix = np.zeros((len(batch), wv.vector_size), dtype=np.float32)

        for i, row in enumerate(batch):
            claim_matrix[i] = claim_embedding(row.get("claim") or "")

        claim_matrix = normalize_matrix(claim_matrix)
        scores = claim_matrix @ doc_matrix.T

        # top-k unsorted candidates, then sort only top-k
        top_idx = np.argpartition(-scores, kth=TOP_DOCS - 1, axis=1)[:, :TOP_DOCS]

        for row_idx, row in enumerate(batch):
            indices = top_idx[row_idx]
            indices = indices[np.argsort(-scores[row_idx, indices])]

            retrieved = []
            top_doc_ids = []
            for idx in indices:
                doc_id = doc_ids[int(idx)]
                top_doc_ids.append(doc_id)
                retrieved.append(
                    {
                        "doc_id": doc_id,
                        "score": round(float(scores[row_idx, idx]), 6),
                        "title": doc_titles.get(doc_id, ""),
                    }
                )

            full_row = {
                "id": row.get("id"),
                "verifiable": row.get("verifiable"),
                "label": row.get("label"),
                "claim": row.get("claim"),
                "evidence": row.get("evidence"),
                "top_doc_ids": top_doc_ids,
                "retrieved": retrieved,
            }
            slim_row = {
                "id": row.get("id"),
                "label": row.get("label"),
                "claim": row.get("claim"),
                "top_doc_ids": top_doc_ids,
            }

            full_out.write(json.dumps(full_row, ensure_ascii=False) + "\n")
            slim_out.write(json.dumps(slim_row, ensure_ascii=False) + "\n")

            gold_titles = gold_titles_from_evidence(row.get("evidence"))
            if gold_titles:
                claims_with_gold += 1
                retrieved_titles = [item["title"] for item in retrieved]
                rel = sum(1 for title in retrieved_titles if title in gold_titles)
                relevant_retrieved += rel
                retrieved_total += len(retrieved)
                precision_sum += rel / max(len(retrieved), 1)
                if rel > 0:
                    recall_hits += 1

elapsed = time.time() - start
print(f"Done retrieval in {elapsed / 60:.1f} min")

if claims_with_gold:
    print(f"Recall@{TOP_DOCS}: {recall_hits / claims_with_gold:.4f}")
    print(f"Macro Precision@{TOP_DOCS}: {precision_sum / claims_with_gold:.4f}")
    print(f"Micro Precision@{TOP_DOCS}: {relevant_retrieved / max(retrieved_total, 1):.4f}")
    print(f"Recall hits: {recall_hits:,}/{claims_with_gold:,}")
    print(f"Relevant retrieved: {relevant_retrieved:,}/{retrieved_total:,}")

print(f"Full output: {OUTPUT_FULL}")
print(f"Slim output: {OUTPUT_SLIM}")


# %% [markdown]
# ## Inspect Output

# %%
for path in [OUTPUT_SLIM, OUTPUT_FULL]:
    print(path, f"{path.stat().st_size / (1024 * 1024):.1f} MB")
    with path.open("r", encoding="utf-8") as f:
        print(f.readline()[:1000])

