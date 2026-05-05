# Retrieval Methods

This folder contains Lai's FEVER/KILT retrieval implementations.

## Data Layout

Run scripts from the repository root with the processed data in:

```text
fever_data/
  train.jsonl
  shared_task_dev.jsonl

kilt_passages_minimal_fever_evidence/
  kilt_passages_00000.jsonl
  kilt_passages_00001.jsonl
```

Large data and retrieval outputs are intentionally not tracked by Git.

## Install

```bash
pip install -r retrieval/requirements.txt
```

## TF-IDF Retrieval

```bash
python retrieval/run_tfidf_retrieval.py \
  --claims fever_data/train.jsonl \
  --corpus-dir kilt_passages_minimal_fever_evidence \
  --output fever_retrieved_docs_tfidf.jsonl \
  --top-docs 5
```

The output JSONL includes:

```text
id, verifiable, label, claim, evidence, top_doc_ids, retrieved
```

For eval, `top_doc_ids` is the main field.

## Pure Pretrained Word2Vec Retrieval

The Word2Vec script is a Jupyter/VS Code cell-style Python script (`# %%`) and
can also be run as a normal Python file after editing the config block.

```bash
python retrieval/run_pretrained_word2vec_retrieval.py
```

It uses Google News pretrained Word2Vec by default:

```python
api.load("word2vec-google-news-300")
```

The first run downloads about 1.5GB through gensim. CPU is sufficient; RAM is
the main constraint.

Method summary:

- aggregate KILT passages by `doc_id`
- repeat Wikipedia title tokens to emphasize entity names
- build SIF-weighted average Word2Vec vectors for claims and documents
- rank documents by cosine similarity

