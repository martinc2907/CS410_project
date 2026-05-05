#!/usr/bin/env python3
"""TF-IDF retrieval for FEVER claims over KILT passages.

Output is JSONL with one row per claim. Each row includes top_doc_ids for
downstream evaluation, plus richer retrieved metadata for debugging.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_passages(corpus_dir: Path, max_passages: int | None = None) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    meta: list[dict] = []

    files = sorted(corpus_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {corpus_dir}")

    for path in files:
        for row in iter_jsonl(path):
            title = row.get("title") or ""
            text = row.get("text") or ""
            texts.append(f"{title} {text}")
            meta.append(
                {
                    "passage_id": str(row.get("passage_id")),
                    "doc_id": str(row.get("doc_id")),
                    "title": title,
                }
            )
            if max_passages is not None and len(texts) >= max_passages:
                return texts, meta

    return texts, meta


def load_claims(claims_path: Path, max_claims: int | None = None) -> list[dict]:
    claims: list[dict] = []
    for row in iter_jsonl(claims_path):
        claims.append(row)
        if max_claims is not None and len(claims) >= max_claims:
            break
    return claims


def gold_titles_from_evidence(evidence: list) -> set[str]:
    titles: set[str] = set()
    for group in evidence or []:
        for item in group or []:
            if len(item) >= 3 and item[2]:
                titles.add(str(item[2]).replace("_", " "))
    return titles


def dedupe_docs(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    passage_meta: list[dict],
    top_docs: int,
) -> tuple[list[str], list[dict]]:
    seen: set[str] = set()
    top_doc_ids: list[str] = []
    retrieved: list[dict] = []

    for idx, distance in zip(neighbor_indices, neighbor_distances):
        item = passage_meta[int(idx)]
        doc_id = item["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        top_doc_ids.append(doc_id)
        retrieved.append(
            {
                "doc_id": doc_id,
                "score": round(float(1.0 - distance), 6),
                "best_passage_id": item["passage_id"],
                "title": item["title"],
            }
        )
        if len(top_doc_ids) >= top_docs:
            break

    return top_doc_ids, retrieved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF retrieval for FEVER claims.")
    parser.add_argument("--claims", type=Path, default=Path("fever_data/train.jsonl"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("kilt_passages_minimal_fever_evidence"))
    parser.add_argument("--output", type=Path, default=Path("fever_retrieved_docs_tfidf.jsonl"))
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument(
        "--passage-neighbors",
        type=int,
        default=50,
        help="Passage candidates to retrieve before deduplicating to document ids.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-features", type=int, default=500_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-claims", type=int, default=None)
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--no-bigrams", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    print(f"Loading passages from {args.corpus_dir}...")
    passage_texts, passage_meta = load_passages(args.corpus_dir, args.max_passages)
    print(f"Loaded {len(passage_texts):,} passages.")

    print(f"Loading claims from {args.claims}...")
    claims = load_claims(args.claims, args.max_claims)
    claim_texts = [str(row.get("claim") or "") for row in claims]
    print(f"Loaded {len(claims):,} claims.")

    ngram_range = (1, 1) if args.no_bigrams else (1, 2)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=args.max_features,
        min_df=args.min_df,
        dtype=np.float32,
    )

    print("Fitting TF-IDF vectorizer on passages...")
    passage_matrix = vectorizer.fit_transform(passage_texts)
    print(f"Passage matrix shape: {passage_matrix.shape}")

    n_neighbors = min(args.passage_neighbors, len(passage_meta))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    print("Fitting nearest-neighbor index...")
    nn.fit(passage_matrix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    verifiable_count = 0
    recall_hits = 0

    print(f"Writing retrieval output to {args.output}...")
    with args.output.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(claims), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(claims))
            batch_claims = claims[batch_start:batch_end]
            query_matrix = vectorizer.transform(claim_texts[batch_start:batch_end])
            distances, indices = nn.kneighbors(query_matrix, return_distance=True)

            for row, row_indices, row_distances in zip(batch_claims, indices, distances):
                top_doc_ids, retrieved = dedupe_docs(
                    row_indices,
                    row_distances,
                    passage_meta,
                    args.top_docs,
                )
                output_row = {
                    "id": row.get("id"),
                    "verifiable": row.get("verifiable"),
                    "label": row.get("label"),
                    "claim": row.get("claim"),
                    "evidence": row.get("evidence"),
                    "top_doc_ids": top_doc_ids,
                    "retrieved": retrieved,
                }
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

                gold_titles = gold_titles_from_evidence(row.get("evidence"))
                if gold_titles:
                    verifiable_count += 1
                    retrieved_titles = {item["title"] for item in retrieved}
                    if gold_titles & retrieved_titles:
                        recall_hits += 1

            print(f"Processed {batch_end:,}/{len(claims):,} claims.", flush=True)

    elapsed = time.time() - start
    if verifiable_count:
        print(f"Approx title recall@{args.top_docs}: {recall_hits / verifiable_count:.4f}")
        print(f"Recall hits: {recall_hits:,}/{verifiable_count:,}")
    print(f"Done in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
