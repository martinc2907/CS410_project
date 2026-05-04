import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
DOC_INDEX_PATH = PROJECT_ROOT / "data" / "kilt" / "kilt_docs_fever_evidence.jsonl"
RETRIEVED_DIR = PROJECT_ROOT / "data" / "retrieved"
RETRIEVED_PATHS = {
    "transformer": RETRIEVED_DIR / "fever_retrieved_docs_new_trans.jsonl",
    "tfidf": RETRIEVED_DIR / "fever_retrieved_doc_ids_tfidf.jsonl",
    "word2vec": RETRIEVED_DIR / "fever_retrieved_doc_ids_pretrained_word2vec_pure.jsonl",
    "probabilistic": RETRIEVED_DIR / "fever_qlm_retrieved.jsonl",
}
DEFAULT_CLAIMS_PATH = RETRIEVED_PATHS["transformer"]
RESULTS_DIR = Path(__file__).parent / "results"

MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
SEED = 42

LABELS = ["SUPPORTS", "REFUTES"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "You are an expert fact-checker. Given a claim and a set of retrieved "
    "evidence passages from Wikipedia, decide whether the claim is SUPPORTS "
    "(true) or REFUTES (false). The retrieved passages may include irrelevant "
    "or off-topic material; ignore any passage that does not pertain to the "
    "claim and base your verdict only on the passages that are actually "
    "relevant. If no passage is clearly relevant, rely on your own knowledge. "
    "Respond with exactly one word: SUPPORTS or REFUTES."
)

USER_PROMPT_BASELINE = """Claim: {claim}

Your verdict (SUPPORTS or REFUTES):"""

USER_PROMPT_RETRIEVAL = """Evidence:
{evidence}

Claim: {claim}

Your verdict (SUPPORTS or REFUTES):"""


def load_doc_index(path: Path) -> dict[str, dict]:
    print(f"loading doc index from {path}")
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            index[doc["doc_id"]] = doc
    print(f"  {len(index)} docs indexed")
    return index


def load_claims(path: Path, limit: int | None) -> list[dict]:
    print(f"loading claims from {path}")
    claims = []
    skipped_nei = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["label"] not in LABEL_TO_ID:
                skipped_nei += 1
                continue
            claims.append(entry)
            if limit is not None and len(claims) >= limit:
                break
    print(f"  {len(claims)} claims loaded ({skipped_nei} NEI claims skipped)")
    return claims


def build_evidence_text(doc_ids, doc_index, max_passages_per_doc, max_chars):
    parts = []
    for did in doc_ids:
        doc = doc_index.get(str(did))
        if doc is None:
            continue
        title = doc["title"]
        for p in doc["passages"][:max_passages_per_doc]:
            parts.append(f"[{title}] {p['text']}")
    text = "\n\n".join(parts)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_messages(claim, evidence_text, mode):
    if mode == "retrieval" and evidence_text:
        user = USER_PROMPT_RETRIEVAL.format(evidence=evidence_text, claim=claim)
    else:
        user = USER_PROMPT_BASELINE.format(claim=claim)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_prediction(raw: str):
    cleaned = raw.strip().upper()
    if "REFUTE" in cleaned:
        return LABEL_TO_ID["REFUTES"]
    if "SUPPORT" in cleaned:
        return LABEL_TO_ID["SUPPORTS"]
    return None


def run_inference(claims, doc_index, model, tokenizer, mode, batch_size,
                  max_passages_per_doc, max_evidence_chars, max_input_tokens):
    results = []
    total = len(claims)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = claims[start:end]

        prompts = []
        for c in batch:
            evidence_text = ""
            if mode == "retrieval":
                evidence_text = build_evidence_text(
                    c.get("top_doc_ids", []),
                    doc_index,
                    max_passages_per_doc,
                    max_evidence_chars,
                )
            prompts.append(
                tokenizer.apply_chat_template(
                    build_messages(c["claim"], evidence_text, mode),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, c in enumerate(batch):
            new_tokens = outputs[i][inputs["input_ids"][i].shape[0]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            results.append({
                "id": c["id"],
                "claim": c["claim"][:200],
                "label": LABEL_TO_ID.get(c["label"]),
                "label_str": c["label"],
                "prediction": parse_prediction(response),
                "raw_response": response.strip(),
                "n_evidence_docs": len(c.get("top_doc_ids", [])) if mode == "retrieval" else 0,
            })

        valid = sum(1 for r in results if r["prediction"] is not None)
        print(f"  {end}/{total}  ({valid} parseable)", flush=True)

    return results


def evaluate(results):
    valid = [r for r in results if r["prediction"] is not None and r["label"] is not None]
    unparseable = len(results) - len(valid)

    labels = [r["label"] for r in valid]
    preds = [r["prediction"] for r in valid]

    acc = accuracy_score(labels, preds)
    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))

    print(f"\n--- results ({len(valid)}/{len(results)} parsed) ---")
    print(f"acc={acc:.4f}  macro_prec={prec_macro:.4f}  macro_rec={rec_macro:.4f}  macro_f1={f1_macro:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=LABELS, digits=4, zero_division=0)}")

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
        "label_order": LABELS,
        "total_samples": len(results),
        "valid_predictions": len(valid),
        "unparseable": unparseable,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "retrieval"], required=True,
                   help="baseline = claim only; retrieval = include retrieved passages")
    p.add_argument("--method", default="transformer",
                   choices=list(RETRIEVED_PATHS.keys()),
                   help="retrieval method; selects the retrieved-docs file and tags output")
    p.add_argument("--limit", type=int, default=1000,
                   help="number of claims to evaluate (None for all)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-passages-per-doc", type=int, default=3)
    p.add_argument("--max-evidence-chars", type=int, default=4000)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--retrieved-path", default=None,
                   help="override the retrieved-docs file path (default: based on --method)")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    doc_index = load_doc_index(DOC_INDEX_PATH) if args.mode == "retrieval" else {}

    claims_path = Path(args.retrieved_path) if args.retrieved_path else RETRIEVED_PATHS[args.method]
    claims = load_claims(claims_path, args.limit)

    print(f"loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    t0 = time.time()
    results = run_inference(
        claims, doc_index, model, tokenizer,
        mode=args.mode,
        batch_size=args.batch_size,
        max_passages_per_doc=args.max_passages_per_doc,
        max_evidence_chars=args.max_evidence_chars,
        max_input_tokens=args.max_input_tokens,
    )
    elapsed = time.time() - t0
    print(f"done, {elapsed:.1f}s total ({elapsed/len(results):.2f}s/claim)")

    metrics = evaluate(results)
    metrics["model"] = MODEL_ID
    metrics["mode"] = args.mode
    metrics["method"] = args.method if args.mode == "retrieval" else "none"
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["config"] = {
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_passages_per_doc": args.max_passages_per_doc,
        "max_evidence_chars": args.max_evidence_chars,
        "max_input_tokens": args.max_input_tokens,
    }

    tag = f"{args.mode}" + (f"_{args.method}" if args.mode == "retrieval" else "")
    out = RESULTS_DIR / f"fever_{tag}_results.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics to {out}")

    preds_out = RESULTS_DIR / f"fever_{tag}_predictions.json"
    preds_out.write_text(json.dumps(results, indent=2))
    print(f"saved predictions to {preds_out}")


if __name__ == "__main__":
    main()
