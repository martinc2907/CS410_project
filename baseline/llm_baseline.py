import json
import time
from pathlib import Path

import pandas as pd
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
DATA_DIR = PROJECT_ROOT / "data" / "news_dataset"
RESULTS_DIR = Path(__file__).parent / "results"
MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
BATCH_SIZE = 2
SEED = 42

SYSTEM_PROMPT = (
    "You are an expert media analyst specializing in detecting misinformation. "
    "Your task is to determine whether a news article is REAL or FAKE. "
    "Analyze the writing style, tone, claims, and sourcing. "
    "Respond with exactly one word: REAL or FAKE."
)

USER_PROMPT_TEMPLATE = """Classify the following news article as REAL or FAKE.

Title: {title}

Article text:
{text}

Your classification (REAL or FAKE):"""


def load_dataset():
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    df = df.dropna(subset=["text", "title"])
    df = df[df["text"].str.strip().str.len() > 50]
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"loaded {len(df)} articles ({(df['label']==0).sum()} fake, {(df['label']==1).sum()} real)")
    return df


def build_messages(title, text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title, text=text)},
    ]


def parse_prediction(raw):
    cleaned = raw.strip().upper()
    for tok in cleaned.split():
        tok = tok.strip(".,!?")
        if tok == "FAKE":
            return 0
        if tok == "REAL":
            return 1
    if "FAKE" in cleaned:
        return 0
    if "REAL" in cleaned:
        return 1
    return None


def run_inference(df, model, tokenizer, batch_size=4):
    results = []
    total = len(df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        prompts = [
            tokenizer.apply_chat_template(
                build_messages(row["title"], row["text"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, row in batch.iterrows()
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
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

        for i, (_, row) in enumerate(batch.iterrows()):
            new_tokens = outputs[i][inputs["input_ids"][i].shape[0]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            results.append({
                "index": row.name,
                "title": row["title"][:100],
                "label": row["label"],
                "prediction": parse_prediction(response),
                "raw_response": response.strip(),
            })

        valid = sum(1 for r in results if r["prediction"] is not None)
        print(f"  {end}/{total}  ({valid} parseable)", flush=True)

    return results


def evaluate(results):
    valid = [r for r in results if r["prediction"] is not None]
    unparseable = len(results) - len(valid)

    labels = [r["label"] for r in valid]
    preds = [r["prediction"] for r in valid]

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", pos_label=0)
    rec = recall_score(labels, preds, average="binary", pos_label=0)
    f1 = f1_score(labels, preds, average="binary", pos_label=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    print(f"\n--- results ({len(valid)}/{len(results)} parsed) ---")
    print(f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['FAKE', 'REAL'], digits=4)}")

    return {
        "accuracy": round(acc, 4),
        "precision_fake": round(prec, 4),
        "recall_fake": round(rec, 4),
        "f1_fake": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(results),
        "valid_predictions": len(valid),
        "unparseable": unparseable,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    df = load_dataset()

    t0 = time.time()
    results = run_inference(df, model, tokenizer, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"done, {elapsed:.1f}s total ({elapsed/len(results):.2f}s/article)")

    metrics = evaluate(results)
    metrics["model"] = MODEL_ID
    metrics["elapsed_seconds"] = round(elapsed, 1)

    out = RESULTS_DIR / "llm_baseline_full_text_results.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics to {out}")

    preds_out = RESULTS_DIR / "llm_baseline_full_text_predictions.json"
    preds_out.write_text(json.dumps(results, indent=2))
    print(f"saved predictions to {preds_out}")


if __name__ == "__main__":
    main()
