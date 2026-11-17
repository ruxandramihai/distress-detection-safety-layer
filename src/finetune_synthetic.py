import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          EarlyStoppingCallback, DataCollatorWithPadding)

LABEL_MAP = {"neutral": 0, "moderate": 1, "high-risk": 2}
LABELS = ["neutral", "moderate", "high-risk"]

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="name for this model/condition, used in outputs")
ap.add_argument("--condition", choices=["synthetic", "combined", "reddit"], default="combined")
ap.add_argument("--base_model", default="roberta-base")
ap.add_argument("--synth_repeat", type=int, default=1, help="repeat synthetic training examples N times (combined only)")
ap.add_argument("--eval_only", action="store_true")
ap.add_argument("--model_path", default=None, help="checkpoint to evaluate with --eval_only")
ap.add_argument("--tokenizer_path", default=None)
ap.add_argument("--max_length", type=int, default=None, help="default: 128 for synthetic, 512 otherwise")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fp16", action="store_true", help="mixed precision on GPU (fine for BERT/RoBERTa, avoid for DeBERTa-v3)")
ap.add_argument("--target_recall", type=float, default=0.98)
args = ap.parse_args()

out_dir = os.path.join("models", "runs", args.name)
os.makedirs(out_dir, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def load_json(path):
    with open(path) as f:
        df = pd.DataFrame(json.load(f)["data"])
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["label"] = df["my_label"].map(LABEL_MAP)
    return df[["text", "label"]]


reddit_train, reddit_val = load_json("data/train_data.json"), load_json("data/val_data.json")
synth_train, synth_val = load_json("data/synthetic_train.json"), load_json("data/synthetic_val.json")

if args.condition == "reddit":
    train_df, val_df = reddit_train, reddit_val
elif args.condition == "synthetic":
    train_df, val_df = synth_train, synth_val
else:
    train_df = pd.concat([reddit_train] + [synth_train] * args.synth_repeat, ignore_index=True)
    val_df = pd.concat([reddit_val, synth_val], ignore_index=True)

max_length = args.max_length or (128 if args.condition == "synthetic" else 512)

if args.eval_only:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3,
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=LABEL_MAP)


def to_dataset(df):
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, pred), "f1": f1_score(labels, pred, average="weighted")}


small = args.condition == "synthetic"
training_args = TrainingArguments(
    output_dir=out_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1 if small else 4,
    num_train_epochs=10 if small else 5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    seed=args.seed,
    fp16=args.fp16 and torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None if args.eval_only else to_dataset(train_df),
    eval_dataset=to_dataset(val_df),
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3 if small else 2, early_stopping_threshold=0.001)],
)

if not args.eval_only:
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "best_model"))
    with open(os.path.join(out_dir, "trainer_log.json"), "w") as f:
        json.dump({"best_checkpoint": trainer.state.best_model_checkpoint,
                   "log_history": trainer.state.log_history}, f, indent=2)


def logits_for(texts):
    ds = to_dataset(pd.DataFrame({"text": texts, "label": 0}))
    return trainer.predict(ds).predictions


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(z, y):
    nll = lambda t: -np.mean(np.log(softmax(z / t)[np.arange(len(y)), y] + 1e-12))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def pick_threshold(p_high, y, target):
    for t in np.sort(np.unique(p_high))[::-1]:
        if (p_high[y == 2] >= t).mean() >= target:
            return float(t)
    return float(p_high.min())


def predict(probs, threshold=None):
    if threshold is None:
        return probs.argmax(axis=1)
    pred = probs[:, :2].argmax(axis=1)
    pred[probs[:, 2] >= threshold] = 2
    return pred


def scores(y, pred):
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "neutral_recall": round(recall_score(y == 0, pred == 0), 4),
        "high_risk_recall": round(recall_score(y == 2, pred == 2), 4),
        "high_risk_precision": round(precision_score(y == 2, pred == 2, zero_division=0), 4),
        "specificity": round(float(((pred != 2) & (y != 2)).sum() / (y != 2).sum()), 4),
        "confusion_matrix": json.dumps(confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()),
    }


z_val = logits_for(val_df["text"].tolist())
y_val = val_df["label"].values
T = fit_temperature(z_val, y_val)
thr = pick_threshold(softmax(z_val / T)[:, 2], y_val, args.target_recall)
with open(os.path.join(out_dir, "detector_config.json"), "w") as f:
    json.dump({"temperature": T, "high_risk_threshold": thr}, f, indent=2)
print(f"\nTemperature {T:.3f} | high-risk threshold {thr:.4f} (validation recall >= {args.target_recall})")

test_sets = {"reddit_test": load_json("data/test_data.json"), "conversational": load_json("data/conversational_eval.json")}
if os.path.exists("data/vignettes.csv"):
    v = pd.read_csv("data/vignettes.csv")
    test_sets["vignettes"] = pd.DataFrame({"text": v["text"], "label": v["expected_level"].map(LABEL_MAP)})

rows = []
for set_name, df in test_sets.items():
    y = df["label"].values
    probs = softmax(logits_for(df["text"].tolist()) / T)
    pred_arg, pred_thr = predict(probs), predict(probs, thr)
    pd.DataFrame({"text": df["text"], "true_label": y, "p_neutral": probs[:, 0], "p_moderate": probs[:, 1],
                  "p_high": probs[:, 2], "pred_argmax": pred_arg, "pred_tuned": pred_thr}
                 ).to_csv(os.path.join(out_dir, f"predictions_{set_name}.csv"), index=False)
    for rule, pred in [("argmax", pred_arg), ("tuned", pred_thr)]:
        rows.append({"name": args.name, "condition": "eval_only" if args.eval_only else args.condition,
                     "base_model": args.model_path if args.eval_only else args.base_model,
                     "synth_repeat": args.synth_repeat, "test_set": set_name, "rule": rule,
                     "threshold": round(thr, 4) if rule == "tuned" else None, **scores(y, pred)})

res = pd.DataFrame(rows)
res.to_csv("results/results_summary.csv", mode="a", header=not os.path.exists("results/results_summary.csv"), index=False)
print("\n", res.drop(columns=["confusion_matrix", "base_model", "synth_repeat"]).to_string(index=False))
print(f"\nSaved predictions to {out_dir}/ and appended to results_summary.csv")
