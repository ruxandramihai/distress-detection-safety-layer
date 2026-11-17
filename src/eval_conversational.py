import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

from distress_detector import DistressDetector, LABELS

LABEL_MAP = {"neutral": 0, "moderate": 1, "high-risk": 2}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="./models/resultsroberta/checkpoint-258")
ap.add_argument("--tokenizer", default="./models/resultsroberta")
ap.add_argument("--config", default="./results/calibration/detector_config.json")
ap.add_argument("--reddit_report", default="./results/calibration/calibration_report.json")
ap.add_argument("--data", default="data/conversational_eval.json")
ap.add_argument("--out_dir", default="./results/conversational_results")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

with open(args.data) as f:
    df = pd.DataFrame(json.load(f)["data"])
y = df["my_label"].map(LABEL_MAP).values

det = DistressDetector(args.model, tokenizer_path=args.tokenizer,
                       config_path=args.config if os.path.exists(args.config) else None)
print(f"Loaded {args.model} | temperature={det.temperature} | threshold={det.high_risk_threshold}")

probs = det.probabilities(df["text"].tolist())
pred_argmax = probs.argmax(axis=1)
pred_tuned = np.array([LABELS.index(det.decide(p)) for p in probs])


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "high_risk_recall": recall_score(y_true == 2, y_pred == 2),
        "high_risk_precision": precision_score(y_true == 2, y_pred == 2, zero_division=0),
        "specificity": float(((y_pred != 2) & (y_true != 2)).sum() / (y_true != 2).sum()),
        "per_class_f1": dict(zip(LABELS, f1_score(y_true, y_pred, average=None, labels=[0, 1, 2]).round(3).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


rng = np.random.default_rng(42)
idx = rng.integers(0, len(y), (2000, len(y)))


def ci(y_true, y_pred):
    mf1 = [f1_score(y_true[i], y_pred[i], average="macro") for i in idx]
    hrr = [recall_score(y_true[i] == 2, y_pred[i] == 2) for i in idx]
    return {"macro_f1_95ci": np.percentile(mf1, [2.5, 97.5]).round(3).tolist(),
            "high_risk_recall_95ci": np.percentile(hrr, [2.5, 97.5]).round(3).tolist()}


report = {
    "n_messages": int(len(y)),
    "mean_words_per_class": df.assign(w=df.text.str.split().str.len()).groupby("my_label").w.mean().round(1).to_dict(),
    "conversational_argmax": {**metrics(y, pred_argmax), **ci(y, pred_argmax)},
    "conversational_tuned": {**metrics(y, pred_tuned), **ci(y, pred_tuned)},
}
if os.path.exists(args.reddit_report):
    with open(args.reddit_report) as f:
        reddit = json.load(f)
    report["reddit_argmax"] = reddit["test_argmax"]
    report["reddit_tuned"] = reddit["test_tuned_threshold"]

with open(os.path.join(args.out_dir, "conversational_report.json"), "w") as f:
    json.dump(report, f, indent=2, default=float)

out = df[["id", "my_label", "original_label_description", "test_idx", "text"]].copy()
out["true_label"] = y
out["p_neutral"], out["p_moderate"], out["p_high"] = probs[:, 0], probs[:, 1], probs[:, 2]
out["pred_argmax"], out["pred_tuned"] = pred_argmax, pred_tuned
out.to_csv(os.path.join(args.out_dir, "conversational_predictions.csv"), index=False)

keys = ["accuracy", "macro_f1", "high_risk_recall", "high_risk_precision", "specificity"]
table = {name: {k: report[name][k] for k in keys}
         for name in ["reddit_argmax", "conversational_argmax", "reddit_tuned", "conversational_tuned"] if name in report}
print("\n", pd.DataFrame(table).round(3).to_string())
print("\nConversational, tuned threshold:\n", classification_report(y, pred_tuned, target_names=LABELS, digits=3))
print("95% CI (tuned): macro-F1", report["conversational_tuned"]["macro_f1_95ci"],
      "| high-risk recall", report["conversational_tuned"]["high_risk_recall_95ci"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, p, title in [(axes[0], pred_argmax, "Argmax"), (axes[1], pred_tuned, "Tuned threshold")]:
    ConfusionMatrixDisplay(confusion_matrix(y, p, labels=[0, 1, 2]), display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Conversational set - {title}")
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "confusion_conversational.png"), dpi=300)
print("\nSaved results to", args.out_dir)
