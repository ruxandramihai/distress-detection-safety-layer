import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from distress_detector import DistressDetector, LABELS

LABEL_MAP = {"neutral": 0, "moderate": 1, "high-risk": 2}


def first_n_words(text, n):
    return " ".join(str(text).split()[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/resultsroberta/checkpoint-258")
    ap.add_argument("--tokenizer", default="./models/resultsroberta")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vignettes", default="data/vignettes.csv")
    ap.add_argument("--out_dir", default="./results/robustness")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    det = DistressDetector(args.model, tokenizer_path=args.tokenizer, config_path=args.config)

    with open(os.path.join(args.data_dir, "test_data.json")) as f:
        test = pd.DataFrame(json.load(f)["data"]).dropna(subset=["text"])
    y = test["my_label"].map(LABEL_MAP).values

    rows = []
    for n in [10, 20, 40, 80, None]:
        texts = [first_n_words(t, n) if n else t for t in test["text"]]
        probs = det.probabilities(texts)
        pred = np.array([LABELS.index(det.decide(p)) for p in probs])
        rows.append({"first_n_words": n or "full",
                     "accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"),
                     "high_risk_recall": recall_score(y == 2, pred == 2),
                     "high_risk_precision": precision_score(y == 2, pred == 2, zero_division=0)})
    trunc = pd.DataFrame(rows)
    trunc.to_csv(os.path.join(args.out_dir, "truncation_results.csv"), index=False)
    print("\n(a) Truncated test posts\n", trunc.round(3).to_string(index=False))

    vig = pd.read_csv(args.vignettes).sort_values(["conversation_id", "turn"])
    out = []
    for cid, conv in vig.groupby("conversation_id", sort=False):
        history = []
        for _, r in conv.iterrows():
            alone = det.assess(r["text"])
            ctx = det.assess(r["text"], history=history)
            out.append({**r.to_dict(), "pred_message_only": alone["label"], "pred_with_context": ctx["label"],
                        "p_high_message_only": alone["probabilities"]["high-risk"],
                        "latency_ms": alone["latency_ms"]})
            history.append(r["text"])
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(args.out_dir, "vignette_predictions.csv"), index=False)

    res["correct_message_only"] = res["pred_message_only"] == res["expected_level"]
    res["correct_with_context"] = res["pred_with_context"] == res["expected_level"]
    by_cat = res.groupby("category")[["correct_message_only", "correct_with_context"]].mean()
    by_cat["n"] = res.groupby("category").size()
    by_cat.to_csv(os.path.join(args.out_dir, "vignette_accuracy_by_category.csv"))
    print("\n(b) Vignette accuracy by category\n", by_cat.round(3).to_string())

    hr = res["expected_level"] == "high-risk"
    print("\nHigh-risk vignette recall: message-only %.3f, with context %.3f" %
          ((res.loc[hr, "pred_message_only"] == "high-risk").mean(),
           (res.loc[hr, "pred_with_context"] == "high-risk").mean()))
    print("Single-message latency: mean %.1f ms, p95 %.1f ms" %
          (res["latency_ms"].mean(), np.percentile(res["latency_ms"], 95)))


if __name__ == "__main__":
    main()
