import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (precision_recall_curve, average_precision_score, f1_score,
                             recall_score, precision_score, accuracy_score, confusion_matrix)

LABEL_MAP = {"neutral": 0, "moderate": 1, "high-risk": 2}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y):
    def nll(t):
        p = softmax(logits / t)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def pick_threshold(p_high, y, target_recall):
    is_high = (y == 2)
    for t in np.sort(np.unique(p_high))[::-1]:
        if (p_high[is_high] >= t).mean() >= target_recall:
            return float(t)
    return float(p_high.min())


def predict_with_threshold(probs, threshold):
    pred = probs[:, :2].argmax(axis=1)
    if threshold is None:
        return probs.argmax(axis=1)
    pred[probs[:, 2] >= threshold] = 2
    return pred


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "high_risk_recall": recall_score(y == 2, pred == 2),
        "high_risk_precision": precision_score(y == 2, pred == 2, zero_division=0),
        "specificity": float(((pred != 2) & (y != 2)).sum() / (y != 2).sum()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def load_split(data_dir, name):
    with open(os.path.join(data_dir, f"{name}_data.json")) as f:
        df = pd.DataFrame(json.load(f)["data"])
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df, df["my_label"].map(LABEL_MAP).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/resultsroberta/checkpoint-258")
    ap.add_argument("--tokenizer", default="./models/resultsroberta")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="./results/calibration")
    ap.add_argument("--target_recall", type=float, default=0.98)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from distress_detector import DistressDetector
    det = DistressDetector(args.model, tokenizer_path=args.tokenizer)

    val_df, y_val = load_split(args.data_dir, "val")
    test_df, y_test = load_split(args.data_dir, "test")
    z_val = det.logits(val_df["text"].tolist())
    z_test = det.logits(test_df["text"].tolist())

    for df, z, y, name in [(val_df, z_val, y_val, "val"), (test_df, z_test, y_test, "test")]:
        out = pd.DataFrame({"text": df["text"], "true_label": y,
                            "logit_neutral": z[:, 0], "logit_moderate": z[:, 1], "logit_high": z[:, 2]})
        out.to_csv(os.path.join(args.out_dir, f"{name}_logits.csv"), index=False)

    T = fit_temperature(z_val, y_val)
    p_val_raw, p_val = softmax(z_val), softmax(z_val / T)
    p_test_raw, p_test = softmax(z_test), softmax(z_test / T)
    thr = pick_threshold(p_val[:, 2], y_val, args.target_recall)

    pd.DataFrame({"text": test_df["text"], "true_label": y_test,
                  "p_neutral": p_test[:, 0], "p_moderate": p_test[:, 1], "p_high": p_test[:, 2],
                  "pred_argmax": predict_with_threshold(p_test, None),
                  "pred_tuned": predict_with_threshold(p_test, thr)}).to_csv(
        os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    report = {
        "temperature": T,
        "target_recall_on_val": args.target_recall,
        "high_risk_threshold": thr,
        "ece_test_before": ece(p_test_raw, y_test),
        "ece_test_after": ece(p_test, y_test),
        "ece_val_before": ece(p_val_raw, y_val),
        "ece_val_after": ece(p_val, y_val),
        "pr_auc_high_risk_test": float(average_precision_score(y_test == 2, p_test[:, 2])),
        "test_argmax": metrics(y_test, predict_with_threshold(p_test, None)),
        "test_tuned_threshold": metrics(y_test, predict_with_threshold(p_test, thr)),
    }
    with open(os.path.join(args.out_dir, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, "detector_config.json"), "w") as f:
        json.dump({"temperature": T, "high_risk_threshold": thr}, f, indent=2)

    import matplotlib.pyplot as plt
    prec, rec, _ = precision_recall_curve(y_test == 2, p_test[:, 2])
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall (high-risk)"); plt.ylabel("Precision (high-risk)")
    plt.title(f"High-risk PR curve, test (AP = {report['pr_auc_high_risk_test']:.3f})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pr_curve_high_risk.png"), dpi=300)

    plt.figure(figsize=(5, 4))
    for p, lab in [(p_test_raw, "before"), (p_test, f"after (T={T:.2f})")]:
        conf, corr = p.max(axis=1), (p.argmax(axis=1) == y_test)
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [corr[idx == b].mean() for b in range(10) if (idx == b).any()]
        plt.plot(xs, ys, marker="o", label=lab)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Reliability diagram, test")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "reliability_before_after.png"), dpi=300)

    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    print("argmax:", {k: v for k, v in report["test_argmax"].items() if k != "confusion_matrix"})
    print("tuned :", {k: v for k, v in report["test_tuned_threshold"].items() if k != "confusion_matrix"})


if __name__ == "__main__":
    main()
