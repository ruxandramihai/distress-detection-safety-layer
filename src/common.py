import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

LABELS = ["neutral", "moderate", "high-risk"]
LABEL_MAP = {"neutral": 0, "moderate": 1, "high-risk": 2}


def load_json(path):
    with open(path) as f:
        df = pd.DataFrame(json.load(f)["data"])
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["label"] = df["my_label"].map(LABEL_MAP)
    return df


def load_vignettes(path="data/vignettes.csv"):
    v = pd.read_csv(path)
    v["label"] = v["expected_level"].map(LABEL_MAP)
    return v


def load_test_sets():
    return {"reddit_test": load_json("data/test_data.json"),
            "conversational": load_json("data/conversational_eval.json"),
            "vignettes": load_vignettes()}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decide(probs, threshold=None):
    if threshold is None:
        return probs.argmax(axis=1)
    pred = probs[:, :2].argmax(axis=1)
    pred[probs[:, 2] >= threshold] = 2
    return pred


def pick_threshold(p_high, y, target=0.98):
    for t in np.sort(np.unique(p_high))[::-1]:
        if (p_high[y == 2] >= t).mean() >= target:
            return float(t)
    return float(p_high.min())


def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "neutral_recall": recall_score(y == 0, pred == 0, zero_division=0),
        "hr_recall": recall_score(y == 2, pred == 2, zero_division=0),
        "hr_precision": precision_score(y == 2, pred == 2, zero_division=0),
        "specificity": float(((pred != 2) & (y != 2)).sum() / max((y != 2).sum(), 1)),
        "confusion_matrix": json.dumps(confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()),
    }


def bootstrap_ci(y, pred, metric, n_boot=2000, seed=42):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(scores(y[i], pred[i])[metric])
    return np.percentile(vals, [2.5, 97.5])
