import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score

OUT = "results/analysis"
os.makedirs(OUT, exist_ok=True)
LABEL_MAP = {"neutral": 0, "moderate": 1, "high-risk": 2}
rng = np.random.default_rng(42)
N_BOOT = 2000

cfg = json.load(open("results/calibration/detector_config.json"))
T, TAU = cfg["temperature"], cfg["high_risk_threshold"]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decide(p, tau=None):
    if tau is None:
        return p.argmax(axis=1)
    pred = p[:, :2].argmax(axis=1)
    pred[p[:, 2] >= tau] = 2
    return pred


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "neutral_recall": recall_score(y == 0, pred == 0, zero_division=0),
        "hr_recall": recall_score(y == 2, pred == 2, zero_division=0),
        "hr_precision": precision_score(y == 2, pred == 2, zero_division=0),
        "specificity": ((pred != 2) & (y != 2)).sum() / max((y != 2).sum(), 1),
    }


def with_ci(y, pred):
    point = metrics(y, pred)
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        for k, v in metrics(y[i], pred[i]).items():
            boot[k].append(v)
    return {k: f"{point[k]:.2f} [{np.percentile(boot[k], 2.5):.2f}, {np.percentile(boot[k], 97.5):.2f}]" for k in point}


logits = pd.read_csv("results/calibration/test_logits.csv")
reddit = pd.DataFrame(json.load(open("data/test_data.json"))["data"]).dropna(subset=["text"])
logits = logits.merge(reddit[["text", "original_label_description"]].drop_duplicates("text"), on="text", how="left")
y_r = logits["true_label"].values
p_r = softmax(logits[["logit_neutral", "logit_moderate", "logit_high"]].values / T)
words_r = logits["text"].str.split().str.len().values

conv = pd.read_csv("results/conversational_results/conversational_predictions.csv")
y_c = conv["true_label"].values
p_c = conv[["p_neutral", "p_moderate", "p_high"]].values

vig = pd.read_csv("results/robustness/vignette_predictions.csv")
y_v = vig["expected_level"].map(LABEL_MAP).values
pred_v = vig["pred_message_only"].map(LABEL_MAP).values

rows = []
for name, y, p in [("reddit_test", y_r, p_r), ("conversational", y_c, p_c)]:
    for rule, tau in [("argmax", None), ("tuned", TAU)]:
        rows.append({"set": name, "rule": rule, "n": len(y), **with_ci(y, decide(p, tau))})
rows.append({"set": "vignettes", "rule": "tuned", "n": len(y_v), **with_ci(y_v, pred_v)})
ci = pd.DataFrame(rows)
ci.to_csv(f"{OUT}/bootstrap_ci.csv", index=False)
print("\n1. Point estimate [95% bootstrap CI]\n", ci.to_string(index=False))

print("\nAUROC high-risk vs rest: Reddit %.3f | conversational %.3f"
      % (roc_auc_score(y_r == 2, p_r[:, 2]), roc_auc_score(y_c == 2, p_c[:, 2])))

taus = np.concatenate([np.linspace(0.005, 0.1, 20), np.linspace(0.1, 0.9, 17)])
sweep = []
for name, y, p in [("reddit_test", y_r, p_r), ("conversational", y_c, p_c)]:
    for t in taus:
        pred = decide(p, t)
        m = metrics(y, pred)
        sweep.append({"set": name, "tau": t, "hr_recall": m["hr_recall"],
                      "false_alarm_rate": 1 - m["specificity"], "macro_f1": m["macro_f1"]})
sweep = pd.DataFrame(sweep)
sweep.to_csv(f"{OUT}/threshold_sweep.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for name, d in sweep.groupby("set"):
    ax[0].plot(d["tau"], d["hr_recall"], label=f"{name} recall")
    ax[0].plot(d["tau"], d["false_alarm_rate"], "--", label=f"{name} false-alarm rate")
    ax[1].plot(d["false_alarm_rate"], d["hr_recall"], marker=".", label=name)
for a in ax[:1]:
    a.axvline(TAU, color="grey", lw=0.8)
    a.axvline(0.5, color="grey", lw=0.8, ls=":")
ax[0].set_xscale("log"); ax[0].set_xlabel("high-risk threshold τ"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("false-alarm rate (1 - specificity)"); ax[1].set_ylabel("high-risk recall"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/threshold_sweep.png", dpi=200)

lo, hi = np.percentile(words_r[y_r != 2], [20, 80])
mask = (words_r >= lo) & (words_r <= hi)
sub = pd.DataFrame({"y": y_r[mask], "w": words_r[mask], "i": np.where(mask)[0]})
n = sub["y"].value_counts().min()
sub = pd.concat([sub[sub["y"] == k].sample(n, random_state=42) for k in range(3)])
i = sub["i"].values
print(f"\n3. Length-matched subset: {lo:.0f}-{hi:.0f} words, {n} posts per class, "
      f"mean words per class {sub.groupby('y')['w'].mean().round(1).to_dict()}")
lm = pd.DataFrame([{"rule": r, **with_ci(y_r[i], decide(p_r[i], t))} for r, t in [("argmax", None), ("tuned", TAU)]])
print(lm.to_string(index=False))
lm.to_csv(f"{OUT}/length_matched_reddit.csv", index=False)

logits["pred_argmax"] = decide(p_r)
logits["pred_tuned"] = decide(p_r, TAU)
by_src = (logits.assign(correct_argmax=logits.pred_argmax == logits.true_label,
                        correct_tuned=logits.pred_tuned == logits.true_label,
                        flagged_high=logits.pred_tuned == 2)
          .groupby("original_label_description")[["correct_argmax", "correct_tuned", "flagged_high"]]
          .agg(["mean", "size"]))
by_src.to_csv(f"{OUT}/errors_by_source.csv")
print("\n4. By source community\n", by_src.round(2).to_string())
