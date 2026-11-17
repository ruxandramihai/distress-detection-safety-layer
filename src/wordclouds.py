import os
import re
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

KEEP = {"i", "me", "my", "myself", "not", "no", "never", "nothing", "always", "all", "everything", "nobody", "anyone"}
STOP = set(ENGLISH_STOP_WORDS) - KEEP | {"fucking","im", "ive", "dont", "just", "like", "really", "know", "amp", "x200b"}


def tokens(text):
    return [w for w in re.findall(r"[a-z']+", str(text).lower().replace("’", "'"))
            if len(w) > 2 and w.replace("'", "") not in STOP]


def log_odds(texts_a, texts_b, min_count=5):
    ca, cb = Counter(), Counter()
    for t in texts_a:
        ca.update(tokens(t))
    for t in texts_b:
        cb.update(tokens(t))
    prior = ca + cb
    a0, na, nb = sum(prior.values()), sum(ca.values()), sum(cb.values())
    rows = []
    for w, p in prior.items():
        if p < min_count:
            continue
        ya, yb = ca[w], cb[w]
        la = np.log((ya + p) / (na + a0 - ya - p))
        lb = np.log((yb + p) / (nb + a0 - yb - p))
        var = 1 / (ya + p) + 1 / (yb + p)
        rows.append((w, (la - lb) / np.sqrt(var), ya, yb))
    return pd.DataFrame(rows, columns=["word", "z", "count_a", "count_b"]).sort_values("z", ascending=False)


def two_clouds(df_z, name_a, name_b, title, path, top=60):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sub, name, cmap in [(axes[0], df_z[df_z.z > 0].head(top), name_a, "Greens"),
                                (axes[1], df_z[df_z.z < 0].assign(z=lambda d: -d.z).sort_values("z", ascending=False).head(top), name_b, "Reds")]:
        if len(sub) == 0:
            ax.axis("off"); continue
        colour = "#1b7837" if cmap == "Greens" else "#b2182b"
        wc = WordCloud(width=700, height=500, background_color="white", color_func=lambda *a, **k: colour,
                       prefer_horizontal=0.9, random_state=42)
        wc.generate_from_frequencies(dict(zip(sub.word, sub.z)))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(name, fontsize=13)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/roberta_predictions.csv")
    ap.add_argument("--pred_col", default="predicted_label_roberta")
    ap.add_argument("--out_dir", default="./results/wordclouds")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred)
    y, p = df["true_label"].values, df[args.pred_col].values
    all_terms = []

    z = log_odds(df.text[y == p], df.text[y != p])
    two_clouds(z, f"Correctly classified (n={(y == p).sum()})", f"Misclassified (n={(y != p).sum()})",
               "Words distinctive of correctly vs incorrectly classified test posts",
               os.path.join(args.out_dir, "wc_correct_vs_errors.png"))
    all_terms.append(z.assign(comparison="correct_vs_errors"))

    a, b = (y == 0) & (p == 0), (y == 0) & (p == 1)
    z = log_odds(df.text[a], df.text[b], min_count=3)
    two_clouds(z, f"Neutral, correct (n={a.sum()})", f"Neutral predicted moderate (n={b.sum()})",
               "Neutral posts: correctly classified vs confused with moderate",
               os.path.join(args.out_dir, "wc_neutral_correct_vs_moderate.png"))
    all_terms.append(z.assign(comparison="neutral_correct_vs_moderate"))

    a, b = (y == 2) & (p == 2), (y == 2) & (p != 2)
    if b.sum() >= 10:
        z = log_odds(df.text[a], df.text[b], min_count=3)
        two_clouds(z, f"High-risk, detected (n={a.sum()})", f"High-risk, missed (n={b.sum()})",
                   "High-risk posts: detected vs missed",
                   os.path.join(args.out_dir, "wc_highrisk_flagged_vs_missed.png"))
        all_terms.append(z.assign(comparison="highrisk_detected_vs_missed"))

    terms = pd.concat(all_terms)
    terms.to_csv(os.path.join(args.out_dir, "distinctive_terms.csv"), index=False)
    for comp, d in terms.groupby("comparison"):
        print(f"\n{comp}\n  group A:", ", ".join(d.head(12).word),
              "\n  group B:", ", ".join(d.tail(12).word[::-1]))


if __name__ == "__main__":
    main()
