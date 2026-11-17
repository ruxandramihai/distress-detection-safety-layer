import argparse
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

PROMPTS = {
    "train": (
        "Rewrite the following Reddit post as a single message that the author could send to a mental health support "
        "chatbot. Write in the first person, between 10 and 36 words. Vary the style: it may be informal, start with a "
        "greeting, or be phrased as a question. Keep the main concern and the level of distress of the original post. "
        "Do not include any details of self-harm methods. If the post is about another person, is not a personal post, "
        "or its content does not match the label '{label}', reply only with SKIP.\n\nPost:\n{text}"
    ),
    "test": (
        "Rewrite the following Reddit post as a single first-person message of one to three sentences that the author "
        "could send to a mental health support chatbot. Keep the main concern and the level of distress of the original "
        "post. Do not include any details of self-harm methods. If the post is about another person, is not a personal "
        "post, or its content does not match the label '{label}', reply only with SKIP.\n\nPost:\n{text}"
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train", "test"], default="train")
ap.add_argument("--per_class", type=int, default=100)
ap.add_argument("--min_words", type=int, default=20)
ap.add_argument("--model", default="gemini-2.5-flash")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=None)
args = ap.parse_args()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

df = pd.DataFrame(json.load(open(f"data/{args.split}_data.json"))["data"]).dropna(subset=["text"])
df["idx"] = df.index
df = df[df["text"].str.split().str.len() >= args.min_words]


def rewrite(text, label):
    prompt = PROMPTS[args.split].format(text=text[:4000], label=label)
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=args.model, contents=prompt,
                config=types.GenerateContentConfig(temperature=1.0))
            return (resp.text or "").strip()
        except Exception as e:
            print(f"retry {attempt + 1}: {str(e)[:80]}")
            time.sleep(2 ** attempt * 5)
    return ""


rows = []
for label, group in df.groupby("my_label"):
    shares = group["original_label_description"].value_counts(normalize=True)
    order = pd.concat([
        group[group["original_label_description"] == src].sample(frac=1, random_state=args.seed)
        for src in shares.index])
    quota = (shares * args.per_class).round().astype(int).to_dict()
    taken = {src: 0 for src in quota}
    for _, r in order.iterrows():
        src = r["original_label_description"]
        if taken[src] >= quota[src]:
            continue
        msg = rewrite(r["text"], label)
        if not msg or msg.upper().startswith("SKIP"):
            continue
        rows.append({"text": msg, "my_label": label, "source": "synthetic_gemini",
                     "original_label_description": src, f"{args.split}_idx": int(r["idx"]),
                     "original_text": r["text"]})
        taken[src] += 1
        if sum(taken.values()) >= args.per_class:
            break
    print(label, sum(taken.values()))

out = args.out or f"data/synthetic_{args.split}_gemini.json"
json.dump({"generator": args.model, "split": args.split, "data": rows}, open(out, "w"), indent=1, ensure_ascii=False)
print(f"saved {len(rows)} messages to {out}")
