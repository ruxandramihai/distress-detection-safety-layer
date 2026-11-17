import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

os.makedirs("results/wordclouds", exist_ok=True)

df = pd.read_csv("results/calibration/test_predictions.csv")

general_fillers = {
    "not", "no", "never", "nothing", "nobody", "want", "make", "people", "things",
    "friend", "feeling", "feel", "way", "say", "tell", "talking", "talk", "years",
    "month", "days", "started", "time", "got", "said", "went", "think", "happened",
    "fucking", "fuckig", "fuckinf", "fuck", "im", "ive", "dont", "just", "like",
    "really", "know", "amp", "x200b"
}

custom_stopwords = set(ENGLISH_STOP_WORDS).union(set(STOPWORDS)).union(general_fillers)

adds_value_mask = (df["true_label"].isin([1, 2, "1", "2", "high", "moderate"])) & (df["pred_tuned"] == 1)
no_value_mask = (df["true_label"].isin([0, "0", "neutral"])) & (df["pred_tuned"] == 1)

adds_value_text = " ".join(df[adds_value_mask]["text"].dropna().astype(str))
no_value_text = " ".join(df[no_value_mask]["text"].dropna().astype(str))

if adds_value_text.strip():
    wc_adds = WordCloud(
        stopwords=custom_stopwords,
        width=1000,
        height=500,
        background_color="white",
        min_word_length=3,
        collocations=False
    ).generate(adds_value_text)
    wc_adds.to_file("results/wordclouds/adds_value_true_positives.png")

if no_value_text.strip():
    wc_no_val = WordCloud(
        stopwords=custom_stopwords,
        width=1000,
        height=500,
        background_color="white",
        min_word_length=3,
        collocations=False
    ).generate(no_value_text)
    wc_no_val.to_file("results/wordclouds/does_not_add_value_false_positives.png")

print("Saved word clouds to results/wordclouds/")
