import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="Calibri")

tau = json.load(open("results/calibration/detector_config.json"))["high_risk_threshold"]
sweep = pd.read_csv("results/analysis/threshold_sweep.csv")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for name, d in sweep.groupby("set"):
    ax[0].plot(d["tau"], d["hr_recall"], label=f"{name} recall")
    ax[0].plot(d["tau"], d["false_alarm_rate"], "--", label=f"{name} false-alarm rate")
    ax[1].plot(d["false_alarm_rate"], d["hr_recall"], marker=".", label=name)
ax[0].axvline(tau, color="grey", lw=0.8)
ax[0].axvline(0.5, color="grey", lw=0.8, ls=":")
ax[0].set_xscale("log")
ax[0].set_xlabel("high-risk threshold τ")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("false-alarm rate (1 - specificity)")
ax[1].set_ylabel("high-risk recall")
ax[1].legend(fontsize=9)
plt.tight_layout()
fig.savefig("results/analysis/figure9_threshold_sweep.png", dpi=300, facecolor="white")
