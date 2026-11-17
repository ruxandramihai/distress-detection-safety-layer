import json
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="./models/resultsroberta/checkpoint-516/trainer_state.json")
ap.add_argument("--out", default="results/training_history.png")
args = ap.parse_args()

state = json.load(open(args.state))
train = [h for h in state["log_history"] if "loss" in h]
evals = [h for h in state["log_history"] if "eval_loss" in h]
best = state["best_global_step"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot([h["epoch"] for h in train], [h["loss"] for h in train], label="Training loss", alpha=0.8)
axes[0].plot([h["epoch"] for h in evals], [h["eval_loss"] for h in evals], "o-", label="Validation loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot([h["epoch"] for h in evals], [h["eval_f1"] for h in evals], "o-", label="Validation weighted F1")
axes[1].plot([h["epoch"] for h in evals], [h["eval_accuracy"] for h in evals], "s--", label="Validation accuracy")
best_epoch = [h["epoch"] for h in evals if h["step"] == best][0]
axes[1].axvline(best_epoch, color="grey", ls=":", label=f"Selected checkpoint (step {best})")
axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0.8, 0.9); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(args.out, dpi=300)
for h in evals:
    print(f"epoch {h['epoch']:.0f}: val loss {h['eval_loss']:.3f}, val weighted F1 {h['eval_f1']:.4f}, val acc {h['eval_accuracy']:.4f}")
print("best step:", best, "->", state["best_model_checkpoint"])
