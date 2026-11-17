import os
import sys
import subprocess

MODEL = "./models/resultsroberta/checkpoint-258"
TOKENIZER = "./models/resultsroberta"
STATE = "./models/resultsroberta/checkpoint-516/trainer_state.json"
CONFIG = "./results/calibration/detector_config.json"

for path in [MODEL, TOKENIZER, "train_data.json", "val_data.json", "test_data.json", "data/vignettes.csv"]:
    if not os.path.exists(path):
        sys.exit(f"Missing: {path}")

steps = [
    ["src/plot_training_history.py", "--state", STATE],
    ["src/calibrate_and_threshold.py", "--model", MODEL, "--tokenizer", TOKENIZER, "--target_recall", "0.98"],
    ["src/robustness_eval.py", "--model", MODEL, "--tokenizer", TOKENIZER, "--config", CONFIG],
    ["src/with_without_eval.py", "--model", MODEL, "--tokenizer", TOKENIZER, "--config", CONFIG, "--host", "stub"],
]
if "--openai" in sys.argv:
    steps.append(["src/with_without_eval.py", "--model", MODEL, "--tokenizer", TOKENIZER, "--config", CONFIG, "--host", "openai"])
steps.append(["src/wordclouds.py", "--pred", "results/calibration/test_predictions.csv", "--pred_col", "pred_tuned"])

for k, step in enumerate(steps, 1):
    if step[0] == "src/plot_training_history.py" and not os.path.exists(STATE):
        print(f"\n[{k}/{len(steps)}] skipped {step[0]} (no {STATE})")
        continue
    print(f"\n[{k}/{len(steps)}] python {' '.join(step)}")
    result = subprocess.run([sys.executable] + step)
    if result.returncode != 0:
        sys.exit(f"Step failed: {step[0]}")

print("\nDone. Outputs saved in results/")
