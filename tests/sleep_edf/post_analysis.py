import json
import numpy as np

with open("experiments/figs/sleep_edf/sleep_edf_classification_sig_N3.json", "r") as f:
    data = json.load(f)

for name, vals in data["methods"].items():
    acc = np.array(vals["accuracy_per_record"], dtype=float)
    mf1 = np.array(vals["macro_f1_per_record"], dtype=float)
    wf1 = np.array(vals["weighted_f1_per_record"], dtype=float)

    acc_se = acc.std(ddof=1) / np.sqrt(len(acc))
    mf1_se = mf1.std(ddof=1) / np.sqrt(len(mf1))
    wf1_se = wf1.std(ddof=1) / np.sqrt(len(wf1))

    print(
        f"{name:10s} | "
        f"acc mean={acc.mean():.3f} (se={acc_se:.3f}) | "
        f"macro-F1 mean={mf1.mean():.3f} (se={mf1_se:.3f}) | "
        f"weighted-F1 mean={wf1.mean():.3f} (se={wf1_se:.3f})"
    )