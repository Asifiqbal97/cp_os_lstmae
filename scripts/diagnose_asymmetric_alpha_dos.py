"""Tests Option B from the DDoS/DoS confusion analysis: does giving DoS a
STRICTER (larger) per-class alpha in the Mondrian gate reduce DDoS's
Path-B escape rate, and at what cost to genuine DoS traffic's fast/
confident exit rate?

Mechanism: Mondrian conformal inclusion test is (p-value > alpha_class).
Larger alpha_DoS -> harder for DoS to be included in ANY prediction set ->
fewer DDoS rows get {DoS} as their sole (wrongly confident) class -> more
DDoS rows correctly deferred. Cost: genuine DoS rows also lose DoS from
their own set more often, so fewer of them get the fast singleton exit.

Cheap and read-only: the DDoS-held-out rotation's model already has DoS as
a KNOWN class (only DDoS is excluded), so both the gain (DDoS) and the cost
(DoS) can be measured from that ONE rotation's already-trained artifacts --
no retraining, no config change, no other rotation needed. P-values are
precomputed once; the alpha sweep is just threshold comparisons afterward.

Writes to a NEW isolated folder (artifacts/diagnostics/asymmetric_alpha_dos/)
-- does not touch summary.md, result.joblib, configs/mvp.yaml, or any
trained artifact.

Usage:
    python -m scripts.diagnose_asymmetric_alpha_dos --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.loader import load_csv
from src.data.zero_day_split import make_zero_day_split
from src.analysis import load_rotation_artifacts

OUT_DIR = Path("artifacts") / "diagnostics" / "asymmetric_alpha_dos"
ALPHA_DOS_GRID = np.concatenate([np.array([0.05]), np.arange(0.10, 0.95, 0.05)])


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    alpha1 = cfg["alpha_stage1"]

    print("Loading data (read-only)...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])

    print("Loading DDoS rotation's Gate 1 artifacts (read-only, no retraining)...")
    booster, label_encoder, nonconf_by_class, _ = load_rotation_artifacts("DDoS")
    classes = list(label_encoder.classes_)
    if "DoS" not in classes:
        print("ERROR: DoS is not a known class in the DDoS rotation -- cannot proceed.")
        return
    dos_idx = classes.index("DoS")

    print("Rebuilding deterministic split (same seed)...")
    split = make_zero_day_split(train_pool, test_pool, "DDoS", random_state=cfg["random_state"])
    proba = booster.predict(split.X_te_s, num_iteration=booster.best_iteration)

    print("Precomputing per-class p-values (once, sweep reuses these)...")
    n_rows, n_classes = proba.shape
    pvals = np.zeros((n_rows, n_classes))
    for j, cls in enumerate(classes):
        scores = 1.0 - proba[:, j]
        calib_sorted = nonconf_by_class[cls]
        n_ge = len(calib_sorted) - np.searchsorted(calib_sorted, scores, side="left")
        pvals[:, j] = (n_ge + 1) / (len(calib_sorted) + 1)

    ddos_mask = (split.y_te == "DDoS")
    dos_mask = (split.y_te == "DoS")
    print(f"DDoS zero-day rows: {ddos_mask.sum()}   Genuine DoS rows: {dos_mask.sum()}")

    rows = []
    for alpha_dos in ALPHA_DOS_GRID:
        alpha_arr = np.full(n_classes, alpha1)
        alpha_arr[dos_idx] = alpha_dos
        included = pvals > alpha_arr[None, :]
        set_sizes = included.sum(axis=1)

        # Gain: fraction of DDoS zero-day rows now correctly deferred
        # (NOT a singleton -- i.e. no longer wrongly confident)
        ddos_deferral_rate = float((set_sizes[ddos_mask] != 1).mean())

        # Cost: fraction of GENUINE DoS rows that still get DoS as a
        # confident singleton exit (their own correct, fast path)
        dos_singleton_correct = (set_sizes == 1) & included[:, dos_idx]
        dos_fast_exit_rate = float(dos_singleton_correct[dos_mask].mean())

        rows.append({"alpha_dos": float(alpha_dos), "ddos_deferral_rate": ddos_deferral_rate,
                     "dos_fast_exit_rate": dos_fast_exit_rate})
        print(f"  alpha_DoS={alpha_dos:.2f}  DDoS deferral (gain)={ddos_deferral_rate:.4f}  "
              f"DoS fast-exit (cost, want high)={dos_fast_exit_rate:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alphas = [r["alpha_dos"] for r in rows]
    gains = [r["ddos_deferral_rate"] for r in rows]
    costs = [r["dos_fast_exit_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, gains, marker="o", label="DDoS deferral rate (gain -- want high)", color="#55A868")
    ax.plot(alphas, costs, marker="s", label="DoS fast-exit rate (cost -- want to stay high)", color="#C44E52")
    ax.axvline(alpha1, color="black", linestyle="--", linewidth=1.5, label=f"current alpha={alpha1}")
    ax.set_xlabel("alpha_DoS (per-class Mondrian alpha for DoS only, all other classes unchanged)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Asymmetric alpha_DoS: DDoS-recall gain vs DoS-fast-exit cost\n"
                 "(DDoS-held-out rotation, DoS still a known class in this rotation)")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_dos_tradeoff.png", dpi=120)
    plt.close(fig)

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("# Asymmetric alpha_DoS tradeoff -- values used in the plot\n\n")
        f.write(f"Current global alpha_stage1: **{alpha1}**  \n")
        f.write(f"DDoS zero-day rows evaluated: **{int(ddos_mask.sum())}**  \n")
        f.write(f"Genuine DoS rows evaluated: **{int(dos_mask.sum())}**\n\n")
        f.write("| alpha_DoS | DDoS deferral rate (gain) | DoS fast-exit rate (cost) |\n")
        f.write("|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['alpha_dos']:.2f} | {r['ddos_deferral_rate']:.4f} | {r['dos_fast_exit_rate']:.4f} |\n")
        f.write("\nPlot: `alpha_dos_tradeoff.png`\n")
        f.write("\nNote: read-only diagnostic. No config or trained artifact was changed. "
                "To actually apply a chosen alpha_DoS, Gate 1's calibration logic would need "
                "to support per-class alpha (currently a single global alpha_stage1).\n")

    print(f"\nSaved: {OUT_DIR}/result.md, {OUT_DIR}/alpha_dos_tradeoff.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
