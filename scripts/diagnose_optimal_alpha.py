"""STANDALONE EXPERIMENT -- finds the AUC-optimal score_alpha (the
MSE/latent-deviation blend weight) PER FAMILY, instead of the current
single global value (score_alpha=0.5 in configs/mvp.yaml).

score_alpha=1.0 means MSE-alone; score_alpha=0.0 means latent-deviation-
alone; the current default sits at 0.5. This sweeps the full range and
finds each family's optimum, extending the diagnose_mse_vs_dual.py finding
(which only checked the two endpoints + the current default) to the whole
curve, showing whether an intermediate value beats both endpoints for any
family.

Read-only: only loads the already-trained shared AE (never touches Gate 1,
never retrains). Writes to a NEW isolated folder (artifacts/diagnostics/
optimal_alpha/) -- does not touch summary.md, result.joblib, variance_table,
or any other script's output. Does NOT change configs/mvp.yaml or any
trained artifact -- this is a read-only "what if" experiment, not a change
to the pipeline.

Usage:
    python -m scripts.diagnose_optimal_alpha --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from src.data.loader import load_csv, FEATURE_COLUMNS
from src.data.family_map import KNOWN_FAMILIES, BENIGN_LABEL
from src.analysis import load_shared_ae_artifacts
from src.stage2_anomaly.lstm_ae import predict_recon_latent
from src.stage2_anomaly.sequences import build_sequences

OUT_DIR = Path("artifacts") / "diagnostics" / "optimal_alpha"  # isolated
ALPHA_GRID = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00


def compute_mse_ldev(df, model, scaler, centroid, seq_len):
    X = scaler.transform(df[FEATURE_COLUMNS].values.astype(np.float32)).astype(np.float32)
    seqs = build_sequences(X, seq_len=seq_len)
    if len(seqs) == 0:
        return np.array([]), np.array([])
    recon, z = predict_recon_latent(model, seqs)
    mse = np.mean(np.square(seqs - recon), axis=(1, 2))
    ldev = np.linalg.norm(z - centroid, axis=1)
    return mse, ldev


def plot_bar_summary(rows, current_alpha, save_path):
    """One bar per family (not overlapping lines) for at-a-glance comparison:
    left panel = each family's optimal score_alpha vs the current fixed value;
    right panel = AUC improvement each family would gain from that optimum."""
    families = [r["family"] for r in rows]
    best_alphas = [r["best_alpha"] for r in rows]
    improvements = [r["improvement"] for r in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(families, best_alphas, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(current_alpha, color="black", linestyle="--", linewidth=1.5,
                label=f"current score_alpha={current_alpha}")
    ax1.set_ylabel("Optimal score_alpha")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Best score_alpha per family")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(families, improvements, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("AUC improvement over current alpha")
    ax2.set_title("Gain from per-family tuning")
    ax2.tick_params(axis="x", rotation=20)

    fig.suptitle("Per-family score_alpha: optimal value and improvement (bar view)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data (read-only)...")
    test_pool = load_csv(cfg["test_csv"])

    print("Loading shared AE artifacts (read-only, no retraining)...")
    model, scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()
    centroid = gate2_bundle["centroid"]
    mse_p99, ldev_p99 = gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"]
    current_alpha = cfg["score_alpha"]

    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    mse_benign, ldev_benign = compute_mse_ldev(benign_rows, model, scaler, centroid, seq_len)
    mse_benign_n = mse_benign / (mse_p99 + 1e-8)
    ldev_benign_n = ldev_benign / (ldev_p99 + 1e-8)

    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_all, ax_all = plt.subplots(figsize=(9, 6))

    for family in KNOWN_FAMILIES:
        print(f"\n{family}: sweeping score_alpha...")
        zd_rows = test_pool[test_pool["family"] == family]
        mse_zd, ldev_zd = compute_mse_ldev(zd_rows, model, scaler, centroid, seq_len)
        if len(mse_zd) == 0 or len(mse_benign) == 0:
            print("  skipped -- not enough sequences")
            continue
        mse_zd_n = mse_zd / (mse_p99 + 1e-8)
        ldev_zd_n = ldev_zd / (ldev_p99 + 1e-8)

        y = np.concatenate([np.ones(len(mse_zd)), np.zeros(len(mse_benign))])
        aucs = []
        for a in ALPHA_GRID:
            score_zd = a * np.clip(mse_zd_n, 0, None) + (1 - a) * np.clip(ldev_zd_n, 0, None)
            score_benign = a * np.clip(mse_benign_n, 0, None) + (1 - a) * np.clip(ldev_benign_n, 0, None)
            aucs.append(roc_auc_score(y, np.concatenate([score_zd, score_benign])))
        aucs = np.array(aucs)

        best_idx = int(np.argmax(aucs))
        best_alpha, best_auc = float(ALPHA_GRID[best_idx]), float(aucs[best_idx])
        auc_at_current = float(np.interp(current_alpha, ALPHA_GRID, aucs))

        rows.append({"family": family, "best_alpha": best_alpha, "best_auc": best_auc,
                     "current_alpha": current_alpha, "auc_at_current": auc_at_current,
                     "improvement": best_auc - auc_at_current})
        print(f"  best score_alpha={best_alpha:.2f} (AUC={best_auc:.4f})  "
              f"vs current alpha={current_alpha:.2f} (AUC={auc_at_current:.4f})  "
              f"improvement={best_auc - auc_at_current:+.4f}")

        ax_all.plot(ALPHA_GRID, aucs, marker="o", markersize=3, label=family)

    ax_all.axvline(current_alpha, color="black", linestyle="--", linewidth=1.5,
                   label=f"current score_alpha={current_alpha}")
    ax_all.set_xlabel("score_alpha (1.0 = MSE-alone, 0.0 = latent-deviation-alone)")
    ax_all.set_ylabel("AUC (benign vs zero-day family)")
    ax_all.set_title("AUC vs score_alpha, per family\n(where each curve peaks = that family's optimal blend weight)")
    ax_all.legend(loc="best", fontsize=8)
    fig_all.tight_layout()
    fig_all.savefig(OUT_DIR / "auc_vs_alpha_all_families.png", dpi=120)
    plt.close(fig_all)

    plot_bar_summary(rows, current_alpha, OUT_DIR / "alpha_bar_summary.png")

    print("\n" + "=" * 90)
    print(f"{'Family':<12}{'Best alpha':<14}{'Best AUC':<12}{'Current AUC':<14}{'Improvement':<14}")
    print("-" * 90)
    for r in rows:
        print(f"{r['family']:<12}{r['best_alpha']:<14.2f}{r['best_auc']:<12.4f}"
              f"{r['auc_at_current']:<14.4f}{r['improvement']:<+14.4f}")
    print("=" * 90)

    with open(OUT_DIR / "optimal_alpha_report.md", "w") as f:
        f.write("# Per-family optimal score_alpha (diagnostic, not applied to the pipeline)\n\n")
        f.write(f"Current global score_alpha in configs/mvp.yaml: **{current_alpha}**\n\n")
        f.write("| Family | Best alpha | Best AUC | AUC at current alpha | Improvement |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['family']} | {r['best_alpha']:.2f} | {r['best_auc']:.4f} | "
                    f"{r['auc_at_current']:.4f} | {r['improvement']:+.4f} |\n")
        f.write("\nSee `auc_vs_alpha_all_families.png` for the full curves, "
               "or `alpha_bar_summary.png` for the at-a-glance bar comparison.\n")
        f.write("\nNote: this is a read-only diagnostic. No trained artifact or config file was changed.\n")

    print(f"\nSaved: {OUT_DIR}/optimal_alpha_report.md, {OUT_DIR}/auc_vs_alpha_all_families.png, "
          f"{OUT_DIR}/alpha_bar_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
