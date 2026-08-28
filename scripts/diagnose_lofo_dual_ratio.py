"""LOFO (Leave-One-Family-Out) DIAGNOSTIC 2 -- Dual Score Ratio (combined
MSE+latent-deviation score, x times benign) and dual score distribution
plots, per held-out family. Uses the ACTUAL calibrated tau from the shared
AE (the same threshold the real pipeline uses), unlike script 1's
MSE-only threshold. Saves every value used to draw each plot into a table
in result.md.

Read-only: only loads the already-trained shared AE (never touches Gate 1,
never retrains). Writes to a NEW isolated folder (artifacts/diagnostics/
lofo_dual_ratio/) -- does not touch summary.md, result.joblib,
variance_table, or any other script's output.

Usage:
    python -m scripts.diagnose_lofo_dual_ratio --config configs/mvp.yaml
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
from src.stage2_anomaly.conformal_gate import dual_score

OUT_DIR = Path("artifacts") / "diagnostics" / "lofo_dual_ratio"
N_BOOTSTRAP = 200


def compute_scores(df, model, scaler, gate2_bundle, seq_len):
    X = scaler.transform(df[FEATURE_COLUMNS].values.astype(np.float32)).astype(np.float32)
    seqs = build_sequences(X, seq_len=seq_len)
    if len(seqs) == 0:
        return np.array([])
    recon, z = predict_recon_latent(model, seqs)
    mse = np.mean(np.square(seqs - recon), axis=(1, 2))
    ldev = np.linalg.norm(z - gate2_bundle["centroid"], axis=1)
    return dual_score(mse, ldev, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])


def bootstrap_ratio(scores_family, benign_mean_score, n_bootstrap=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores_family)
    if n == 0:
        return float("nan"), float("nan")
    ratios = [scores_family[rng.integers(0, n, n)].mean() / benign_mean_score for _ in range(n_bootstrap)]
    return float(np.mean(ratios)), float(np.std(ratios))


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data (read-only)...")
    test_pool = load_csv(cfg["test_csv"])

    print("Loading shared AE artifacts (read-only, no retraining)...")
    model, scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()
    tau = gate2_bundle["threshold"]
    print(f"Calibrated dual-score threshold (tau): {tau:.4f}")

    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    dual_benign = compute_scores(benign_rows, model, scaler, gate2_bundle, seq_len)
    benign_mean_dual = float(dual_benign.mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    ratio_means = []

    for family in KNOWN_FAMILIES:
        print(f"\n{family} (LOFO)...")
        zd_rows = test_pool[test_pool["family"] == family]
        dual_zd = compute_scores(zd_rows, model, scaler, gate2_bundle, seq_len)
        if len(dual_zd) == 0:
            print("  skipped -- no sequences")
            continue

        ratio_mean, ratio_std = bootstrap_ratio(dual_zd, benign_mean_dual)
        y = np.concatenate([np.ones(len(dual_zd)), np.zeros(len(dual_benign))])
        auc = roc_auc_score(y, np.concatenate([dual_zd, dual_benign]))
        dr = float((dual_zd > tau).mean())
        fpr = float((dual_benign > tau).mean())

        rows.append({"family": family, "dual_ratio_mean": ratio_mean, "dual_ratio_std": ratio_std,
                     "benign_mean_dual": benign_mean_dual, "family_mean_dual": float(dual_zd.mean()),
                     "family_std_dual": float(dual_zd.std()), "tau": tau,
                     "dual_auc": float(auc), "dual_dr": dr, "dual_fpr": fpr,
                     "n_windows": len(dual_zd)})
        ratio_means.append(ratio_mean)
        print(f"  Dual Ratio={ratio_mean:.4f}±{ratio_std:.4f}  AUC={auc:.4f}  DR={dr:.4f}  FPR={fpr:.4f}")

        # Per-family dual score histogram
        x_max = max(np.percentile(np.concatenate([dual_zd, dual_benign]), 95), tau * 1.2)
        bins = np.linspace(0, x_max, 50)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.clip(dual_benign, 0, x_max), bins=bins, density=True, alpha=0.6,
                label="Benign (test)", color="#4C72B0", edgecolor="black", linewidth=0.3)
        ax.hist(np.clip(dual_zd, 0, x_max), bins=bins, density=True, alpha=0.6,
                label=f"{family} (LOFO zero-day, test)", color="#C44E52", edgecolor="black", linewidth=0.3)
        ax.axvline(tau, color="black", linestyle="--", linewidth=2, label="Threshold τ (calibrated)")
        ax.text(0.97, 0.75, f"AUC={auc:.4f}\nDR={dr*100:.2f}%\nFPR={fpr*100:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
        ax.set_xlabel("Dual score (MSE + latent-centroid deviation, combined)")
        ax.set_ylabel("Density")
        ax.set_title(f"LOFO {family}: dual-score distribution, Benign vs held-out family")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{family}_dual_distribution.png", dpi=120)
        plt.close(fig)

    families_ok = [r["family"] for r in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families_ok)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(families_ok, ratio_means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Benign baseline (ratio=1)")
    ax.set_ylabel("Dual Score Ratio")
    ax.set_title("LOFO Dual Score Ratio per held-out family")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dual_ratio_bar.png", dpi=120)
    plt.close(fig)

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("# LOFO Dual Score Ratio + Distribution -- values used in every plot\n\n")
        f.write(f"Calibrated dual-score threshold (tau): **{tau:.4f}**  \n")
        f.write(f"Benign mean dual score: **{benign_mean_dual:.6f}**\n\n")
        f.write("| Family | Dual Ratio | Family mean score | Family std score | Dual AUC | DR@τ | FPR@τ | n_windows |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['family']} | {r['dual_ratio_mean']:.4f}±{r['dual_ratio_std']:.4f} | "
                    f"{r['family_mean_dual']:.6f} | {r['family_std_dual']:.6f} | {r['dual_auc']:.4f} | "
                    f"{r['dual_dr']:.4f} | {r['dual_fpr']:.4f} | {r['n_windows']} |\n")
        f.write("\nPlots: `<family>_dual_distribution.png` (per family), `dual_ratio_bar.png` (all families).\n")

    print(f"\nSaved: {OUT_DIR}/result.md + plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
