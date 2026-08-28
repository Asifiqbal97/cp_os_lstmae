"""LOFO (Leave-One-Family-Out) DIAGNOSTIC 1 -- Recon Ratio (raw MSE, x times
benign) and MSE distribution plots, per held-out family. Combines what
diagnose_recon_ratio.py and the MSE-alone half of diagnose_mse_vs_dual.py
did separately, plus saves every value used to draw each plot into a table
in result.md (not just the plot images).

Read-only: only loads the already-trained shared AE (never touches Gate 1,
never retrains). Writes to a NEW isolated folder (artifacts/diagnostics/
lofo_recon_mse/) -- does not touch summary.md, result.joblib, variance_table,
or any other script's output.

Usage:
    python -m scripts.diagnose_lofo_recon_mse --config configs/mvp.yaml
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
from src.data.benign_split import make_benign_split
from src.analysis import load_shared_ae_artifacts
from src.stage2_anomaly.lstm_ae import predict_recon_latent
from src.stage2_anomaly.sequences import build_sequences

OUT_DIR = Path("artifacts") / "diagnostics" / "lofo_recon_mse"
N_BOOTSTRAP = 200


def raw_mse(df, model, scaler, seq_len):
    X = scaler.transform(df[FEATURE_COLUMNS].values.astype(np.float32)).astype(np.float32)
    seqs = build_sequences(X, seq_len=seq_len)
    if len(seqs) == 0:
        return np.array([])
    recon, _ = predict_recon_latent(model, seqs)
    return np.mean(np.square(seqs - recon), axis=(1, 2))


def bootstrap_ratio(mse_family, benign_mean_mse, n_bootstrap=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n = len(mse_family)
    if n == 0:
        return float("nan"), float("nan")
    ratios = [mse_family[rng.integers(0, n, n)].mean() / benign_mean_mse for _ in range(n_bootstrap)]
    return float(np.mean(ratios)), float(np.std(ratios))


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data (read-only)...")
    test_pool = load_csv(cfg["test_csv"])
    benign_train = load_csv(cfg["benign_train_csv"])

    print("Loading shared AE artifacts (read-only, no retraining)...")
    model, scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()

    # MSE-only threshold: same order-statistic recipe as calibrate_gate2(),
    # applied to raw MSE alone -- rebuilds the SAME benign calib split
    # (same seed, deterministic, not persisted anywhere).
    bsplit = make_benign_split(benign_train, calib_frac=0.2, random_state=cfg["random_state"])
    X_calib_scaled = scaler.transform(bsplit.X_ae_calib).astype(np.float32)
    calib_seqs = build_sequences(X_calib_scaled, seq_len=seq_len)
    calib_mse, _ = predict_recon_latent(model, calib_seqs)
    calib_mse = np.mean(np.square(calib_seqs - calib_mse), axis=(1, 2))
    alpha2 = cfg["alpha_stage2"]
    k = min(int(np.ceil((len(calib_mse) + 1) * (1 - alpha2))), len(calib_mse))
    mse_threshold = float(np.sort(calib_mse)[k - 1])
    print(f"MSE-only threshold: {mse_threshold:.4f}")

    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    mse_benign = raw_mse(benign_rows, model, scaler, seq_len)
    benign_mean_mse = float(mse_benign.mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    ratio_means = []

    for family in KNOWN_FAMILIES:
        print(f"\n{family} (LOFO)...")
        zd_rows = test_pool[test_pool["family"] == family]
        mse_zd = raw_mse(zd_rows, model, scaler, seq_len)
        if len(mse_zd) == 0:
            print("  skipped -- no sequences")
            continue

        ratio_mean, ratio_std = bootstrap_ratio(mse_zd, benign_mean_mse)
        y = np.concatenate([np.ones(len(mse_zd)), np.zeros(len(mse_benign))])
        auc = roc_auc_score(y, np.concatenate([mse_zd, mse_benign]))
        dr = float((mse_zd > mse_threshold).mean())
        fpr = float((mse_benign > mse_threshold).mean())

        rows.append({"family": family, "recon_ratio_mean": ratio_mean, "recon_ratio_std": ratio_std,
                     "benign_mean_mse": benign_mean_mse, "family_mean_mse": float(mse_zd.mean()),
                     "family_std_mse": float(mse_zd.std()), "mse_threshold": mse_threshold,
                     "mse_auc": float(auc), "mse_dr": dr, "mse_fpr": fpr,
                     "n_windows": len(mse_zd)})
        ratio_means.append(ratio_mean)
        print(f"  Recon Ratio={ratio_mean:.4f}±{ratio_std:.4f}  MSE AUC={auc:.4f}  DR={dr:.4f}  FPR={fpr:.4f}")

        # Per-family MSE histogram
        x_max = max(np.percentile(np.concatenate([mse_zd, mse_benign]), 95), mse_threshold * 1.2)
        bins = np.linspace(0, x_max, 50)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.clip(mse_benign, 0, x_max), bins=bins, density=True, alpha=0.6,
                label="Benign (test)", color="#4C72B0", edgecolor="black", linewidth=0.3)
        ax.hist(np.clip(mse_zd, 0, x_max), bins=bins, density=True, alpha=0.6,
                label=f"{family} (LOFO zero-day, test)", color="#C44E52", edgecolor="black", linewidth=0.3)
        ax.axvline(mse_threshold, color="black", linestyle="--", linewidth=2, label="MSE-only threshold")
        ax.text(0.97, 0.75, f"AUC={auc:.4f}\nDR={dr*100:.2f}%\nFPR={fpr*100:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
        ax.set_xlabel("Raw reconstruction error (MSE)")
        ax.set_ylabel("Density")
        ax.set_title(f"LOFO {family}: MSE distribution, Benign vs held-out family")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{family}_mse_distribution.png", dpi=120)
        plt.close(fig)

    # Recon Ratio bar chart (log scale, since ratios can span orders of magnitude)
    families_ok = [r["family"] for r in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families_ok)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(families_ok, ratio_means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Benign baseline (ratio=1)")
    ax.set_yscale("log")
    ax.set_ylabel("Recon Ratio (log scale)")
    ax.set_title("LOFO Recon Ratio per held-out family")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recon_ratio_bar_logscale.png", dpi=120)
    plt.close(fig)

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("# LOFO Recon Ratio + MSE Distribution -- values used in every plot\n\n")
        f.write(f"MSE-only threshold (alpha={alpha2}): **{mse_threshold:.4f}**  \n")
        f.write(f"Benign mean raw MSE: **{benign_mean_mse:.6f}**\n\n")
        f.write("| Family | Recon Ratio | Family mean MSE | Family std MSE | MSE AUC | DR@thresh | FPR@thresh | n_windows |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['family']} | {r['recon_ratio_mean']:.4f}±{r['recon_ratio_std']:.4f} | "
                    f"{r['family_mean_mse']:.6f} | {r['family_std_mse']:.6f} | {r['mse_auc']:.4f} | "
                    f"{r['mse_dr']:.4f} | {r['mse_fpr']:.4f} | {r['n_windows']} |\n")
        f.write("\nPlots: `<family>_mse_distribution.png` (per family), `recon_ratio_bar_logscale.png` (all families).\n")

    print(f"\nSaved: {OUT_DIR}/result.md + plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
