"""STANDALONE EXPERIMENT -- checks whether raw reconstruction error (MSE)
ALONE separates benign vs zero-day better than the combined dual score.
Motivation: if MSE alone is a strong signal but latent-deviation is weak
for some class, a fixed 50/50 blend (score_alpha) can DILUTE separation
rather than improve it.

Read-only: only loads the already-trained shared AE (never touches Gate 1,
never retrains). Writes to a NEW isolated folder (artifacts/diagnostics/
mse_vs_dual/) -- does not touch summary.md, result.joblib, variance_table,
or any other script's output.

Usage:
    python -m scripts.diagnose_mse_vs_dual --config configs/mvp.yaml
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
from src.stage2_anomaly.conformal_gate import dual_score

OUT_DIR = Path("artifacts") / "diagnostics" / "mse_vs_dual"  # isolated


def compute_mse_ldev(df, model, scaler, seq_len):
    X = scaler.transform(df[FEATURE_COLUMNS].values.astype(np.float32)).astype(np.float32)
    seqs = build_sequences(X, seq_len=seq_len)
    if len(seqs) == 0:
        return np.array([]), np.array([])
    recon, z = predict_recon_latent(model, seqs)
    mse = np.mean(np.square(seqs - recon), axis=(1, 2))
    return mse, z


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data (read-only)...")
    test_pool = load_csv(cfg["test_csv"])
    benign_train = load_csv(cfg["benign_train_csv"])

    print("Loading shared AE artifacts (read-only, no retraining)...")
    model, scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()
    centroid = gate2_bundle["centroid"]

    # MSE-only threshold: same order-statistic calibration recipe as
    # calibrate_gate2(), applied to raw MSE alone instead of the combined
    # dual score. Rebuilds the SAME benign calibration split (same seed) --
    # not persisted anywhere, deterministic, no leakage risk.
    print("Rebuilding benign calibration split (deterministic, same seed as training)...")
    bsplit = make_benign_split(benign_train, calib_frac=0.2, random_state=cfg["random_state"])
    X_calib_scaled = scaler.transform(bsplit.X_ae_calib).astype(np.float32)
    calib_seqs = build_sequences(X_calib_scaled, seq_len=seq_len)
    calib_mse, _ = predict_recon_latent(model, calib_seqs)
    calib_mse = np.mean(np.square(calib_seqs - calib_mse), axis=(1, 2))
    alpha2 = cfg["alpha_stage2"]
    k = min(int(np.ceil((len(calib_mse) + 1) * (1 - alpha2))), len(calib_mse))
    mse_threshold = float(np.sort(calib_mse)[k - 1])
    print(f"MSE-only threshold (alpha={alpha2}): {mse_threshold:.4f}")

    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    mse_benign, _ = compute_mse_ldev(benign_rows, model, scaler, seq_len)

    rows = []
    for family in KNOWN_FAMILIES:
        print(f"\n{family}...")
        zd_rows = test_pool[test_pool["family"] == family]
        mse_zd, z_zd = compute_mse_ldev(zd_rows, model, scaler, seq_len)
        if len(mse_zd) == 0 or len(mse_benign) == 0:
            print("  skipped -- not enough sequences")
            continue
        _, z_benign = compute_mse_ldev(benign_rows, model, scaler, seq_len)
        ldev_zd = np.linalg.norm(z_zd - centroid, axis=1)
        ldev_benign = np.linalg.norm(z_benign - centroid, axis=1)

        # MSE-alone
        y = np.concatenate([np.ones(len(mse_zd)), np.zeros(len(mse_benign))])
        mse_auc = roc_auc_score(y, np.concatenate([mse_zd, mse_benign]))
        mse_dr = float((mse_zd > mse_threshold).mean())
        mse_fpr = float((mse_benign > mse_threshold).mean())

        # Dual score (existing combined metric, for direct comparison)
        dual_zd = dual_score(mse_zd, ldev_zd, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])
        dual_benign = dual_score(mse_benign, ldev_benign, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])
        dual_auc = roc_auc_score(y, np.concatenate([dual_zd, dual_benign]))

        better = "MSE-alone" if mse_auc > dual_auc else ("Dual score" if dual_auc > mse_auc else "tie")
        rows.append({"family": family, "mse_auc": mse_auc, "dual_auc": dual_auc,
                     "mse_dr": mse_dr, "mse_fpr": mse_fpr, "better": better})
        print(f"  MSE-alone AUC={mse_auc:.4f}  Dual-score AUC={dual_auc:.4f}  -> {better} separates better")

        # Plot: MSE-alone histogram, x-axis clipped at 95th pct (avoids the
        # Recon-style unreadable-outlier problem from earlier plots)
        x_max = max(np.percentile(np.concatenate([mse_zd, mse_benign]), 95), mse_threshold * 1.2)
        bins = np.linspace(0, x_max, 50)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.clip(mse_benign, 0, x_max), bins=bins, density=True, alpha=0.6,
                label="Benign (test)", color="#4C72B0", edgecolor="black", linewidth=0.3)
        ax.hist(np.clip(mse_zd, 0, x_max), bins=bins, density=True, alpha=0.6,
                label=f"{family} (zero-day, test)", color="#C44E52", edgecolor="black", linewidth=0.3)
        ax.axvline(mse_threshold, color="black", linestyle="--", linewidth=2, label="MSE-only threshold")
        ax.text(0.97, 0.75, f"MSE AUC = {mse_auc:.4f}\nDual AUC = {dual_auc:.4f}\nBetter: {better}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
        ax.set_xlabel("Raw reconstruction error (MSE) -- no latent-deviation blend")
        ax.set_ylabel("Density")
        ax.set_title(f"MSE-ALONE vs Benign: {family}")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / f"{family}_mse_alone.png", dpi=120)
        plt.close(fig)

    print("\n" + "=" * 70)
    print(f"{'Family':<12}{'MSE AUC':<12}{'Dual AUC':<12}{'Better':<12}")
    print("-" * 70)
    for r in rows:
        print(f"{r['family']:<12}{r['mse_auc']:<12.4f}{r['dual_auc']:<12.4f}{r['better']:<12}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mse_vs_dual_report.md", "w") as f:
        f.write("# MSE-alone vs Dual-score AUC comparison (diagnostic)\n\n")
        f.write("| Family | MSE AUC | Dual AUC | MSE DR@thresh | MSE FPR@thresh | Better separation |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['family']} | {r['mse_auc']:.4f} | {r['dual_auc']:.4f} | "
                    f"{r['mse_dr']:.4f} | {r['mse_fpr']:.4f} | {r['better']} |\n")
    print(f"\nSaved: {OUT_DIR}/mse_vs_dual_report.md, {OUT_DIR}/<family>_mse_alone.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
