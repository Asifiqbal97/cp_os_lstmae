"""STANDALONE EXPERIMENT -- plots the "Recon Ratio" (mean raw reconstruction
error / benign's mean raw reconstruction error) per family, as its own bar
chart with bootstrap error bars. This number already exists as a column in
build_variance_table.py's variance_table.md -- this script is the missing
VISUAL for that same number (was previously only a table value).

Read-only: only loads the already-trained shared AE (never touches Gate 1,
never retrains, never touches result.joblib). Writes to a NEW isolated
folder (artifacts/diagnostics/recon_ratio/) -- does not touch summary.md,
variance_table, or any other script's output.

Usage:
    python -m scripts.diagnose_recon_ratio --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.loader import load_csv, FEATURE_COLUMNS
from src.data.family_map import KNOWN_FAMILIES, BENIGN_LABEL
from src.analysis import load_shared_ae_artifacts
from src.stage2_anomaly.lstm_ae import predict_recon_latent
from src.stage2_anomaly.sequences import build_sequences

OUT_DIR = Path("artifacts") / "diagnostics" / "recon_ratio"  # isolated
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

    print("Loading shared AE artifacts (read-only, no retraining)...")
    model, scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()

    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    mse_benign = raw_mse(benign_rows, model, scaler, seq_len)
    benign_mean_mse = float(mse_benign.mean())
    print(f"Benign mean raw MSE: {benign_mean_mse:.6f}")

    rows = []
    for family in KNOWN_FAMILIES:
        print(f"{family}...")
        zd_rows = test_pool[test_pool["family"] == family]
        mse_zd = raw_mse(zd_rows, model, scaler, seq_len)
        ratio_mean, ratio_std = bootstrap_ratio(mse_zd, benign_mean_mse)
        rows.append({"family": family, "ratio_mean": ratio_mean, "ratio_std": ratio_std,
                     "n_windows": len(mse_zd)})
        print(f"  Recon Ratio = {ratio_mean:.4f} ± {ratio_std:.4f}  (n={len(mse_zd)} windows)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    families = [r["family"] for r in rows]
    means = [r["ratio_mean"] for r in rows]
    stds = [r["ratio_std"] for r in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(families, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Benign baseline (ratio=1)")
    ax.set_ylabel("Recon Ratio (mean MSE ÷ benign mean MSE)")
    ax.set_title("Reconstruction-error ratio per family\n(raw MSE, no latent-deviation blend, no normalization)")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recon_ratio_bar.png", dpi=120)
    plt.close(fig)

    # Log-scale companion plot -- linear scale makes small-ratio families
    # (e.g. Spoofing near 1x) invisible next to large-ratio ones (Recon at
    # 1000x+), same issue as the earlier mis-scaled histograms.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(families, means, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Benign baseline (ratio=1)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Recon Ratio (log scale)")
    ax2.set_title("Reconstruction-error ratio per family (log scale)")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="x", rotation=20)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "recon_ratio_bar_logscale.png", dpi=120)
    plt.close(fig2)

    print("\n" + "=" * 60)
    print(f"{'Family':<12}{'Recon Ratio':<20}{'n_windows':<12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['family']:<12}{r['ratio_mean']:.4f} ± {r['ratio_std']:.4f}    {r['n_windows']:<12}")
    print("=" * 60)

    with open(OUT_DIR / "recon_ratio_report.md", "w") as f:
        f.write("# Reconstruction-Error Ratio per family (diagnostic plot)\n\n")
        f.write("Raw MSE only -- no latent-deviation blend, no P99 normalization "
                "(unlike the dual score used elsewhere). Ratio = family's mean MSE "
                "÷ benign's mean MSE, bootstrap ± std.\n\n")
        f.write("| Family | Recon Ratio | n_windows |\n")
        f.write("|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['family']} | {r['ratio_mean']:.4f} ± {r['ratio_std']:.4f} | {r['n_windows']} |\n")
        f.write("\nSee `recon_ratio_bar.png` (linear scale) and "
                "`recon_ratio_bar_logscale.png` (log scale, for comparing "
                "small- and large-ratio families on one chart).\n")

    print(f"\nSaved: {OUT_DIR}/recon_ratio_report.md, {OUT_DIR}/recon_ratio_bar.png, "
          f"{OUT_DIR}/recon_ratio_bar_logscale.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
