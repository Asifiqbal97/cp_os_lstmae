"""Matplotlib plots (Agg backend, no display needed) for the coverage-
validation experiment and ROC curves. Saved as PNG under
artifacts/<family>/plots/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_coverage_validation(coverage_result: dict, alpha1: float, save_path):
    rows = coverage_result["rows"]
    classes = [r["class"] for r in rows]
    x = np.arange(len(classes))

    mondrian = [r["mondrian_coverage"] for r in rows]
    marginal = [r["marginal_cp_coverage"] for r in rows]
    msp = [r["msp_coverage"] for r in rows]
    target = 1 - alpha1

    band_lo = [r["beta_tolerance_band"][0] for r in rows]
    band_hi = [r["beta_tolerance_band"][1] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    ax.bar(x - width, mondrian, width, label="Mondrian (per-class CP)", color="#4C72B0")
    ax.bar(x, marginal, width, label="Marginal-CP (global threshold)", color="#DD8452")
    ax.bar(x + width, msp, width, label="MSP (swept softmax)", color="#55A868")

    ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"nominal target (1-α={target:.2f})")

    for i, (lo, hi) in enumerate(zip(band_lo, band_hi)):
        if not np.isnan(lo):
            ax.plot([i - width, i - width], [lo, hi], color="black", linewidth=2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20)
    ax.set_ylabel("Achieved coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage validation: achieved vs nominal, per class\n"
                 "(black bars = Beta finite-sample tolerance band, Mondrian only)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_gate1_roc(roc_curves: dict, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, curve in roc_curves.items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{cls} (AUC={curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Gate 1 (LightGBM) — one-vs-rest ROC per known class")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_gate2_roc(roc_result: dict, held_out_family: str, save_path):
    if roc_result is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_result["fpr"], roc_result["tpr"],
            label=f"Benign vs {held_out_family} (AUC={roc_result['auc']:.3f})", color="#C44E52")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Gate 2 (shared LSTM-AE) — Benign vs {held_out_family}\n"
                 f"(dual anomaly score, all thresholds -- not just the fixed calibrated one)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_score_distribution(roc_result: dict, held_out_family: str, save_path, bins: int = 60):
    """Anomaly-score histogram (benign vs zero-day family) with the current
    calibrated threshold tau marked -- for visually judging whether a
    different tau would trade DR/FPR differently. Same style as the paper's
    own threshold-selection figure.

    X-AXIS FIX: originally scaled to the 99.5th percentile, which for
    heavy-tailed families (Recon specifically) got dragged out by rare
    extreme-outlier windows, crushing the entire readable distribution into
    an unreadable sliver near 0. Now uses the 95th percentile for the
    visible range, with a text note on how much data (if any) falls beyond
    it -- so outliers don't destroy readability, but their existence isn't
    hidden either.
    """
    if roc_result is None:
        return
    benign_scores = roc_result["benign_scores"]
    zd_scores = roc_result["zd_scores"]
    tau = roc_result["tau"]
    all_scores = np.concatenate([benign_scores, zd_scores])

    x_max = max(np.percentile(all_scores, 95), tau * 1.2)
    frac_beyond = float((all_scores > x_max).mean())
    bin_edges = np.linspace(0, x_max, bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.clip(benign_scores, 0, x_max), bins=bin_edges, density=True, alpha=0.6,
            label="Benign (test)", color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.hist(np.clip(zd_scores, 0, x_max), bins=bin_edges, density=True, alpha=0.6,
            label=f"{held_out_family} (zero-day, test)", color="#C44E52", edgecolor="black", linewidth=0.3)
    ax.axvline(tau, color="black", linestyle="--", linewidth=2, label="Threshold τ (calibrated)")

    stats_text = (f"DR = {roc_result['dr_at_tau']*100:.2f}%\n"
                  f"FPR = {roc_result['fpr_at_tau']*100:.2f}%\n"
                  f"AUC = {roc_result['auc']:.4f}")
    ax.text(0.97, 0.75, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlabel("Anomaly score (dual score: reconstruction error + latent-centroid deviation)")
    ax.set_ylabel("Density")
    title = (f"Gate 2 score distribution: Benign vs {held_out_family}\n"
             f"at the current calibrated τ — use this to judge if a different τ trades DR/FPR better")
    if frac_beyond > 0.001:
        title += f"\n(x-axis clipped at {x_max:.2f}; {frac_beyond*100:.1f}% of points extend further right)"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
