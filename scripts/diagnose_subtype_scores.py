"""STANDALONE DIAGNOSTIC -- checks whether the bimodal score clusters seen
in DDoS/DoS/MQTT/Recon's score-distribution plots correspond to distinct
attack SUBTYPES (Attack_type column) rather than being an artifact of
non-shuffled sequence windowing.

Read-only: only loads the already-trained shared AE (never touches Gate 1
models, never touches result.joblib, never retrains anything). Writes to
a NEW, isolated folder (artifacts/diagnostics/) that no other script reads
from or writes to -- cannot collide with or overwrite anything from
run_full_study.py, run_analysis.py, or build_variance_table.py.

Usage:
    python -m scripts.diagnose_subtype_scores --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.loader import load_csv, FEATURE_COLUMNS
from src.data.family_map import KNOWN_FAMILIES
from src.analysis import load_shared_ae_artifacts, compute_dual_scores

OUT_DIR = Path("artifacts") / "diagnostics"  # isolated -- not used by any other script


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading test data (read-only)...")
    test_pool = load_csv(cfg["test_csv"])

    print("Loading shared AE artifacts (read-only, no retraining)...")
    stage2_model, stage2_scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()
    tau = gate2_bundle["threshold"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Subtype-level score breakdown (diagnostic, not part of the main pipeline)\n",
                     f"Threshold τ = {tau:.4f}\n"]

    for family in KNOWN_FAMILIES:
        subtypes = sorted(test_pool.loc[test_pool["family"] == family, "Attack_type"].unique())
        print(f"\n{family}: {len(subtypes)} subtype(s) -- {subtypes}")

        if len(subtypes) <= 1:
            print(f"  skipping plot -- only one subtype, nothing to break down")
            report_lines.append(f"## {family}\nOnly one subtype (`{subtypes[0] if subtypes else 'none'}`) -- "
                                f"no breakdown possible, consistent with its continuous (non-bimodal) score plot.\n")
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes)))
        report_lines.append(f"## {family}\n")
        report_lines.append("| subtype | n_windows | mean score | above τ (%) |")
        report_lines.append("|---|---|---|---|")

        all_scores_for_range = []
        subtype_scores = {}
        for subtype in subtypes:
            rows = test_pool[test_pool["Attack_type"] == subtype]
            scores = compute_dual_scores(rows, stage2_model, stage2_scaler, gate2_bundle,
                                          FEATURE_COLUMNS, seq_len)
            subtype_scores[subtype] = scores
            if len(scores) > 0:
                all_scores_for_range.append(scores)

        if not all_scores_for_range:
            print(f"  skipping -- no sequences built for any subtype")
            continue

        x_max = np.percentile(np.concatenate(all_scores_for_range), 99.5)
        bin_edges = np.linspace(0, max(x_max, tau * 1.2), 50)

        for subtype, color in zip(subtypes, colors):
            scores = subtype_scores[subtype]
            if len(scores) == 0:
                report_lines.append(f"| {subtype} | 0 | n/a | n/a |")
                continue
            ax.hist(scores, bins=bin_edges, density=True, alpha=0.5, label=subtype,
                    color=color, edgecolor="black", linewidth=0.3)
            above_pct = (scores > tau).mean() * 100
            report_lines.append(f"| {subtype} | {len(scores)} | {scores.mean():.4f} | {above_pct:.2f}% |")

        ax.axvline(tau, color="black", linestyle="--", linewidth=2, label="Threshold τ")
        ax.set_xlabel("Anomaly score (dual score)")
        ax.set_ylabel("Density")
        ax.set_title(f"{family} — score distribution broken down by subtype\n"
                     f"(diagnostic: checks if the bimodal pattern = distinct subtypes)")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        save_path = OUT_DIR / f"{family}_subtype_breakdown.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        report_lines.append(f"\nPlot: `{save_path}`\n")
        print(f"  saved: {save_path}")

    with open(OUT_DIR / "subtype_breakdown_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nDone. Report: {OUT_DIR}/subtype_breakdown_report.md")
    print("Nothing outside artifacts/diagnostics/ was read from or written to (besides the read-only shared AE load).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
