"""STANDALONE DIAGNOSTIC -- subtype-level MSE score distributions.

Read-only:
- Loads the already-trained shared AE.
- Does not load or modify Gate 1 models.
- Does not load result.joblib.
- Does not retrain anything.
- Writes only to artifacts/diagnostics/.

This version plots RECONSTRUCTION MSE ONLY, not the dual score.
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
from src.analysis import load_shared_ae_artifacts


OUT_DIR = Path("artifacts") / "diagnostics"


def compute_mse_scores(
    rows,
    model,
    scaler,
    feature_columns,
    seq_len,
):
    """Build sequences and calculate reconstruction MSE only."""

    X = rows[feature_columns].to_numpy(dtype=np.float32)

    if len(X) < seq_len:
        return np.array([])

    # Scale using the already-fitted AE scaler.
    X_scaled = scaler.transform(X)

    # Build non-overlapping/rolling sequence windows in the same basic
    # sequential manner expected by the AE.
    n_windows = len(X_scaled) - seq_len + 1

    X_seq = np.stack(
        [
            X_scaled[i:i + seq_len]
            for i in range(n_windows)
        ],
        axis=0,
    )

    # Reconstruct with the already-trained AE.
    X_reconstructed = model.predict(X_seq, verbose=0)

    # MSE for each sequence window.
    mse_scores = np.mean(
        np.square(X_seq - X_reconstructed),
        axis=(1, 2),
    )

    return mse_scores


def main(cfg_path: str):

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading test data (read-only)...")
    test_pool = load_csv(cfg["test_csv"])

    print("Loading shared AE artifacts (read-only, no retraining)...")
    stage2_model, stage2_scaler, gate2_bundle, seq_len = (
        load_shared_ae_artifacts()
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Subtype-level reconstruction-MSE breakdown\n",
        "This diagnostic uses reconstruction MSE only, "
        "not the dual anomaly score.\n",
    ]

    for family in KNOWN_FAMILIES:

        family_rows = test_pool[test_pool["family"] == family]

        subtypes = sorted(
            family_rows["Attack_type"].dropna().unique()
        )

        print(f"\n{family}: {len(subtypes)} subtype(s) -- {subtypes}")

        if len(subtypes) <= 1:
            print("  skipping plot -- only one subtype")

            report_lines.append(
                f"## {family}\n"
                f"Only one subtype "
                f"(`{subtypes[0] if subtypes else 'none'}`) -- "
                "no subtype breakdown possible.\n"
            )
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        colors = plt.cm.tab10(
            np.linspace(0, 1, len(subtypes))
        )

        report_lines.append(f"## {family}\n")
        report_lines.append(
            "| subtype | n_windows | mean MSE | median MSE |"
        )
        report_lines.append(
            "|---|---:|---:|---:|"
        )

        subtype_scores = {}
        all_scores = []

        # ---------------------------------------------------------
        # Calculate MSE separately for every Attack_type.
        # ---------------------------------------------------------
        for subtype in subtypes:

            rows = family_rows[
                family_rows["Attack_type"] == subtype
            ]

            scores = compute_mse_scores(
                rows,
                stage2_model,
                stage2_scaler,
                FEATURE_COLUMNS,
                seq_len,
            )

            subtype_scores[subtype] = scores

            if len(scores) > 0:
                all_scores.append(scores)

        if not all_scores:
            print("  skipping -- no sequences built")
            plt.close(fig)
            continue

        # Use a common x-axis range for all subtype histograms.
        concatenated = np.concatenate(all_scores)

        x_max = np.percentile(concatenated, 99.5)

        bin_edges = np.linspace(
            0,
            max(x_max, 1e-12),
            50,
        )

        # ---------------------------------------------------------
        # Plot each subtype's MSE distribution.
        # ---------------------------------------------------------
        for subtype, color in zip(subtypes, colors):

            scores = subtype_scores[subtype]

            if len(scores) == 0:
                report_lines.append(
                    f"| {subtype} | 0 | n/a | n/a |"
                )
                continue

            ax.hist(
                scores,
                bins=bin_edges,
                density=True,
                alpha=0.5,
                label=subtype,
                color=color,
                edgecolor="black",
                linewidth=0.3,
            )

            report_lines.append(
                f"| {subtype} | "
                f"{len(scores)} | "
                f"{scores.mean():.6f} | "
                f"{np.median(scores):.6f} |"
            )

        ax.set_xlabel("Reconstruction MSE")
        ax.set_ylabel("Density")

        ax.set_title(
            f"{family} — reconstruction-MSE distribution "
            f"broken down by subtype\n"
            f"(diagnostic: MSE only, no latent-centroid term)"
        )

        ax.legend(
            loc="upper right",
            fontsize=7,
        )

        fig.tight_layout()

        save_path = (
            OUT_DIR /
            f"{family}_subtype_mse_breakdown.png"
        )

        fig.savefig(
            save_path,
            dpi=120,
        )

        plt.close(fig)

        report_lines.append(
            f"\nPlot: `{save_path}`\n"
        )

        print(f"  saved: {save_path}")

    # -------------------------------------------------------------
    # Write report.
    # -------------------------------------------------------------
    report_path = (
        OUT_DIR /
        "subtype_mse_breakdown_report.md"
    )

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nDone. Report: {report_path}")
    print(
        "Only artifacts/diagnostics/ was written to."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
    )

    args = parser.parse_args()

    main(args.config)
