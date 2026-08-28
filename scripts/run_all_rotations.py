"""Runs the full 5-family zero-day rotation (Spoofing, MQTT, Recon, DDoS,
DoS), loading the CSVs and building the shared AE ONCE, reusing both
across all five rotations. Prints a final summary table with mean +/- std
across rotations, matching the paper's own reporting convention.

Note: does NOT train the closed-set LightGBM (M6) -- use
run_full_study.py for that. This script is kept for rotation-only runs.

Usage:
    python -m scripts.run_all_rotations --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.loader import load_csv
from src.data.family_map import KNOWN_FAMILIES
from src.shared_ae import build_shared_ae
from src.run_rotation import run_one_rotation
from src.report_saving import save_summary_report


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("[1/3] Loading data (once, reused across all rotations)...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])
    benign_train = load_csv(cfg["benign_train_csv"])

    print("[2/3] Building shared AE (once, reused across all rotations)...")
    shared_ae, shared_scaler, shared_gate2_bundle = build_shared_ae(benign_train, cfg)

    print("[3/3] Running rotations...")
    all_results = {}
    for family in KNOWN_FAMILIES:
        print(f"\n\n{'#'*60}\n# ROTATION: {family}\n{'#'*60}")
        rotation_cfg = dict(cfg)
        rotation_cfg["held_out_family"] = family
        all_results[family] = run_one_rotation(
            train_pool, test_pool, family, rotation_cfg,
            shared_ae, shared_scaler, shared_gate2_bundle,
        )

    print_summary_table(all_results)
    save_summary_report(Path("artifacts"), all_results)
    print("\nSummary saved to artifacts/summary.json and artifacts/summary.md")
    return all_results


def print_summary_table(all_results: dict):
    print(f"\n\n{'='*70}\nSUMMARY ACROSS ALL {len(all_results)} ROTATIONS\n{'='*70}")

    metrics_to_summarize = [
        ("known_class_accuracy", "pipeline_metrics"),
        ("known_class_macro_f1", "pipeline_metrics"),
        ("zero_day_recall", "pipeline_metrics"),
        ("benign_false_alarm_rate", "pipeline_metrics"),
    ]

    header = f"{'family':<12}" + "".join(f"{m:>22}" for m, _ in metrics_to_summarize)
    print(header)
    values_by_metric = {m: [] for m, _ in metrics_to_summarize}
    for family, res in all_results.items():
        row = f"{family:<12}"
        for m, section in metrics_to_summarize:
            v = res[section][m]
            values_by_metric[m].append(v)
            row += f"{v:>22.4f}"
        print(row)

    print("-" * len(header))
    mean_row, std_row = f"{'mean':<12}", f"{'std':<12}"
    for m, _ in metrics_to_summarize:
        vals = values_by_metric[m]
        mean_row += f"{np.mean(vals):>22.4f}"
        std_row += f"{np.std(vals):>22.4f}"
    print(mean_row)
    print(std_row)
    print("=" * 70)

    print("\nOpen-set path decomposition per rotation (fractions):")
    for family, res in all_results.items():
        f = res["path_decomposition"]["fractions"]
        print(f"  {family:<12} A={f['A_correctly_novel']:.3f}  B={f['B_confident_known_attack']:.3f}  "
              f"B0={f['B0_absorbed_benign']:.3f}  C={f['C_reverted_known_attack']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
