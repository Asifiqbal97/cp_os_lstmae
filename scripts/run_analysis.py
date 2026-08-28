"""Read-only analysis pass over already-completed rotations: coverage
validation (Mondrian vs marginal-CP vs MSP with Beta tolerance bands),
Gate 1 ROC, Gate 2 ROC. No retraining -- reuses artifacts from a completed
scripts/run_full_study.py run.

Usage:
    python -m scripts.run_analysis --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import yaml

from src.data.loader import load_csv
from src.data.family_map import KNOWN_FAMILIES
from src.analysis import (
    load_rotation_artifacts, load_shared_ae_artifacts, rebuild_split_and_scores,
    coverage_validation, gate1_roc, gate2_roc,
)
from src.analysis_plots import plot_coverage_validation, plot_gate1_roc, plot_gate2_roc
from src.report_saving import save_analysis_report


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("[1/3] Loading data...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])

    print("[2/3] Loading shared AE artifacts...")
    stage2_model, stage2_scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()

    print("[3/3] Running analysis per rotation...")
    for family in KNOWN_FAMILIES:
        print(f"\n{'='*50}\nAnalyzing: {family}\n{'='*50}")
        artifacts_dir = Path("artifacts") / family
        plots_dir = artifacts_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        booster, label_encoder, nonconf_by_class, alpha1 = load_rotation_artifacts(family)
        split, proba_te, proba_cal = rebuild_split_and_scores(train_pool, test_pool, family, booster, cfg)

        print("  computing coverage validation (Mondrian vs marginal-CP vs MSP)...")
        cov_result = coverage_validation(split, proba_te, proba_cal, alpha1, nonconf_by_class)
        plot_coverage_validation(cov_result, alpha1, plots_dir / "coverage_validation.png")

        print("  computing Gate 1 ROC...")
        g1_curves = gate1_roc(proba_te, split.y_te, label_encoder)
        plot_gate1_roc(g1_curves, plots_dir / "gate1_roc.png")

        print("  computing Gate 2 ROC...")
        g2_roc = gate2_roc(test_pool, family, stage2_model, stage2_scaler, gate2_bundle,
                            split.feat_cols, seq_len)
        plot_gate2_roc(g2_roc, family, plots_dir / "gate2_roc.png")

        save_analysis_report(artifacts_dir, family, cov_result, g1_curves, g2_roc, alpha1)
        print(f"  saved: {artifacts_dir}/analysis_report.md, analysis_metrics.json, plots/*.png")

    print("\nAnalysis complete for all rotations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
