"""Core logic for one zero-day rotation. Gate 2 (AE) is now SHARED across
all rotations (see src/shared_ae.py) -- this module only trains/calibrates
Gate 1 (rotation-specific) and runs the pipeline reusing the shared AE.
"""

import random
from pathlib import Path

import joblib
import numpy as np
import torch

from .data.zero_day_split import make_zero_day_split
from .stage1_classifier.train import train_lgbm, calibrate_gate1
from .stage2_anomaly.clustering import LeaderFollowerClusterer
from .pipeline import run_pipeline
from .evaluate import evaluate_mvp, path_decomposition, gate1_raw_report, gate1_coverage_report
from .report_saving import save_rotation_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_rotation(train_pool, test_pool, held_out_family: str, cfg: dict,
                      shared_ae, shared_scaler, shared_gate2_bundle,
                      artifacts_root: str = "artifacts", verbose: bool = True):
    """shared_ae / shared_scaler / shared_gate2_bundle come from
    src/shared_ae.py's build_shared_ae() -- trained/calibrated ONCE, reused
    across every rotation."""
    set_seed(cfg["random_state"])

    artifacts_dir = Path(artifacts_root) / held_out_family
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg)

    log(f"Building zero-day split (held out: {held_out_family})...")
    split = make_zero_day_split(train_pool, test_pool, held_out_family,
                                 random_state=cfg["random_state"])
    log(f"       known_classes={split.known_classes}  "
        f"train={len(split.X_tr)}  calib={len(split.X_cal)}  test={len(split.X_te)}")

    num_class = len(split.label_encoder.classes_)
    log(f"Training Gate 1 (LightGBM, {num_class} known classes)...")
    booster = train_lgbm(split, num_class)
    joblib.dump({"model": booster, "label_encoder": split.label_encoder},
                artifacts_dir / "lightgbm_known_iomt.pkl", compress=3)

    log(f"Calibrating Gate 1 Mondrian gate (alpha={cfg['alpha_stage1']})...")
    nonconf_by_class = calibrate_gate1(booster, split, split.label_encoder)
    joblib.dump({"nonconf_by_class": nonconf_by_class, "alpha_1": cfg["alpha_stage1"],
                 "feature_cols": split.feat_cols},
                artifacts_dir / "lgb_conformal_calibration.pkl", compress=3)

    log("Running pipeline on test set (Gate1 -> shared Gate2 -> Gate3)...")
    # Fresh clusterer per rotation (cluster assignments are rotation-specific
    # -- which flows get deferred depends on this rotation's Gate 1), but
    # reuses the shared Gate 2 calibration's cluster_radius.
    clusterer = LeaderFollowerClusterer(shared_gate2_bundle["cluster_radius"], cfg["min_cluster_size"])
    result = run_pipeline(
        split.X_te, split.y_te, booster, split.label_encoder, nonconf_by_class,
        cfg["alpha_stage1"], split.scaler, shared_ae, shared_scaler, shared_gate2_bundle,
        clusterer, seq_len=cfg["seq_len"],
    )
    joblib.dump(result, artifacts_dir / "result.joblib", compress=3)

    log("Evaluating...")
    pipeline_metrics = evaluate_mvp(result, held_out_family)
    gate1_raw = gate1_raw_report(result, held_out_family)
    gate1_coverage = gate1_coverage_report(result, split.known_classes, cfg["alpha_stage1"])
    paths = path_decomposition(result, held_out_family)

    save_rotation_report(artifacts_dir, held_out_family, gate1_raw, gate1_coverage, pipeline_metrics, paths, cfg)
    log(f"       report saved to {artifacts_dir}/metrics.json and {artifacts_dir}/report.md")

    if verbose:
        print_rotation_report(held_out_family, gate1_raw, gate1_coverage, pipeline_metrics, paths, cfg)

    return {
        "held_out_family": held_out_family,
        "pipeline_metrics": pipeline_metrics,
        "gate1_raw": gate1_raw,
        "gate1_coverage": gate1_coverage,
        "path_decomposition": paths,
    }


def print_rotation_report(held_out_family, gate1_raw, gate1_coverage, pipeline_metrics, paths, cfg):
    print(f"\n{'='*60}\nRESULTS -- held out: {held_out_family}\n{'='*60}")

    print("\n--- GATE 1 (LightGBM + Mondrian) ---")
    print(f"raw classifier accuracy (bypassing conformal gate): {gate1_raw['stage1_raw_accuracy']:.4f}")
    print(f"raw classifier macro-F1:                            {gate1_raw['stage1_raw_macro_f1']:.4f}")
    print(f"singleton rate:      {gate1_raw['singleton_rate']:.4f}")
    print(f"empty-set rate:      {gate1_raw['empty_set_rate']:.4f}")
    print(f"multi-element rate:  {gate1_raw['multi_element_set_rate']:.4f}")
    print(f"mean prediction set size: {gate1_raw['mean_prediction_set_size']:.3f}")
    print(f"\nper-class coverage validation (target={1-cfg['alpha_stage1']:.2f}):")
    print(f"{'class':<12} {'n':>10} {'achieved':>10} {'target':>8} {'meets?':>7}")
    for row in gate1_coverage:
        print(f"{row['class']:<12} {row['n']:>10} {row['achieved_coverage']:>10.4f} "
              f"{row['target_coverage']:>8.2f} {'yes' if row['meets_target'] else 'NO':>7}")

    print("\n--- GATE 2 (shared LSTM-AE + split-conformal) ---")
    print(f"benign false alarm rate: {pipeline_metrics['benign_false_alarm_rate']:.4f} "
          f"(target alpha={cfg['alpha_stage2']:.2f})")
    print(f"zero-day recall:         {pipeline_metrics['zero_day_recall']:.4f}")

    print(f"\n--- OPEN-SET PATH DECOMPOSITION (n={paths['n_zero_day_rows']} zero-day rows) ---")
    for k in ["A_correctly_novel", "B_confident_known_attack", "B0_absorbed_benign", "C_reverted_known_attack"]:
        print(f"{k:<28} count={paths['counts'][k]:>8}  fraction={paths['fractions'][k]:.4f}")

    print("\n--- FULL PIPELINE (final verdict, all gates combined) ---")
    print(f"known_class_accuracy: {pipeline_metrics['known_class_accuracy']:.4f}")
    print(f"known_class_macro_f1: {pipeline_metrics['known_class_macro_f1']:.4f}")
    print(f"path_counts: {pipeline_metrics['path_counts']}")
    print(f"{'='*60}\n")
