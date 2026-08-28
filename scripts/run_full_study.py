"""Runs the full study: shared AE (once) -> closed-set LightGBM (once) ->
5 zero-day rotations (each reusing the shared AE).

Usage:
    python -m scripts.run_full_study --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import yaml

from src.data.loader import load_csv
from src.data.family_map import KNOWN_FAMILIES
from src.shared_ae import build_shared_ae
from src.closed_set import train_closed_set
from src.run_rotation import run_one_rotation
from src.report_saving import save_summary_report


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("[1/4] Loading data...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])
    benign_train = load_csv(cfg["benign_train_csv"])

    print("\n[2/4] Building shared AE (Gate 2, trained/calibrated once)...")
    shared_ae, shared_scaler, shared_gate2_bundle = build_shared_ae(benign_train, cfg)

    print("\n[3/4] Training closed-set LightGBM (no zero-day holdout)...")
    train_closed_set(train_pool, test_pool, cfg)

    print("\n[4/4] Running 5 zero-day rotations (reusing shared AE)...")
    all_results = {}
    for family in KNOWN_FAMILIES:
        print(f"\n\n{'#'*60}\n# ROTATION: {family}\n{'#'*60}")
        rotation_cfg = dict(cfg)
        rotation_cfg["held_out_family"] = family
        all_results[family] = run_one_rotation(
            train_pool, test_pool, family, rotation_cfg,
            shared_ae, shared_scaler, shared_gate2_bundle,
        )

    save_summary_report(Path("artifacts"), all_results)
    print("\nFull study complete. Summary saved to artifacts/summary.json and artifacts/summary.md")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
