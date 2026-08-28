"""Runs one zero-day rotation (whichever family configs/mvp.yaml specifies).
Builds the shared AE first (only needed once, but this script is a
convenience single-rotation entrypoint, so it pays that cost every time --
use run_full_study.py for the full study to avoid rebuilding it 5x).

Usage:
    python -m scripts.run_mvp --config configs/mvp.yaml
"""

import argparse

import yaml

from src.data.loader import load_csv
from src.shared_ae import build_shared_ae
from src.run_rotation import run_one_rotation


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("[1/3] Loading data...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])
    benign_train = load_csv(cfg["benign_train_csv"])

    print("[2/3] Building shared AE...")
    shared_ae, shared_scaler, shared_gate2_bundle = build_shared_ae(benign_train, cfg)

    print("[3/3] Running rotation...")
    run_one_rotation(train_pool, test_pool, cfg["held_out_family"], cfg,
                      shared_ae, shared_scaler, shared_gate2_bundle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
