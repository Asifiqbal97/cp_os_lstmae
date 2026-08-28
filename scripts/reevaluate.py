"""Recomputes metrics from a saved result.joblib without rerunning the full
pipeline. Use this any time evaluate.py changes.

Usage:
    python -m scripts.reevaluate --held-out-family Spoofing
"""

import argparse

import joblib

from src.evaluate import evaluate_mvp


def main(held_out_family: str):
    result = joblib.load(f"artifacts/{held_out_family}/result.joblib")
    metrics = evaluate_mvp(result, held_out_family)
    print(f"=== Recomputed metrics ({held_out_family}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out-family", required=True)
    args = parser.parse_args()
    main(args.held_out_family)
