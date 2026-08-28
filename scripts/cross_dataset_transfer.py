"""Cross-dataset transfer: CICIoT2023 (Mirai + Benign) scored through the
EXISTING trained pipeline. No retraining -- pure inference test.

Schema: 39/46 features match by name exactly; 7 missing (Duration, Srate,
Drate, Magnitue, Radius, Covariance, Weight) are zero-filled, disclosed
explicitly per user's decision -- not hidden.

Read-only w.r.t. artifacts/ -- loads existing Gate 1 (Spoofing rotation)
+ shared AE, writes only to artifacts_cross_dataset/.

Usage:
    python -m scripts.cross_dataset_transfer --config configs/mvp.yaml
"""

import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.loader import FEATURE_COLUMNS
from src.analysis import load_shared_ae_artifacts
from src.stage2_anomaly.clustering import LeaderFollowerClusterer
from src.pipeline import run_pipeline

ARTIFACTS_ROOT = "artifacts_cross_dataset"
CROSS_DATASET_DIR = "data/raw/cross_dataset"

MISSING_FEATURES = ["Duration", "Srate", "Drate", "Magnitue", "Radius", "Covariance", "Weight"]


# def load_ciciot2023(csv_dir: str) -> pd.DataFrame:
#     rows = []
#     for path in glob.glob(f"{csv_dir}/*.csv"):
#         df = pd.read_csv(path)
#         family = "Mirai" if "Mirai" in path else "Benign"
#         for col in MISSING_FEATURES:
#             df[col] = 0.0  # disclosed zero-fill, not silent
#         missing_in_file = [c for c in FEATURE_COLUMNS if c not in df.columns]
#         for c in missing_in_file:
#             df[c] = 0.0
#         df["family"] = family
#         rows.append(df[FEATURE_COLUMNS + ["family"]])
#         print(f"  loaded {path}: {len(df)} rows, family={family}")
#     return pd.concat(rows, ignore_index=True)

def load_ciciot2023(csv_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(f"{csv_dir}/*.csv"):
        df = pd.read_csv(path)
        family = "Mirai" if "Mirai" in path else "Benign"
        for col in MISSING_FEATURES:
            df[col] = 0.0  # disclosed zero-fill, not silent
        missing_in_file = [c for c in FEATURE_COLUMNS if c not in df.columns]
        for c in missing_in_file:
            df[c] = 0.0
        df["family"] = family
        rows.append(df[FEATURE_COLUMNS + ["family"]])
        print(f"  loaded {path}: {len(df)} rows, family={family}")
    combined = pd.concat(rows, ignore_index=True)

    # CICIoT2023's own Rate/Srate/Drate columns contain inf for near-zero-
    # duration flows (their own computation, not our zero-fill) -- disclosed
    # sanitization, not silent.
    n_before = combined[FEATURE_COLUMNS].isin([float("inf"), float("-inf")]).sum().sum()
    combined[FEATURE_COLUMNS] = combined[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], 0.0)
    combined[FEATURE_COLUMNS] = combined[FEATURE_COLUMNS].fillna(0.0)
    print(f"  sanitized {n_before} inf values (CICIoT2023's own Rate/Srate/Drate near-zero-duration artifact)")

    return combined


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(ARTIFACTS_ROOT)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading CICIoT2023 (zero-filling 7/46 missing features, disclosed)...")
    df = load_ciciot2023(CROSS_DATASET_DIR)
    print(f"  total: {len(df)} rows  ({(df['family']=='Mirai').sum()} Mirai, "
          f"{(df['family']=='Benign').sum()} Benign)")

    print("[2/4] Loading EXISTING Gate 1 (Spoofing rotation's model -- has Mondrian calibration; "
          "closed-set model doesn't, by design -- read-only, no retraining)...")
    lgb_bundle = joblib.load("artifacts/Spoofing/lightgbm_known_iomt.pkl")
    booster, label_encoder = lgb_bundle["model"], lgb_bundle["label_encoder"]
    conf = joblib.load("artifacts/Spoofing/lgb_conformal_calibration.pkl")
    nonconf_by_class, alpha1 = conf["nonconf_by_class"], conf["alpha_1"]
    # Stage-1 scaler isn't persisted separately -- rebuild deterministically
    # (same seed -> identical scaler, same pattern used throughout this project).
    from src.data.loader import load_csv
    from src.data.zero_day_split import make_zero_day_split
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])
    stage1_scaler = make_zero_day_split(train_pool, test_pool, "Spoofing",
                                         random_state=cfg["random_state"]).scaler

    print("[3/4] Loading EXISTING shared AE (read-only, no retraining)...")
    stage2_model, stage2_scaler, gate2_bundle, seq_len = load_shared_ae_artifacts(artifacts_root="artifacts")

    print("[4/4] Running pipeline (existing Gate1 -> existing Gate2 -> Gate3)...")
    X_raw = df[FEATURE_COLUMNS].values.astype(np.float32)
    y_family = df["family"].values
    clusterer = LeaderFollowerClusterer(gate2_bundle["cluster_radius"], cfg["min_cluster_size"])
    result = run_pipeline(
        X_raw, y_family, booster, label_encoder, nonconf_by_class, alpha1,
        stage1_scaler, stage2_model, stage2_scaler, gate2_bundle, clusterer, seq_len=seq_len,
    )
    joblib.dump(result, artifacts_dir / "result.joblib", compress=3)

    verdict = np.asarray(result["verdict"])
    path = np.asarray(result["path"])
    scored = path != "buffering_incomplete"

    def is_novel(v):
        return v == "Zero-day-Unclustered" or (isinstance(v, str) and v.startswith("Candidate-Class-"))
    is_novel_arr = np.vectorize(is_novel)(verdict[scored])
    fam_s = y_family[scored]

    mirai_recall = float(is_novel_arr[fam_s == "Mirai"].mean()) if (fam_s == "Mirai").any() else float("nan")
    benign_far = float(is_novel_arr[fam_s == "Benign"].mean()) if (fam_s == "Benign").any() else float("nan")

    print(f"\nMirai zero-day recall: {mirai_recall:.4f}")
    print(f"Cross-dataset benign FAR: {benign_far:.4f}")

    with open(artifacts_dir / "result.md", "w") as f:
        f.write("# Cross-Dataset Transfer: CICIoT2023 (Mirai + Benign)\n\n")
        f.write(f"7/46 features zero-filled (disclosed): {MISSING_FEATURES}\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Mirai zero-day recall | {mirai_recall:.4f} |\n")
        f.write(f"| Cross-dataset benign FAR | {benign_far:.4f} |\n")
        f.write(f"| Mirai rows | {int((fam_s=='Mirai').sum())} |\n")
        f.write(f"| Benign rows | {int((fam_s=='Benign').sum())} |\n")
        f.write("\nUses EXISTING Gate 1 (Spoofing rotation) + shared AE, no retraining. "
                "Feature-schema gap disclosed above, not hidden.\n")

    print(f"\nSaved: {artifacts_dir}/result.md, {artifacts_dir}/result.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
