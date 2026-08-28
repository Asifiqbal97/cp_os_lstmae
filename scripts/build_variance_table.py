"""Builds the paper-style summary table (per-class LightGBM prec/rec/F1,
AE mean-recon-error/detection-rate/ROC-AUC, Hybrid detection rate) WITH
variance estimates -- using the lower-compute hybrid approach:

  - LightGBM columns: k-fold stratified cross-validation on the CLOSED-SET
    model (all 5 known families + Benign, no zero-day holdout). Matches the
    paper's "stratified splits" wording most directly. Real added compute:
    trains LightGBM k times instead of once.
  - AE / Hybrid columns: bootstrap resampling of each rotation's ALREADY-
    COMPUTED per-row results (raw AE scores + result.joblib's saved
    pipeline verdicts). No retraining at all -- just resampling with
    replacement, cheap and fast.

Self-contained: does not modify or depend on any src/ file beyond what's
already public (load_csv, load_shared_ae_artifacts, load_rotation_artifacts,
predict_recon_latent, build_sequences, dual_score, make_zero_day_split).

Usage:
    python -m scripts.build_variance_table --config configs/mvp.yaml
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

from src.data.loader import load_csv, FEATURE_COLUMNS
from src.data.family_map import KNOWN_FAMILIES, BENIGN_LABEL
from src.data.zero_day_split import make_zero_day_split
from src.analysis import load_shared_ae_artifacts, load_rotation_artifacts
from src.stage2_anomaly.lstm_ae import predict_recon_latent
from src.stage2_anomaly.sequences import build_sequences
from src.stage2_anomaly.conformal_gate import dual_score
from src.stage1_classifier.train import SEED

N_FOLDS = 5
N_BOOTSTRAP = 200


# ----------------------------------------------------------- LightGBM k-fold

def lgbm_kfold_report(train_pool, cfg):
    print(f"[LightGBM] {N_FOLDS}-fold stratified CV (all known classes, no holdout)...")
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(train_pool["family"].values)
    X_all = train_pool[FEATURE_COLUMNS].values.astype(np.float32)
    classes = list(label_encoder.classes_)

    per_fold_reports = []
    per_fold_overall = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg["random_state"])

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"  fold {fold+1}/{N_FOLDS}...")
        # Early-stopping validation split, carved from this fold's TRAINING
        # portion only -- never the fold's held-out test portion, which
        # would leak test information into model selection.
        fold_tr_idx, fold_val_idx = train_test_split(
            tr_idx, test_size=0.2, stratify=y_all[tr_idx], random_state=cfg["random_state"]
        )

        scaler = StandardScaler().fit(X_all[fold_tr_idx])
        X_tr_s = scaler.transform(X_all[fold_tr_idx]).astype(np.float32)
        X_val_s = scaler.transform(X_all[fold_val_idx]).astype(np.float32)
        X_te_s = scaler.transform(X_all[te_idx]).astype(np.float32)

        dtrain = lgb.Dataset(X_tr_s, label=y_all[fold_tr_idx])
        dvalid = lgb.Dataset(X_val_s, label=y_all[fold_val_idx], reference=dtrain)
        params = {
            "objective": "multiclass", "num_class": len(classes),
            "metric": "multi_logloss", "learning_rate": 0.05,
            "num_leaves": 63, "feature_fraction": 0.9,
            "bagging_fraction": 0.9, "bagging_freq": 1,
            "seed": SEED, "verbosity": -1,
        }
        # Same config as train_lgbm() in stage1_classifier/train.py -- 800
        # rounds with early stopping, not a fixed round count with no
        # convergence check (that was the bug: fixed 300 rounds, no early
        # stopping, produced underfit/unstable folds especially for
        # minority classes).
        booster = lgb.train(
            params, dtrain, num_boost_round=800, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred = booster.predict(X_te_s, num_iteration=booster.best_iteration).argmax(axis=1)
        report = classification_report(y_all[te_idx], y_pred, labels=range(len(classes)),
                                        target_names=classes, output_dict=True, zero_division=0)
        per_fold_reports.append(report)
        per_fold_overall.append({"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]})

    def agg(metric_path):
        vals = [r[metric_path[0]][metric_path[1]] if len(metric_path) == 2 else r[metric_path[0]]
                for r in per_fold_reports]
        return float(np.mean(vals)), float(np.std(vals))

    per_class = {}
    for cls in classes:
        p_mean, p_std = agg((cls, "precision"))
        r_mean, r_std = agg((cls, "recall"))
        f_mean, f_std = agg((cls, "f1-score"))
        per_class[cls] = {"precision": (p_mean, p_std), "recall": (r_mean, r_std), "f1": (f_mean, f_std)}

    acc_vals = [o["accuracy"] for o in per_fold_overall]
    f1_vals = [o["macro_f1"] for o in per_fold_overall]
    overall = {"accuracy": (float(np.mean(acc_vals)), float(np.std(acc_vals))),
               "macro_f1": (float(np.mean(f1_vals)), float(np.std(f1_vals)))}

    return per_class, overall


# ----------------------------------------------------- AE/Hybrid bootstrap

def raw_ae_components(df, stage2_model, stage2_scaler, gate2_bundle, seq_len):
    """MSE and latent-centroid-deviation separately (analysis.py's
    compute_dual_scores only returns the combined score)."""
    X = stage2_scaler.transform(df[FEATURE_COLUMNS].values.astype(np.float32)).astype(np.float32)
    seqs = build_sequences(X, seq_len=seq_len)
    if len(seqs) == 0:
        return np.array([]), np.array([])
    recon, z = predict_recon_latent(stage2_model, seqs)
    mse = np.mean(np.square(seqs - recon), axis=(1, 2))
    ldev = np.linalg.norm(z - gate2_bundle["centroid"], axis=1)
    return mse, ldev


def bootstrap_mean_std(values, statistic_fn, n_bootstrap=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stats = [statistic_fn(values[rng.integers(0, n, n)]) for _ in range(n_bootstrap)]
    return float(np.mean(stats)), float(np.std(stats))


def bootstrap_two_sample(a, b, statistic_fn, n_bootstrap=N_BOOTSTRAP, seed=0):
    """statistic_fn(a_resampled, b_resampled) -> scalar, e.g. ROC-AUC."""
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    stats = [statistic_fn(a[rng.integers(0, na, na)], b[rng.integers(0, nb, nb)]) for _ in range(n_bootstrap)]
    return float(np.mean(stats)), float(np.std(stats))


def ae_hybrid_bootstrap(test_pool, cfg, stage2_model, stage2_scaler, gate2_bundle, seq_len):
    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    mse_benign, _ = raw_ae_components(benign_rows, stage2_model, stage2_scaler, gate2_bundle, seq_len)
    benign_mean_mse = float(mse_benign.mean())

    results = {}
    pooled_benign_hybrid_flags = []

    for family in KNOWN_FAMILIES:
        print(f"[AE/Hybrid bootstrap] {family}...")
        zd_rows = test_pool[test_pool["family"] == family]
        mse_zd, ldev_zd = raw_ae_components(zd_rows, stage2_model, stage2_scaler, gate2_bundle, seq_len)
        _, ldev_benign = raw_ae_components(benign_rows, stage2_model, stage2_scaler, gate2_bundle, seq_len)

        dual_zd = dual_score(mse_zd, ldev_zd, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])
        dual_benign = dual_score(mse_benign, ldev_benign, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])
        tau = gate2_bundle["threshold"]

        # Mean reconstruction error (x benign mean)
        ratio_mean, ratio_std = bootstrap_mean_std(mse_zd, lambda v: v.mean() / benign_mean_mse)

        # AE detection rate @ tau
        dr_mean, dr_std = bootstrap_mean_std(dual_zd, lambda v: (v > tau).mean())

        # AE ROC-AUC (benign vs this family)
        def auc_stat(zd_sample, benign_sample):
            y = np.concatenate([np.ones(len(zd_sample)), np.zeros(len(benign_sample))])
            s = np.concatenate([zd_sample, benign_sample])
            return roc_auc_score(y, s)
        auc_mean, auc_std = bootstrap_two_sample(dual_zd, dual_benign, auc_stat)

        # Hybrid detection rate: bootstrap over the ACTUAL full-pipeline
        # verdicts already saved in result.joblib (no retraining -- just
        # resampling which rows we look at).
        result = joblib.load(Path("artifacts") / family / "result.joblib")
        fam_arr = np.asarray(result["family"])
        verdict_arr = np.asarray(result["verdict"])
        path_arr = np.asarray(result["path"])
        scored = path_arr != "buffering_incomplete"
        zd_mask = scored & (fam_arr == family)
        is_novel = np.vectorize(lambda v: v == "Zero-day-Unclustered" or
                                 (isinstance(v, str) and v.startswith("Candidate-Class-")))(verdict_arr[zd_mask])
        hybrid_mean, hybrid_std = bootstrap_mean_std(is_novel.astype(float), lambda v: v.mean())

        benign_mask = scored & (fam_arr == BENIGN_LABEL)
        benign_is_novel = np.vectorize(lambda v: v == "Zero-day-Unclustered" or
                                        (isinstance(v, str) and v.startswith("Candidate-Class-")))(verdict_arr[benign_mask])
        pooled_benign_hybrid_flags.append(benign_is_novel.astype(float))

        results[family] = {
            "mean_recon_error_ratio": (ratio_mean, ratio_std),
            "ae_detection_rate": (dr_mean, dr_std),
            "ae_roc_auc": (auc_mean, auc_std),
            "hybrid_detection_rate": (hybrid_mean, hybrid_std),
        }

    # Benign row: pooled across all 5 rotations' benign subsets (benign
    # behavior shouldn't systematically differ by which family was held out)
    pooled = np.concatenate(pooled_benign_hybrid_flags)
    benign_hybrid_mean, benign_hybrid_std = bootstrap_mean_std(pooled, lambda v: v.mean())
    results["Benign"] = {
        "mean_recon_error_ratio": (1.0, 0.0),  # benign vs itself, trivially 1.0
        "ae_detection_rate": (float((dual_benign > tau).mean()), float("nan")),  # benign FPR@tau, not bootstrapped per-family (uses last loop's dual_benign)
        "ae_roc_auc": (float("nan"), float("nan")),  # not meaningful for benign-vs-benign
        "hybrid_detection_rate": (benign_hybrid_mean, benign_hybrid_std),
    }
    return results


# --------------------------------------------------------------- reporting

def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])

    lgbm_per_class, lgbm_overall = lgbm_kfold_report(train_pool, cfg)

    print("Loading shared AE artifacts...")
    stage2_model, stage2_scaler, gate2_bundle, seq_len = load_shared_ae_artifacts()
    ae_hybrid = ae_hybrid_bootstrap(test_pool, cfg, stage2_model, stage2_scaler, gate2_bundle, seq_len)

    rows = []
    for cls in KNOWN_FAMILIES + [BENIGN_LABEL]:
        lg = lgbm_per_class.get(cls, {"precision": (float("nan"),) * 2, "recall": (float("nan"),) * 2, "f1": (float("nan"),) * 2})
        ae = ae_hybrid[cls]
        rows.append({
            "class": cls,
            "lgbm_precision": lg["precision"], "lgbm_recall": lg["recall"], "lgbm_f1": lg["f1"],
            "mean_recon_error_ratio": ae["mean_recon_error_ratio"],
            "ae_detection_rate": ae["ae_detection_rate"],
            "ae_roc_auc": ae["ae_roc_auc"],
            "hybrid_detection_rate": ae["hybrid_detection_rate"],
        })

    print_and_save(rows, lgbm_overall)


def _fmt(pair):
    m, s = pair
    if np.isnan(m):
        return "n/a"
    if np.isnan(s):
        return f"{m:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def print_and_save(rows, lgbm_overall):
    print("\n" + "=" * 100)
    print(f"{'Class':<14}{'LGBM Prec':<16}{'LGBM Rec':<16}{'LGBM F1':<16}"
          f"{'ReconRatio':<16}{'AE DR@tau':<16}{'AE AUC':<16}{'Hybrid DR':<16}")
    print("-" * 100)
    for r in rows:
        print(f"{r['class']:<14}{_fmt(r['lgbm_precision']):<16}{_fmt(r['lgbm_recall']):<16}"
              f"{_fmt(r['lgbm_f1']):<16}{_fmt(r['mean_recon_error_ratio']):<16}"
              f"{_fmt(r['ae_detection_rate']):<16}{_fmt(r['ae_roc_auc']):<16}{_fmt(r['hybrid_detection_rate']):<16}")
    print("-" * 100)
    print(f"Overall LightGBM accuracy: {_fmt(lgbm_overall['accuracy'])}   "
          f"macro-F1: {_fmt(lgbm_overall['macro_f1'])}")
    print("=" * 100)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    def jsonable(pair):
        return {"mean": pair[0], "std": pair[1]}

    bundle = {
        "rows": [{**{k: v for k, v in r.items() if k == "class"},
                  **{k: jsonable(v) for k, v in r.items() if k != "class"}} for r in rows],
        "lgbm_overall": {k: jsonable(v) for k, v in lgbm_overall.items()},
        "n_folds": N_FOLDS, "n_bootstrap": N_BOOTSTRAP,
    }
    with open(out_dir / "variance_table.json", "w") as f:
        json.dump(bundle, f, indent=2)

    md = ["# CP-OSR-LSTMAE — Variance Table\n",
          f"LightGBM columns: {N_FOLDS}-fold stratified CV. AE/Hybrid columns: {N_BOOTSTRAP}-iteration bootstrap "
          "over already-computed results (no retraining).\n",
          "| Class | LGBM Prec | LGBM Rec | LGBM F1 | Recon Ratio (×benign) | AE DR@τ | AE ROC-AUC | Hybrid DR |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['class']} | {_fmt(r['lgbm_precision'])} | {_fmt(r['lgbm_recall'])} | "
                  f"{_fmt(r['lgbm_f1'])} | {_fmt(r['mean_recon_error_ratio'])} | "
                  f"{_fmt(r['ae_detection_rate'])} | {_fmt(r['ae_roc_auc'])} | {_fmt(r['hybrid_detection_rate'])} |")
    md.append(f"\n**Overall LightGBM accuracy:** {_fmt(lgbm_overall['accuracy'])}  "
              f"**macro-F1:** {_fmt(lgbm_overall['macro_f1'])}\n")
    md.append("\nNote: 'Benign' row's AE ROC-AUC is not meaningful (benign-vs-benign) and reported as n/a. "
              "'AE DR@τ' for Benign is the false positive rate at the calibrated threshold, not bootstrapped "
              "per-iteration (single-pass value only).\n")

    with open(out_dir / "variance_table.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSaved: {out_dir}/variance_table.md, {out_dir}/variance_table.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
