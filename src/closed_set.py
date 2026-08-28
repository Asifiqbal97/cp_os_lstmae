"""M6: closed-set LightGBM baseline -- all 5 known families in training,
no zero-day holdout. Reports the paper's own "closed-set macro-F1 under
stratified splits" requirement. No conformal gate, no AE, no Gate 3 --
this is a plain classifier evaluation, per the paper's literal wording.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data.loader import FEATURE_COLUMNS
from .stage1_classifier.train import SEED
from .report_saving import save_closed_set_report


def train_closed_set(train_pool, test_pool, cfg: dict, artifacts_root: str = "artifacts",
                      verbose: bool = True):
    def log(msg):
        if verbose:
            print(msg)

    artifacts_dir = Path(artifacts_root) / "known_closed_set"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log("Training closed-set LightGBM (all 5 known families, no holdout)...")
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(train_pool["family"].values)
    X_all = train_pool[FEATURE_COLUMNS].values.astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=cfg["random_state"]
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    dtrain = lgb.Dataset(X_tr_s, label=y_tr)
    dvalid = lgb.Dataset(X_val_s, label=y_val, reference=dtrain)
    params = {
        "objective": "multiclass", "num_class": len(label_encoder.classes_),
        "metric": "multi_logloss", "learning_rate": 0.05,
        "num_leaves": 63, "feature_fraction": 0.9,
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "seed": SEED, "verbosity": -1,
    }
    booster = lgb.train(
        params, dtrain, num_boost_round=800, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    joblib.dump({"model": booster, "label_encoder": label_encoder, "scaler": scaler},
                artifacts_dir / "lightgbm_known_iomt.pkl", compress=3)

    log("Evaluating on official merged_test_labelled.csv...")
    X_te = test_pool[FEATURE_COLUMNS].values.astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    y_te = label_encoder.transform(test_pool["family"].values)
    proba_te = booster.predict(X_te_s, num_iteration=booster.best_iteration)
    y_pred = proba_te.argmax(axis=1)

    report = classification_report(
        y_te, y_pred, labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_te, y_pred, labels=range(len(label_encoder.classes_)))

    save_closed_set_report(artifacts_dir, report, cm, label_encoder.classes_)
    log(f"       report saved to {artifacts_dir}/metrics.json and {artifacts_dir}/report.md")
    log(f"       macro avg F1: {report['macro avg']['f1-score']:.4f}")

    return booster, label_encoder, scaler, report, cm
