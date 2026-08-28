"""Stage 1: native LightGBM Booster (not sklearn wrapper) with early stopping
on the calibration set, plus Gate 1: p-value Mondrian conformal gate.

Ported near-literally from cp_osr_lstmae_train_and_export.py's train_lgbm()
and calibrate_gate1(), so the artifacts load directly into
edge_gateway_service.py with no changes to that file.
"""

import lightgbm as lgb
import numpy as np

SEED = 42


def train_lgbm(split, num_class: int):
    dtrain = lgb.Dataset(split.X_tr_s, label=split.y_tr)
    dvalid = lgb.Dataset(split.X_cal_s, label=split.y_cal, reference=dtrain)
    params = {
        "objective": "multiclass", "num_class": num_class,
        "metric": "multi_logloss", "learning_rate": 0.05,
        "num_leaves": 63, "feature_fraction": 0.9,
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "seed": SEED, "verbosity": -1,
    }
    booster = lgb.train(
        params, dtrain, num_boost_round=800, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def calibrate_gate1(booster, split, label_encoder) -> dict:
    """Mondrian nonconformity scores keyed by CLASS NAME (string), exactly
    as edge_gateway_service.gate1_predict expects. Stores the full sorted
    score array (not a pre-computed threshold) so alpha can be varied
    post-hoc via conformal_pvalue() without recalibrating."""
    proba_cal = booster.predict(split.X_cal_s, num_iteration=booster.best_iteration)
    nonconf_by_class = {}
    for j, cls_name in enumerate(label_encoder.classes_):
        idx = split.y_cal == j
        nonconf_by_class[cls_name] = np.sort(1.0 - proba_cal[idx, j])
    return nonconf_by_class


def conformal_pvalue(score, calib_scores):
    """Identical to edge_gateway_service.conformal_pvalue."""
    n = len(calib_scores)
    n_ge = np.sum(calib_scores >= score)
    return (n_ge + 1) / (n + 1)


def gate1_predict(probs: np.ndarray, label_encoder, nonconf_by_class: dict, alpha: float):
    """Same logic as edge_gateway_service.gate1_predict, for a single row's
    probability vector. Returns (prediction_set, argmax_class, max_conf).

    FIX applied per paper fidelity (agreed with user): the symmetric gate
    defers BOTH empty and multi-element prediction sets -- not just empty,
    as the originally-provided edge_gateway_service.py did. This function
    only builds the prediction set; the caller (pipeline.py) decides
    singleton-exit vs defer using len(pred_set) == 1, closing the Path B
    evasion channel the paper describes.
    """
    argmax_idx = int(np.argmax(probs))
    argmax_cls = label_encoder.inverse_transform([argmax_idx])[0]
    max_conf = float(np.max(probs))

    pred_set = set()
    for j, cls in enumerate(label_encoder.classes_):
        score = 1.0 - probs[j]
        pval = conformal_pvalue(score, nonconf_by_class[cls])
        if pval > alpha:
            pred_set.add(cls)
    return pred_set, argmax_cls, max_conf


def gate1_predict_batch(probs_matrix: np.ndarray, label_encoder, nonconf_by_class: dict, alpha: float):
    """Vectorized-ish convenience wrapper over gate1_predict for a batch of
    rows (Python-level loop -- conformal_pvalue's per-class comparison
    against a variable-length calibration array doesn't vectorize cleanly
    across classes, and this only runs on Stage1's *deferred* subset in the
    pipeline, not the full test set)."""
    pred_sets, argmax_classes, confidences = [], [], []
    for row in probs_matrix:
        s, a, c = gate1_predict(row, label_encoder, nonconf_by_class, alpha)
        pred_sets.append(s)
        argmax_classes.append(a)
        confidences.append(c)
    return pred_sets, np.array(argmax_classes), np.array(confidences)
