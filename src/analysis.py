"""Read-only analysis over already-trained artifacts -- no retraining.

Reloads a rotation's saved LightGBM + shared AE, recomputes raw scores
needed for:
  - MSP baseline (M2): simple max-softmax threshold, no conformal guarantee
  - Marginal-CP ablation (M3): one global conformal threshold, not per-class
  - Coverage validation (M4): achieved vs nominal coverage, Mondrian vs
    marginal-CP vs MSP, with Beta finite-sample tolerance bands
  - Gate 1 ROC (M5): one-vs-rest per known class
  - Gate 2 ROC (M6): benign vs held-out zero-day family, dual anomaly score
    swept across thresholds (not just the one fixed calibrated threshold)
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_curve, auc

from .data.loader import load_csv
from .data.zero_day_split import make_zero_day_split
from .data.family_map import BENIGN_LABEL
from .stage2_anomaly.lstm_ae import LSTMAutoencoder, predict_recon_latent
from .stage2_anomaly.conformal_gate import dual_score
from .stage2_anomaly.sequences import build_sequences


# ---------------------------------------------------------------- M1: reload

def load_rotation_artifacts(held_out_family: str, artifacts_root: str = "artifacts"):
    d = Path(artifacts_root) / held_out_family
    lgb_bundle = joblib.load(d / "lightgbm_known_iomt.pkl")
    conf_bundle = joblib.load(d / "lgb_conformal_calibration.pkl")
    return lgb_bundle["model"], lgb_bundle["label_encoder"], conf_bundle["nonconf_by_class"], conf_bundle["alpha_1"]


def load_shared_ae_artifacts(artifacts_root: str = "artifacts"):
    d = Path(artifacts_root) / "shared_ae"
    ckpt = torch.load(d / "stage2_model.pt", weights_only=False)
    model = LSTMAutoencoder(ckpt["n_features"], ckpt["seq_len"], ckpt["hidden_dim"], ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    scaler = joblib.load(d / "benign_scaler.pkl")
    gate2_bundle = joblib.load(d / "cp_osr_ae_threshold.pkl")
    return model, scaler, gate2_bundle, ckpt["seq_len"]


def rebuild_split_and_scores(train_pool, test_pool, held_out_family, booster, cfg):
    """Rebuilds the same deterministic zero-day split used at training time
    (same seed -> identical split, no retraining), then computes raw Gate 1
    softmax probabilities on both calib and test sets."""
    split = make_zero_day_split(train_pool, test_pool, held_out_family,
                                 random_state=cfg["random_state"])
    proba_te = booster.predict(split.X_te_s, num_iteration=booster.best_iteration)
    proba_cal = booster.predict(split.X_cal_s, num_iteration=booster.best_iteration)
    return split, proba_te, proba_cal


# ---------------------------------------------------------------- M2: MSP

def msp_calibrate(proba_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Simple max-softmax threshold (Hendrycks & Gimpel baseline), no
    conformal guarantee. Calibrated so the ERROR rate among ACCEPTED
    calibration points is <= alpha (selective-classification style), unlike
    Mondrian/marginal-CP which control miscoverage of the true label
    directly. This asymmetry is inherent to comparing set-valued conformal
    output against a single-label heuristic baseline."""
    max_prob = proba_cal.max(axis=1)
    argmax = proba_cal.argmax(axis=1)
    correct = (argmax == y_cal)

    order = np.argsort(-max_prob)
    sorted_correct, sorted_maxprob = correct[order], max_prob[order]
    cum_error_rate = np.cumsum(~sorted_correct) / np.arange(1, len(sorted_correct) + 1)

    valid = np.where(cum_error_rate <= alpha)[0]
    return float(sorted_maxprob[valid[-1]]) if len(valid) else 1.1  # 1.1 = reject everything


def msp_covered(proba_te: np.ndarray, y_te_encoded: np.ndarray, threshold: float) -> np.ndarray:
    max_prob = proba_te.max(axis=1)
    argmax = proba_te.argmax(axis=1)
    return (max_prob >= threshold) & (argmax == y_te_encoded)


# ---------------------------------------------------------- M3: marginal-CP

def marginal_cp_calibrate(proba_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Same conformal p-value machinery as Mondrian, but ONE global
    threshold pooled across all classes instead of per-class -- isolates
    what the Mondrian (class-conditional) construction actually buys."""
    scores = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q, method="higher"))


def marginal_cp_covered(proba_te: np.ndarray, y_te_encoded: np.ndarray, threshold: float) -> np.ndarray:
    scores = 1.0 - proba_te[np.arange(len(y_te_encoded)), y_te_encoded]
    return scores <= threshold


# ---------------------------------------------------------- M4: coverage validation

def beta_tolerance_band(n: int, alpha: float, tail: float = 0.05):
    """Finite-sample tolerance band on achieved coverage for a calibration
    set of size n at significance alpha (Vovk 2012): miscoverage ~
    Beta(l, n+1-l) where l = ceil((n+1)*alpha). Returns (lower, upper) on
    the COVERAGE scale (1 - miscoverage), at the given two-sided tail."""
    if n == 0:
        return (float("nan"), float("nan"))
    l = max(int(np.ceil((n + 1) * alpha)), 1)
    a, b = l, n + 1 - l
    m_lo = beta_dist.ppf(tail, a, b)
    m_hi = beta_dist.ppf(1 - tail, a, b)
    return (1 - m_hi, 1 - m_lo)


def coverage_validation(split, proba_te, proba_cal, alpha1, nonconf_by_class) -> dict:
    """Per-class achieved coverage for Mondrian / marginal-CP / MSP, with
    Beta tolerance bands, matching the paper's stated coverage-validation
    experiment."""
    from .stage1_classifier.train import conformal_pvalue

    y_cal = split.y_cal
    y_te_family = split.y_te
    label_encoder = split.label_encoder

    t_msp = msp_calibrate(proba_cal, y_cal, alpha1)
    t_marginal = marginal_cp_calibrate(proba_cal, y_cal, alpha1)

    rows = []
    for cls in split.known_classes + [BENIGN_LABEL]:
        mask = y_te_family == cls
        n_test = int(mask.sum())
        if n_test == 0:
            continue
        cls_idx = list(label_encoder.classes_).index(cls)
        y_te_encoded_cls = np.full(n_test, cls_idx)

        # Mondrian: true label covered iff its own class p-value clears alpha
        # -- same nonconf_by_class used at training time, same calibration
        # inputs as the other two methods for a fair comparison. Vectorized
        # via searchsorted (nonconf_by_class[cls] is pre-sorted ascending
        # from calibrate_gate1) -- a naive per-row loop would be O(n_test *
        # n_calib), catastrophically slow for classes like DDoS (~760k calib
        # points).
        calib_scores_sorted = nonconf_by_class[cls]
        scores_true = 1.0 - proba_te[mask, cls_idx]
        n_ge = len(calib_scores_sorted) - np.searchsorted(calib_scores_sorted, scores_true, side="left")
        p_true_cls = (n_ge + 1) / (len(calib_scores_sorted) + 1)
        mondrian_cov = (p_true_cls > alpha1).mean()

        msp_cov = msp_covered(proba_te[mask], y_te_encoded_cls, t_msp).mean()
        marg_cov = marginal_cp_covered(proba_te[mask], y_te_encoded_cls, t_marginal).mean()

        n_calib_cls = int((y_cal == cls_idx).sum())
        band = beta_tolerance_band(n_calib_cls, alpha1)

        rows.append({
            "class": cls, "n_test": n_test, "n_calib": n_calib_cls,
            "mondrian_coverage": float(mondrian_cov),
            "msp_coverage": float(msp_cov),
            "marginal_cp_coverage": float(marg_cov),
            "target_coverage": 1 - alpha1,
            "beta_tolerance_band": band,
        })
    return {"rows": rows, "t_msp": t_msp, "t_marginal": t_marginal}


# -------------------------------------------------------------- M5: Gate 1 ROC

def gate1_roc(proba_te: np.ndarray, y_te_family: np.ndarray, label_encoder) -> dict:
    """One-vs-rest ROC per known class, from raw softmax probabilities."""
    curves = {}
    for j, cls in enumerate(label_encoder.classes_):
        y_binary = (y_te_family == cls).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue
        fpr, tpr, _ = roc_curve(y_binary, proba_te[:, j])
        curves[cls] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    return curves


# -------------------------------------------------------------- M6: Gate 2 ROC

def compute_dual_scores(df, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len):
    """Shared scoring helper: dual anomaly score for every row in df, via
    non-overlapping seq_len windows in existing file order (same
    approximation used throughout -- see sequences.py docstring)."""
    X = stage2_scaler.transform(df[feat_cols].values.astype(np.float32)).astype(np.float32)
    seqs = build_sequences(X, seq_len=seq_len)
    if len(seqs) == 0:
        return np.array([])
    recon, z = predict_recon_latent(stage2_model, seqs)
    mse = np.mean(np.square(seqs - recon), axis=(1, 2))
    ldev = np.linalg.norm(z - gate2_bundle["centroid"], axis=1)
    return dual_score(mse, ldev, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])


def _roc_from_scores(benign_scores, other_scores, gate2_bundle, label):
    if len(benign_scores) == 0 or len(other_scores) == 0:
        return None
    y_binary = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(other_scores))])
    scores = np.concatenate([benign_scores, other_scores])
    fpr, tpr, _ = roc_curve(y_binary, scores)

    tau = gate2_bundle["threshold"]
    dr_at_tau = float((other_scores > tau).mean())
    fpr_at_tau = float((benign_scores > tau).mean())

    return {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)),
            "benign_scores": benign_scores, "zd_scores": other_scores,
            "tau": float(tau), "dr_at_tau": dr_at_tau, "fpr_at_tau": fpr_at_tau,
            "n_benign_windows": len(benign_scores), "n_zeroday_windows": len(other_scores),
            "label": label}


def gate2_roc(test_pool, held_out_family, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len):
    """Benign vs held-out zero-day family, dual anomaly score swept across
    thresholds (not just the fixed calibrated one) -- the AE's raw
    discriminative power, independent of Gate 1 routing. Also returns the
    raw score arrays (for the score-distribution/tau-selection plot) and
    the achieved DR/FPR at the CURRENT calibrated threshold specifically."""
    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    zd_rows = test_pool[test_pool["family"] == held_out_family]
    benign_scores = compute_dual_scores(benign_rows, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len)
    zd_scores = compute_dual_scores(zd_rows, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len)
    return _roc_from_scores(benign_scores, zd_scores, gate2_bundle, held_out_family)


def gate2_roc_all_attacks(test_pool, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len):
    """Benign vs ALL attack traffic combined (every family, not tied to any
    one zero-day rotation) -- the natural 'shared AE' view, since the AE
    itself doesn't know about rotations, only the per-rotation framing does."""
    benign_rows = test_pool[test_pool["family"] == BENIGN_LABEL]
    attack_rows = test_pool[test_pool["family"] != BENIGN_LABEL]
    benign_scores = compute_dual_scores(benign_rows, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len)
    attack_scores = compute_dual_scores(attack_rows, stage2_model, stage2_scaler, gate2_bundle, feat_cols, seq_len)
    return _roc_from_scores(benign_scores, attack_scores, gate2_bundle, "All Attacks")
