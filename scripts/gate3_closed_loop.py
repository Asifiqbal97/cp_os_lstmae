"""Gate 3 CLOSED LOOP -- the architectural gap flagged in the audit as
missing: paper says "the closed loop retrains the classifier with a
promoted class and recalibrates the gate." Until now, promotion only
produced a "Candidate-Class-N" LABEL; nothing folded it back into Gate 1.

Leak-free design: held-out family's test rows are split into DISCOVERY
(used to find/promote a cluster) and EVAL (completely unseen by anything
in this script until the final check) -- so "does retraining help" is
measured on genuinely fresh flows, not the same ones used to build the
promoted class.

Steps: discover+promote on DISCOVERY (reuses existing Gate1/Gate2, read-
only) -> fold promoted cluster's DISCOVERY flows into an EXPANDED training
set -> retrain Gate 1 with the new class -> recalibrate Mondrian (new
class's own calib carved from DISCOVERY, not EVAL) -> compare EVAL flows'
routing under OLD vs NEW Gate 1 -> regression-check the original known
classes didn't get worse.

REAL new training (unlike the read-only diagnostics). Writes to a NEW
isolated folder (artifacts/diagnostics/closed_loop/) -- does not modify
artifacts/<family>/'s existing Gate 1, only reads it for the "before"
comparison.

Usage:
    python -m scripts.gate3_closed_loop --config configs/mvp.yaml --family Spoofing
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.loader import load_csv, FEATURE_COLUMNS
from src.data.zero_day_split import make_zero_day_split
from src.analysis import load_rotation_artifacts, load_shared_ae_artifacts
from src.stage1_classifier.train import gate1_predict, train_lgbm, calibrate_gate1, SEED
from src.stage2_anomaly.lstm_ae import predict_recon_latent
from src.stage2_anomaly.conformal_gate import dual_score
from src.stage2_anomaly.clustering import LeaderFollowerClusterer

OUT_DIR = Path("artifacts") / "diagnostics" / "closed_loop"


def route_and_cluster(X_raw, booster, label_encoder, nonconf_by_class, alpha1, scaler,
                       stage2_model, stage2_scaler, gate2_bundle, clusterer, seq_len):
    """Minimal reimplementation of pipeline.run_pipeline's routing logic,
    but ALSO returns which raw rows belong to which cluster (pipeline.py's
    saved verdict collapses this for unpromoted clusters -- same limitation
    as evaluate_nmi_purity.py, same fix)."""
    X_s = scaler.transform(X_raw).astype(np.float32)
    proba = booster.predict(X_s, num_iteration=booster.best_iteration)

    deferred_idx = []
    for i in range(len(proba)):
        pred_set, _, _ = gate1_predict(proba[i], label_encoder, nonconf_by_class, alpha1)
        if len(pred_set) != 1:
            deferred_idx.append(i)

    n_windows = len(deferred_idx) // seq_len
    used = n_windows * seq_len
    if n_windows == 0:
        return {}, []

    X_deferred_s = stage2_scaler.transform(X_raw[deferred_idx[:used]]).astype(np.float32)
    windows = X_deferred_s.reshape(n_windows, seq_len, X_deferred_s.shape[1])
    recon, z = predict_recon_latent(stage2_model, windows)
    mse = np.mean(np.square(windows - recon), axis=(1, 2))
    ldev = np.linalg.norm(z - gate2_bundle["centroid"], axis=1)
    scores = dual_score(mse, ldev, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])
    novel_mask = scores > gate2_bundle["threshold"]

    cluster_to_rows = {}  # cluster_idx -> list of RAW ROW INDICES (into X_raw)
    for w in range(n_windows):
        if not novel_mask[w]:
            continue
        _, cluster_idx = clusterer.assign(z[w])
        row_idx = deferred_idx[w * seq_len:(w + 1) * seq_len]
        cluster_to_rows.setdefault(cluster_idx, []).extend(row_idx)

    return cluster_to_rows, deferred_idx


def main(cfg_path: str, family: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    alpha1, seq_len, min_cluster_size = cfg["alpha_stage1"], cfg["seq_len"], cfg["min_cluster_size"]

    artifacts_dir = OUT_DIR / family
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading data + existing artifacts (read-only)...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])
    old_booster, old_le, old_nonconf, _ = load_rotation_artifacts(family)
    stage2_model, stage2_scaler, gate2_bundle, _ = load_shared_ae_artifacts()
    split = make_zero_day_split(train_pool, test_pool, family, random_state=cfg["random_state"])

    family_mask = (split.y_te == family)
    X_family_raw = split.X_te[family_mask]
    print(f"       {len(X_family_raw)} {family} test rows found")

    print("[2/7] Splitting into DISCOVERY (find/promote cluster) vs EVAL (untouched until step 6)...")
    X_discovery, X_eval = train_test_split(X_family_raw, test_size=0.5, random_state=cfg["random_state"])
    print(f"       discovery={len(X_discovery)}  eval={len(X_eval)}")

    print("[3/7] Running discovery pass (existing Gate1+Gate2, read-only) to find a promotable cluster...")
    clusterer = LeaderFollowerClusterer(gate2_bundle["cluster_radius"], min_cluster_size)
    cluster_to_rows, _ = route_and_cluster(
        X_discovery, old_booster, old_le, old_nonconf, alpha1, split.scaler,
        stage2_model, stage2_scaler, gate2_bundle, clusterer, seq_len,
    )
    promoted = {c: rows for c, rows in cluster_to_rows.items() if len(rows) >= min_cluster_size}
    if not promoted:
        print(f"       NO cluster reached min_cluster_size={min_cluster_size} on discovery data. "
              f"Cluster sizes found: {[len(r) for r in cluster_to_rows.values()]}. Stopping.")
        return
    best_cluster = max(promoted, key=lambda c: len(promoted[c]))
    promoted_rows = sorted(set(promoted[best_cluster]))
    print(f"       promoted cluster {best_cluster}: {len(promoted_rows)} rows -> new class 'Candidate-Class-1'")

    print("[4/7] Building EXPANDED training set (original known classes + promoted class)...")
    X_promoted_raw = X_discovery[promoted_rows]
    X_promo_train, X_promo_calib = train_test_split(X_promoted_raw, test_size=0.3, random_state=cfg["random_state"])

    known_labels_str = old_le.inverse_transform(split.y_tr)
    all_labels_str = np.concatenate([known_labels_str, np.full(len(X_promo_train), "Candidate-Class-1")])
    all_X_tr_raw = np.concatenate([split.X_tr, X_promo_train], axis=0)  # split.X_tr is the RAW (unscaled) train set

    print(f"       new label space: {sorted(np.unique(all_labels_str))}")

    print("[5/7] Retraining Gate 1 with expanded label space...")
    new_scaler = StandardScaler().fit(all_X_tr_raw)
    new_le = LabelEncoder()
    y_all_encoded = new_le.fit_transform(all_labels_str)

    known_calib_str = old_le.inverse_transform(split.y_cal)
    all_calib_labels_str = np.concatenate([known_calib_str, np.full(len(X_promo_calib), "Candidate-Class-1")])
    all_X_cal_raw = np.concatenate([split.X_cal, X_promo_calib], axis=0)
    y_cal_encoded = new_le.transform(all_calib_labels_str)

    class _FakeSplit:
        pass
    fake_split = _FakeSplit()
    fake_split.X_tr_s = new_scaler.transform(all_X_tr_raw).astype(np.float32)
    fake_split.y_tr = y_all_encoded
    fake_split.X_cal_s = new_scaler.transform(all_X_cal_raw).astype(np.float32)
    fake_split.y_cal = y_cal_encoded

    new_booster = train_lgbm(fake_split, len(new_le.classes_))
    new_nonconf = calibrate_gate1(new_booster, fake_split, new_le)

    joblib.dump({"model": new_booster, "label_encoder": new_le}, artifacts_dir / "lightgbm_retrained.pkl", compress=3)
    joblib.dump({"nonconf_by_class": new_nonconf, "alpha_1": alpha1}, artifacts_dir / "conformal_retrained.pkl", compress=3)

    print("[6/7] Evaluating on UNSEEN eval flows: OLD Gate1 vs NEW (retrained) Gate1...")
    X_eval_old_s = split.scaler.transform(X_eval).astype(np.float32)
    proba_old = old_booster.predict(X_eval_old_s, num_iteration=old_booster.best_iteration)
    old_singleton = 0
    for i in range(len(proba_old)):
        pred_set, _, _ = gate1_predict(proba_old[i], old_le, old_nonconf, alpha1)
        if len(pred_set) == 1:
            old_singleton += 1
    old_recognized_rate = old_singleton / len(X_eval)

    X_eval_new_s = new_scaler.transform(X_eval).astype(np.float32)
    proba_new = new_booster.predict(X_eval_new_s, num_iteration=new_booster.best_iteration)
    new_correct = 0
    for i in range(len(proba_new)):
        pred_set, _, _ = gate1_predict(proba_new[i], new_le, new_nonconf, alpha1)
        if pred_set == {"Candidate-Class-1"}:
            new_correct += 1
    new_recognized_rate = new_correct / len(X_eval)

    print(f"       BEFORE retrain: {old_recognized_rate:.4f} of eval flows exited Gate1 confidently (any label)")
    print(f"       AFTER  retrain: {new_recognized_rate:.4f} of eval flows correctly exit as 'Candidate-Class-1'")

    print("[7/7] Regression check: did retraining hurt the ORIGINAL known classes?")
    known_test_mask = ~family_mask
    X_known_test = split.X_te[known_test_mask]
    y_known_test = split.y_te[known_test_mask]
    X_known_test_old_s = split.scaler.transform(X_known_test).astype(np.float32)
    X_known_test_new_s = new_scaler.transform(X_known_test).astype(np.float32)
    old_pred = old_le.inverse_transform(old_booster.predict(X_known_test_old_s, num_iteration=old_booster.best_iteration).argmax(axis=1))
    new_pred = new_le.inverse_transform(new_booster.predict(X_known_test_new_s, num_iteration=new_booster.best_iteration).argmax(axis=1))
    old_acc = float((old_pred == y_known_test).mean())
    new_acc = float((new_pred == y_known_test).mean())
    print(f"       Known-class accuracy: before={old_acc:.4f}  after={new_acc:.4f}  "
          f"(delta={new_acc-old_acc:+.4f})")

    with open(artifacts_dir / "result.md", "w") as f:
        f.write(f"# Gate 3 Closed Loop: {family}\n\n")
        f.write(f"Promoted cluster {best_cluster} ({len(promoted_rows)} discovery-set rows) "
                f"folded into Gate 1 as 'Candidate-Class-1'.\n\n")
        f.write("| Metric | Before retrain | After retrain |\n|---|---|---|\n")
        f.write(f"| Eval flows recognized (any confident label / correct new class) | "
                f"{old_recognized_rate:.4f} | {new_recognized_rate:.4f} |\n")
        f.write(f"| Known-class accuracy (regression check) | {old_acc:.4f} | {new_acc:.4f} |\n")
        f.write(f"\nEval set size: {len(X_eval)} rows, completely unseen by discovery/training.\n")

    print(f"\nSaved: {artifacts_dir}/result.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    main(args.config, args.family)
