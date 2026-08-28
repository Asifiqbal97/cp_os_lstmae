"""Routes test flows through Gate 1 -> (singleton exit / deferred) -> Gate 2
-> (known-revert / novel) -> Gate 3 (cluster/promote) for novel flows.

DUAL-SCALER REWRITE (M4): takes RAW (unscaled) features and applies the
correct scaler internally per gate -- the rotation's own Stage-1 scaler for
LightGBM, and the shared benign-only scaler for the AE. Previously both
gates shared one scaler fit on the mixed attack+benign pool, which is what
made a per-rotation-trained AE seem necessary; with the AE now shared
across rotations (trained/calibrated once on benign-only data), its scaler
must be applied here rather than baked into a pre-scaled input array.

FIX applied per paper fidelity (agreed with user): Gate 1 defers BOTH empty
AND multi-element prediction sets -- not just empty, as the originally-
provided edge_gateway_service.py did. This closes the Path B evasion
channel the paper's symmetric-gate design exists to prevent.

SEQUENCE LIMITATION for offline evaluation (documented, not hidden): Gate 2
needs seq_len=10 sequences, but deferred test rows have no source identifier
to group by. We treat the full deferred subset, in its existing file order,
as one stream and build non-overlapping seq_len windows over it (same
"whole set = one ordered stream" approximation used for Stage 2 training).
Each window's Gate-2/Gate-3 verdict is broadcast to all seq_len rows in it,
except the known-revert case, which still uses each row's own Stage-1
argmax label. A remainder of < seq_len deferred rows at the end can't form
a full window and is marked "buffering_incomplete" -- excluded from metrics,
matching cloud_service.py's own "Buffering" outcome for an unfilled sequence
buffer, not silently dropped.
"""

import numpy as np

from .stage1_classifier.train import gate1_predict
from .stage2_anomaly.conformal_gate import dual_score
from .stage2_anomaly.lstm_ae import predict_recon_latent


def run_pipeline(X_te_raw, y_te_family, booster, label_encoder, nonconf_by_class, alpha1,
                  stage1_scaler, stage2_model, stage2_scaler, gate2_bundle, clusterer, seq_len=10):
    n = len(X_te_raw)
    X_te_stage1 = stage1_scaler.transform(X_te_raw).astype(np.float32)
    proba = booster.predict(X_te_stage1, num_iteration=booster.best_iteration)

    verdict = np.empty(n, dtype=object)
    path = np.empty(n, dtype=object)
    argmax_label = np.empty(n, dtype=object)
    pred_set_size = np.zeros(n, dtype=np.int8)
    true_in_set = np.zeros(n, dtype=bool)

    deferred_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        pred_set, argmax_cls, _ = gate1_predict(proba[i], label_encoder, nonconf_by_class, alpha1)
        argmax_label[i] = argmax_cls
        pred_set_size[i] = len(pred_set)
        # Coverage tracking: is this row's TRUE family in its own prediction
        # set? Only meaningful for known classes -- a zero-day row's true
        # family has no encoder slot, so it can never appear here by
        # construction, and is excluded from the coverage report downstream.
        true_in_set[i] = y_te_family[i] in pred_set

        if len(pred_set) == 1:
            verdict[i] = next(iter(pred_set))
            path[i] = "singleton_exit"
        else:
            deferred_mask[i] = True
            path[i] = "deferred_empty" if len(pred_set) == 0 else "deferred_ambiguous"

    deferred_idx = np.where(deferred_mask)[0]
    n_windows = len(deferred_idx) // seq_len
    used_idx = deferred_idx[: n_windows * seq_len]
    leftover_idx = deferred_idx[n_windows * seq_len:]

    if n_windows > 0:
        # Gate 2 uses its OWN (benign-only) scaler, not stage1_scaler.
        X_deferred_ae = stage2_scaler.transform(X_te_raw[used_idx]).astype(np.float32)
        windows = X_deferred_ae.reshape(n_windows, seq_len, X_deferred_ae.shape[1])

        recon, z = predict_recon_latent(stage2_model, windows)
        mse = np.mean(np.square(windows - recon), axis=(1, 2))
        ldev = np.linalg.norm(z - gate2_bundle["centroid"], axis=1)

        scores = dual_score(mse, ldev, gate2_bundle["mse_p99"], gate2_bundle["ldev_p99"], gate2_bundle["alpha"])
        novel_mask = scores > gate2_bundle["threshold"]

        for w in range(n_windows):
            row_idx = used_idx[w * seq_len:(w + 1) * seq_len]
            if novel_mask[w]:
                label, cluster_idx = clusterer.assign(z[w])
                verdict[row_idx] = label
                path[row_idx] = "deferred_novel"
            else:
                verdict[row_idx] = argmax_label[row_idx]
                path[row_idx] = "deferred_known_revert"

    if len(leftover_idx) > 0:
        verdict[leftover_idx] = "buffering_incomplete"
        path[leftover_idx] = "buffering_incomplete"

    return {
        "family": y_te_family,
        "verdict": verdict,
        "path": path,
        "stage1_argmax": argmax_label,
        "gate1_pred_set_size": pred_set_size,
        "gate1_true_in_set": true_in_set,
    }
