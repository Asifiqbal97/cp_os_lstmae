"""Builds the shared Gate 2 (AE) once: benign-only split, scaler, training,
and calibration. Used once per full study run, reused across all 5
zero-day rotations and unaffected by which family is held out (benign
data never changes across rotations).
"""

from pathlib import Path

import joblib
import numpy as np
import torch

from .data.benign_split import make_benign_split
from .stage2_anomaly.sequences import build_sequences
from .stage2_anomaly.lstm_ae import train_lstm_ae
from .stage2_anomaly.conformal_gate import calibrate_gate2
from .report_saving import save_shared_ae_report


def build_shared_ae(benign_train_df, cfg: dict, artifacts_root: str = "artifacts", verbose: bool = True):
    def log(msg):
        if verbose:
            print(msg)

    artifacts_dir = Path(artifacts_root) / "shared_ae"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log("Building benign-only split (M1)...")
    bsplit = make_benign_split(benign_train_df, calib_frac=0.2, random_state=cfg["random_state"])
    log(f"       AE-train={len(bsplit.X_ae_train)} rows  AE-calib={len(bsplit.X_ae_calib)} rows")

    X_ae_train_scaled = bsplit.scaler.transform(bsplit.X_ae_train).astype(np.float32)
    train_seqs = build_sequences(X_ae_train_scaled, seq_len=cfg["seq_len"])
    log(f"       {len(train_seqs)} non-overlapping training sequences built")

    log("Training shared AE (M2)...")
    stage2_model, _, loss_history, recon_errors = train_lstm_ae(
        train_seqs, cfg["seq_len"], cfg["hidden_dim"], cfg["latent_dim"],
        epochs=cfg["stage2_epochs"], batch_size=cfg["stage2_batch_size"],
    )
    torch.save({
        "state_dict": stage2_model.state_dict(),
        "n_features": train_seqs.shape[2], "seq_len": cfg["seq_len"],
        "hidden_dim": cfg["hidden_dim"], "latent_dim": cfg["latent_dim"],
    }, artifacts_dir / "stage2_model.pt")
    joblib.dump(bsplit.scaler, artifacts_dir / "benign_scaler.pkl", compress=3)

    log(f"Calibrating Gate 2 on BENIGN-ONLY calibration data (M3, alpha={cfg['alpha_stage2']})...")
    X_ae_calib_scaled = bsplit.scaler.transform(bsplit.X_ae_calib).astype(np.float32)
    calib_seqs = build_sequences(X_ae_calib_scaled, seq_len=cfg["seq_len"])
    gate2_bundle = calibrate_gate2(stage2_model, calib_seqs, cfg["alpha_stage2"],
                                    cfg["score_alpha"], cfg["random_state"])
    joblib.dump(gate2_bundle, artifacts_dir / "cp_osr_ae_threshold.pkl", compress=3)
    log(f"       threshold={gate2_bundle['threshold']:.4f}  cluster_radius={gate2_bundle['cluster_radius']:.4f}")

    save_shared_ae_report(artifacts_dir, loss_history, recon_errors, gate2_bundle, cfg)
    log(f"       report saved to {artifacts_dir}/metrics.json and {artifacts_dir}/report.md")

    return stage2_model, bsplit.scaler, gate2_bundle
