"""Gate 2 calibration, ported from calibrate_gate2() in
cp_osr_lstmae_train_and_export.py. Matches cloud_service.dual_score exactly.
Adapted for PyTorch (single model with .encode(), not separate Keras
autoencoder/encoder sub-models)."""

import numpy as np
from scipy.spatial.distance import pdist

from .lstm_ae import predict_recon_latent


def dual_score(mse, ldev, mse_p99, ldev_p99, score_alpha):
    """MUST match cloud_service.dual_score exactly."""
    return (score_alpha * np.clip(mse / (mse_p99 + 1e-8), 0, None)
            + (1 - score_alpha) * np.clip(ldev / (ldev_p99 + 1e-8), 0, None))


def calibrate_gate2(model, calib_seqs: np.ndarray, alpha_ae: float, score_alpha: float,
                     random_state: int = 42) -> dict:
    recon, z = predict_recon_latent(model, calib_seqs)
    mse = np.mean(np.square(calib_seqs - recon), axis=(1, 2))
    centroid = z.mean(axis=0)
    ldev = np.linalg.norm(z - centroid, axis=1)

    mse_p99 = float(np.percentile(mse, 99))
    ldev_p99 = float(np.percentile(ldev, 99))
    scores = dual_score(mse, ldev, mse_p99, ldev_p99, score_alpha)

    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha_ae))), n)
    threshold = float(np.sort(scores)[k - 1])

    rng = np.random.default_rng(random_state)
    sub = z[rng.choice(len(z), min(2000, len(z)), replace=False)]
    cluster_radius = float(1.5 * np.median(pdist(sub))) if len(sub) > 1 else 1.0

    return {
        "threshold": threshold, "alpha": score_alpha,
        "mse_p99": mse_p99, "ldev_p99": ldev_p99,
        "centroid": centroid, "cluster_radius": cluster_radius,
    }
