"""M1: 80/20 split of raw benign-only data for the shared AE, with a
scaler fit on benign data only -- separate from each rotation's Stage-1
scaler (which is fit on the full known-side pool: attacks + benign).

This is the actual bug fix: Gate 2's split-conformal threshold bounds the
BENIGN false-alarm rate, so calibration must use benign-only flows, not the
mixed attack+benign calibration set the per-rotation split produces.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .loader import FEATURE_COLUMNS


@dataclass
class BenignSplit:
    scaler: StandardScaler
    X_ae_train: np.ndarray   # raw (unscaled) features
    X_ae_calib: np.ndarray   # raw (unscaled) features


def make_benign_split(benign_train_df: pd.DataFrame, calib_frac: float = 0.2,
                       random_state: int = 42) -> BenignSplit:
    X = benign_train_df[FEATURE_COLUMNS].values.astype(np.float32)

    # Non-shuffled split preserves native capture order within each half --
    # matters because Gate 2 sequence building treats row order as a
    # chronological proxy (see stage2_anomaly/sequences.py docstring).
    n_calib = int(len(X) * calib_frac)
    X_ae_calib = X[:n_calib]
    X_ae_train = X[n_calib:]

    scaler = StandardScaler().fit(X_ae_train)

    return BenignSplit(scaler=scaler, X_ae_train=X_ae_train, X_ae_calib=X_ae_calib)
