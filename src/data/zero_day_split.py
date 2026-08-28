"""3-level zero-day holdout, ported from cp_osr_lstmae_train_and_export.py's
three_level_holdout() onto our schema (`family` column instead of `label`,
explicit FEATURE_COLUMNS instead of "everything except label").

Kept as a near-literal port so behavior matches the deployment-compatible
training script exactly -- same train/calib/test split structure, same
sklearn LabelEncoder fit on known classes only (Level 3), same stratification.
"""

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .family_map import BENIGN_LABEL
from .loader import FEATURE_COLUMNS


@dataclass
class ZeroDaySplit:
    held_out_family: str
    label_encoder: LabelEncoder
    scaler: StandardScaler
    feat_cols: list

    X_tr: np.ndarray
    y_tr: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray

    # scaled versions (what the models actually train/predict on)
    X_tr_s: np.ndarray
    X_cal_s: np.ndarray
    X_te_s: np.ndarray

    known_classes: list


def make_zero_day_split(
    train_pool: pd.DataFrame,
    test_pool: pd.DataFrame,
    held_out_family: str,
    calib_frac: float = 0.2,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> ZeroDaySplit:
    if held_out_family == BENIGN_LABEL:
        raise ValueError("Cannot hold out Benign -- it's not an attack family")

    # Level 1: known-side pool excludes the held-out family entirely.
    known_mask = train_pool["family"] != held_out_family
    df_known = train_pool.loc[known_mask]

    known_classes = sorted(
        df_known.loc[df_known["family"] != BENIGN_LABEL, "family"].unique()
    )
    if held_out_family in known_classes:
        raise ValueError(f"Leak check failed: {held_out_family} still present after filtering")

    # Level 3: label encoder fit on known classes only -- held_out_family
    # gets no slot anywhere downstream.
    label_encoder = LabelEncoder()
    y_known = label_encoder.fit_transform(df_known["family"].values)

    # Level 2: train/calib carved from the known-side pool only, stratified.
    X_known = df_known[FEATURE_COLUMNS].values.astype(np.float32)
    X_tr_full, X_te_holdout, y_tr_full, y_te_holdout = train_test_split(
        X_known, y_known, test_size=test_frac, stratify=y_known, random_state=random_state,
    )
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr_full, y_tr_full, test_size=calib_frac, stratify=y_tr_full, random_state=random_state,
    )
    del X_known, X_tr_full, y_tr_full
    gc.collect()

    # Test set: the full merged test file, all families including held-out --
    # used to measure both closed-set accuracy and zero-day recall. Rows
    # whose family has no label-encoder slot (i.e. the held-out family) are
    # kept as raw features + their true family string, not an encoded y,
    # since they're evaluated on "was this flagged novel?", not classified.
    X_te = test_pool[FEATURE_COLUMNS].values.astype(np.float32)
    y_te_family = test_pool["family"].values  # strings, not encoded

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_cal_s = scaler.transform(X_cal).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    return ZeroDaySplit(
        held_out_family=held_out_family,
        label_encoder=label_encoder,
        scaler=scaler,
        feat_cols=FEATURE_COLUMNS,
        X_tr=X_tr, y_tr=y_tr,
        X_cal=X_cal, y_cal=y_cal,
        X_te=X_te, y_te=y_te_family,
        X_tr_s=X_tr_s, X_cal_s=X_cal_s, X_te_s=X_te_s,
        known_classes=known_classes,
    )
