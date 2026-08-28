"""Builds seq_len=10 sequences of benign flows for Gate 2.

MEMORY FIX vs literal cp_osr_lstmae_train_and_export.py: that script's
build_sequences() defaults to stride=1 (every possible overlapping window),
which over millions of benign rows would produce a multi-GB array on a
~10GB RAM machine. We default to stride=seq_len (non-overlapping windows)
to keep this safe -- agreed with user as the default with "no conflict".

SOURCE-ORDER FIX vs using the merged file: CICIoMT2024 has no source/
timestamp column, so genuine per-source sequences aren't buildable offline.
Using the raw, unmerged Benign_train/test_labelled.csv (native capture
order, not merge-shuffled) is a better proxy for chronological adjacency
than the merged file's arbitrary row order -- still not true per-device
sequences (multiple benign devices likely interleave), but strictly better
than the merged-file alternative. This is an offline-training-only
limitation: cloud_service.py's live sequence_buffers are keyed by genuine
source_key at inference time, so this approximation doesn't propagate to
the live deployment, only to how the model was trained/calibrated.
"""

import numpy as np


def build_sequences(X: np.ndarray, seq_len: int = 10, source_ids=None, stride: int = None):
    if stride is None:
        stride = seq_len  # non-overlapping by default (memory fix)

    seqs = []
    if source_ids is not None:
        order = np.argsort(source_ids, kind="stable")
        X_sorted, src_sorted = X[order], source_ids[order]
        start = 0
        for i in range(1, len(src_sorted) + 1):
            if i == len(src_sorted) or src_sorted[i] != src_sorted[start]:
                block = X_sorted[start:i]
                for j in range(0, max(len(block) - seq_len + 1, 0), stride):
                    seqs.append(block[j:j + seq_len])
                start = i
    else:
        for j in range(0, max(len(X) - seq_len + 1, 0), stride):
            seqs.append(X[j:j + seq_len])

    return np.stack(seqs, axis=0) if seqs else np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
