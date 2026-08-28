"""NMI/purity cluster validation (paper-specified evaluation component,
never built until now): validates Gate 3's discovered clusters against
ground-truth Attack_type SUBTYPES.

Why this can't just read result.joblib: the saved 'verdict' string collapses
ALL unpromoted clusters into one generic "Zero-day-Unclustered" label --
only promoted clusters keep an individual identity ("Candidate-Class-N").
Real NMI/purity validation needs every discovered cluster's raw index, not
just the promoted ones. So this RE-SIMULATES Gate 1 -> Gate 2 -> Gate 3
fresh (same deterministic split, same seed, same order -> IDENTICAL
clustering result to the real run, just with the raw index captured
instead of collapsed).

Read-only: loads existing Gate 1 (per rotation) + shared Gate 2 (read-only,
no retraining). Writes to a NEW isolated folder
(artifacts/diagnostics/nmi_purity/).

Usage:
    python -m scripts.evaluate_nmi_purity --config configs/mvp.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import normalized_mutual_info_score

from src.data.loader import load_csv, FEATURE_COLUMNS
from src.data.family_map import KNOWN_FAMILIES
from src.data.zero_day_split import make_zero_day_split
from src.analysis import load_rotation_artifacts, load_shared_ae_artifacts
from src.stage1_classifier.train import gate1_predict
from src.stage2_anomaly.lstm_ae import predict_recon_latent
from src.stage2_anomaly.conformal_gate import dual_score
from src.stage2_anomaly.clustering import LeaderFollowerClusterer

OUT_DIR = Path("artifacts") / "diagnostics" / "nmi_purity"


def purity_score(true_labels, cluster_labels) -> float:
    """Standard clustering purity.

    For each cluster, count the majority true label, sum the majority
    counts across all clusters, and divide by the total number of samples.

    Uses Counter rather than scipy.stats.mode because true_labels may contain
    categorical/string Attack_type subtype labels.
    """
    total_correct = 0

    for c in np.unique(cluster_labels):
        mask = cluster_labels == c

        if mask.sum() == 0:
            continue

        cluster_true_labels = true_labels[mask]
        majority_count = Counter(cluster_true_labels).most_common(1)[0][1]
        total_correct += majority_count

    return total_correct / len(true_labels)


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    alpha1 = cfg["alpha_stage1"]
    seq_len = cfg["seq_len"]

    print("Loading data (read-only)...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])

    print("Loading shared AE (read-only, no retraining)...")
    stage2_model, stage2_scaler, gate2_bundle, _ = load_shared_ae_artifacts()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for family in KNOWN_FAMILIES:
        print(
            f"\n{family}: re-simulating pipeline to capture raw cluster indices..."
        )

        booster, label_encoder, nonconf_by_class, _ = load_rotation_artifacts(
            family
        )

        split = make_zero_day_split(
            train_pool,
            test_pool,
            family,
            random_state=cfg["random_state"],
        )

        # Test set is never shuffled -- test_pool row order matches split.X_te_s
        # exactly, so Attack_type aligns directly (same technique used in
        # diagnose_lofo_gate1_subtype.py).
        subtypes_all = test_pool["Attack_type"].values

        family_mask = split.y_te == family
        subtypes_family = subtypes_all[family_mask]

        proba = booster.predict(
            split.X_te_s,
            num_iteration=booster.best_iteration,
        )
        proba_family = proba[family_mask]

        deferred_idx = []

        for i in range(len(proba_family)):
            pred_set, _, _ = gate1_predict(
                proba_family[i],
                label_encoder,
                nonconf_by_class,
                alpha1,
            )

            if len(pred_set) != 1:
                deferred_idx.append(i)

        if len(deferred_idx) < seq_len:
            print(
                f"  skipped -- only {len(deferred_idx)} deferred rows, "
                f"need >= {seq_len} for one window"
            )
            continue

        X_family_raw = split.X_te[family_mask]

        X_deferred = split.scaler.transform(
            X_family_raw[deferred_idx]
        ).astype(np.float32)

        n_windows = len(deferred_idx) // seq_len
        used = n_windows * seq_len

        windows = X_deferred[:used].reshape(
            n_windows,
            seq_len,
            X_deferred.shape[1],
        )

        recon, z = predict_recon_latent(
            stage2_model,
            windows,
        )

        mse = np.mean(
            np.square(windows - recon),
            axis=(1, 2),
        )

        ldev = np.linalg.norm(
            z - gate2_bundle["centroid"],
            axis=1,
        )

        scores = dual_score(
            mse,
            ldev,
            gate2_bundle["mse_p99"],
            gate2_bundle["ldev_p99"],
            gate2_bundle["alpha"],
        )

        novel_mask = scores > gate2_bundle["threshold"]

        if novel_mask.sum() == 0:
            print("  skipped -- no windows flagged novel")
            continue

        clusterer = LeaderFollowerClusterer(
            gate2_bundle["cluster_radius"],
            cfg["min_cluster_size"],
        )

        cluster_indices = []
        window_true_subtypes = []

        for w in range(n_windows):
            if not novel_mask[w]:
                continue

            # Assign the window to a raw discovered cluster index.
            _, cluster_idx = clusterer.assign(z[w])
            cluster_indices.append(cluster_idx)

            # Representative subtype for this window: majority vote across
            # its seq_len constituent rows.
            #
            # Attack_type is categorical/string data, so use Counter rather
            # than scipy.stats.mode (which no longer accepts non-numeric
            # arrays in modern SciPy versions).
            window_subtype_rows = subtypes_family[
                deferred_idx[w * seq_len:(w + 1) * seq_len]
            ]

            majority_subtype = Counter(
                window_subtype_rows
            ).most_common(1)[0][0]

            window_true_subtypes.append(majority_subtype)

        cluster_indices = np.array(cluster_indices)
        window_true_subtypes = np.array(window_true_subtypes)

        n_clusters_found = len(np.unique(cluster_indices))
        n_true_subtypes = len(np.unique(window_true_subtypes))

        nmi = float(
            normalized_mutual_info_score(
                window_true_subtypes,
                cluster_indices,
            )
        )

        purity = float(
            purity_score(
                window_true_subtypes,
                cluster_indices,
            )
        )

        all_results[family] = {
            "n_novel_windows": int(novel_mask.sum()),
            "n_clusters_found": n_clusters_found,
            "n_true_subtypes": int(n_true_subtypes),
            "nmi": nmi,
            "purity": purity,
        }

        print(
            f"  {novel_mask.sum()} novel windows -> "
            f"{n_clusters_found} clusters found  "
            f"(vs {n_true_subtypes} true subtypes)  "
            f"NMI={nmi:.4f}  "
            f"Purity={purity:.4f}"
        )

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("# NMI / Purity Cluster Validation\n\n")

        f.write(
            "Re-simulates Gate 1->2->3 to capture EVERY discovered "
            "cluster's raw index (not collapsed to "
            "'Zero-day-Unclustered'), validated against true "
            "Attack_type subtypes via NMI and purity.\n\n"
        )

        f.write(
            "| Family | Novel windows | Clusters found | "
            "True subtypes | NMI | Purity |\n"
        )

        f.write(
            "|---|---|---|---|---|---|\n"
        )

        for fam, r in all_results.items():
            f.write(
                f"| {fam} | "
                f"{r['n_novel_windows']} | "
                f"{r['n_clusters_found']} | "
                f"{r['n_true_subtypes']} | "
                f"{r['nmi']:.4f} | "
                f"{r['purity']:.4f} |\n"
            )

    print(f"\nSaved: {OUT_DIR}/result.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
    )

    args = parser.parse_args()
    main(args.config)
