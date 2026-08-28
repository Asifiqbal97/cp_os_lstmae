"""OpenMax / EFC / OCN baselines -- the 3 comparison methods from the
paper's Table 1 never built (only MSP + marginal-CP exist, in
coverage_validation.py).

HONEST ADAPTATIONS, not literal reproductions -- documented explicitly,
not silently approximated:

  OpenMax (Bendale & Boult 2016): originally defined on DNN penultimate-
  layer activations. Adapted here to use LightGBM's per-class probability
  vector as the "activation vector" (a probability vector is the closest
  analogue a tree ensemble has to a DNN's activation vector) -- per-class
  mean vector + Weibull-tail EVT fit on distances, exactly as OpenMax
  specifies, just on a different input representation.

  EFC (Souza et al. 2025, cited in the paper as "the key empirical
  anchor"): the real method fits an inverse-Potts model via mean-field DCA
  on discretized categorical features -- a substantial undertaking on its
  own. Approximated here with a per-class Gaussian energy score
  (Mahalanobis-style, DIAGONAL covariance for numerical stability given
  this dataset's known feature collinearity). This is NOT the inverse-
  Potts model -- flagged explicitly in every output.

  OCN (Zhang et al.): nearest-class-mean (NCM) prototype distance in
  scaled feature space -- the core mechanism, without OCN's own Fisher/MMD
  loss-trained embedding (that requires training a representation, out of
  scope here; distances are computed directly in the existing Stage-1-
  scaled feature space instead).

All three calibrate a single GLOBAL threshold (pooled across classes, same
convention as marginal-CP in coverage_validation.py) at alpha_stage1, for
a fair comparison against Mondrian/marginal-CP/MSP.

Read-only: reuses existing Gate 1 artifacts, no retraining. Writes to a
NEW isolated folder (artifacts/diagnostics/baselines/).

Usage:
    python -m scripts.baseline_comparison --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import weibull_min
from sklearn.metrics import roc_auc_score

from src.data.loader import load_csv
from src.data.family_map import KNOWN_FAMILIES, BENIGN_LABEL
from src.data.zero_day_split import make_zero_day_split
from src.analysis import load_rotation_artifacts

OUT_DIR = Path("artifacts") / "diagnostics" / "baselines"
TAIL_FRACTION = 0.5  # top 50% largest distances used for Weibull tail fit


# ---------------------------------------------------------------- OpenMax

def fit_openmax(proba_cal, y_cal, n_classes):
    class_means, weibull_params = {}, {}
    for c in range(n_classes):
        mask = (y_cal == c) & (proba_cal.argmax(axis=1) == c)  # correctly-classified only
        if mask.sum() < 5:
            class_means[c] = proba_cal[y_cal == c].mean(axis=0) if (y_cal == c).any() else np.zeros(n_classes)
            weibull_params[c] = (1.0, 0.0, 1.0)
            continue
        mean_vec = proba_cal[mask].mean(axis=0)
        dists = np.linalg.norm(proba_cal[mask] - mean_vec, axis=1)
        tail = np.sort(dists)[-max(int(len(dists) * TAIL_FRACTION), 3):]
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0)
        except Exception:
            shape, loc, scale = 1.0, 0.0, max(tail.mean(), 1e-6)
        class_means[c] = mean_vec
        weibull_params[c] = (shape, loc, scale)
    return class_means, weibull_params


def openmax_unknown_score(proba, class_means, weibull_params, n_classes):
    scores = np.zeros(len(proba))
    for c in range(n_classes):
        dist = np.linalg.norm(proba - class_means[c][None, :], axis=1)
        shape, loc, scale = weibull_params[c]
        survival = 1 - weibull_min.cdf(dist, shape, loc, scale)  # small survival = unusually far = anomalous
        scores += proba[:, c] * (1 - survival)  # weighted "how anomalous", per-class-prob-weighted
    return scores


# -------------------------------------------------------------------- EFC

def fit_efc(X_cal_s, y_cal, n_classes, eps=1e-3):
    means, variances = {}, {}
    for c in range(n_classes):
        Xc = X_cal_s[y_cal == c]
        if len(Xc) < 5:
            means[c] = X_cal_s.mean(axis=0)
            variances[c] = np.ones(X_cal_s.shape[1])
            continue
        means[c] = Xc.mean(axis=0)
        variances[c] = Xc.var(axis=0) + eps  # diagonal only -- avoids singular full covariance
    return means, variances


def efc_energy(X_s, pred_labels, means, variances):
    energy = np.zeros(len(X_s))
    for c in np.unique(pred_labels):
        mask = pred_labels == c
        diff = X_s[mask] - means[c][None, :]
        energy[mask] = 0.5 * np.sum((diff ** 2) / variances[c][None, :], axis=1)
    return energy


# -------------------------------------------------------------------- OCN

def fit_ocn(X_cal_s, y_cal, n_classes):
    prototypes = {}
    for c in range(n_classes):
        Xc = X_cal_s[y_cal == c]
        prototypes[c] = Xc.mean(axis=0) if len(Xc) > 0 else X_cal_s.mean(axis=0)
    return prototypes


def ocn_distance(X_s, prototypes, n_classes):
    dists = np.stack([np.linalg.norm(X_s - prototypes[c][None, :], axis=1) for c in range(n_classes)], axis=1)
    return dists.min(axis=1)


# ------------------------------------------------------------- calibration

def calibrate_global_threshold(scores, alpha):
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    alpha1 = cfg["alpha_stage1"]

    print("Loading data (read-only)...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for family in KNOWN_FAMILIES:
        print(f"\n{family}: loading Gate 1 (read-only, no retraining)...")
        booster, label_encoder, _, _ = load_rotation_artifacts(family)
        split = make_zero_day_split(train_pool, test_pool, family, random_state=cfg["random_state"])
        n_classes = len(label_encoder.classes_)

        proba_cal = booster.predict(split.X_cal_s, num_iteration=booster.best_iteration)
        proba_te = booster.predict(split.X_te_s, num_iteration=booster.best_iteration)
        y_cal = split.y_cal

        zd_mask = split.y_te == family

        print("  fitting OpenMax, EFC, OCN...")
        class_means, weibull_params = fit_openmax(proba_cal, y_cal, n_classes)
        om_scores_cal = openmax_unknown_score(proba_cal, class_means, weibull_params, n_classes)
        om_threshold = calibrate_global_threshold(om_scores_cal, alpha1)
        om_scores_te = openmax_unknown_score(proba_te, class_means, weibull_params, n_classes)

        efc_means, efc_vars = fit_efc(split.X_cal_s, y_cal, n_classes)
        efc_pred_cal = proba_cal.argmax(axis=1)
        efc_energy_cal = efc_energy(split.X_cal_s, efc_pred_cal, efc_means, efc_vars)
        efc_threshold = calibrate_global_threshold(efc_energy_cal, alpha1)
        efc_pred_te = proba_te.argmax(axis=1)
        efc_energy_te = efc_energy(split.X_te_s, efc_pred_te, efc_means, efc_vars)

        ocn_prototypes = fit_ocn(split.X_cal_s, y_cal, n_classes)
        ocn_dist_cal = ocn_distance(split.X_cal_s, ocn_prototypes, n_classes)
        ocn_threshold = calibrate_global_threshold(ocn_dist_cal, alpha1)
        ocn_dist_te = ocn_distance(split.X_te_s, ocn_prototypes, n_classes)

        def dr_fpr_auc(scores_te, threshold, zd_mask):
            rejected = scores_te > threshold
            dr = float(rejected[zd_mask].mean()) if zd_mask.any() else float("nan")
            benign_mask = split.y_te == BENIGN_LABEL
            fpr = float(rejected[benign_mask].mean()) if benign_mask.any() else float("nan")
            if zd_mask.sum() == 0 or benign_mask.sum() == 0:
                return dr, fpr, float("nan")
            y_bin = np.concatenate([np.ones(int(zd_mask.sum())), np.zeros(int(benign_mask.sum()))])
            s = np.concatenate([scores_te[zd_mask], scores_te[benign_mask]])
            auc = float(roc_auc_score(y_bin, s)) if len(np.unique(y_bin)) > 1 else float("nan")
            return dr, fpr, auc

        om_dr, om_fpr, om_auc = dr_fpr_auc(om_scores_te, om_threshold, zd_mask)
        efc_dr, efc_fpr, efc_auc = dr_fpr_auc(efc_energy_te, efc_threshold, zd_mask)
        ocn_dr, ocn_fpr, ocn_auc = dr_fpr_auc(ocn_dist_te, ocn_threshold, zd_mask)

        print(f"  OpenMax: DR={om_dr:.4f} FPR={om_fpr:.4f} AUC={om_auc:.4f}")
        print(f"  EFC:     DR={efc_dr:.4f} FPR={efc_fpr:.4f} AUC={efc_auc:.4f}")
        print(f"  OCN:     DR={ocn_dr:.4f} FPR={ocn_fpr:.4f} AUC={ocn_auc:.4f}")

        all_rows.append({"family": family, "openmax": (om_dr, om_fpr, om_auc),
                          "efc": (efc_dr, efc_fpr, efc_auc), "ocn": (ocn_dr, ocn_fpr, ocn_auc)})

        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ["OpenMax", "EFC (approx)", "OCN"]
        drs = [om_dr, efc_dr, ocn_dr]
        fprs = [om_fpr, efc_fpr, ocn_fpr]
        x = np.arange(3)
        ax.bar(x - 0.15, drs, 0.3, label="Detection rate (zero-day)", color="#55A868")
        ax.bar(x + 0.15, fprs, 0.3, label="False positive rate (benign)", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Baseline comparison: {family} held out")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{family}_baselines.png", dpi=120)
        plt.close(fig)

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("# OpenMax / EFC / OCN Baseline Comparison\n\n")
        f.write("**Adaptations, not literal reproductions** (see script docstring for full detail):\n")
        f.write("- OpenMax: uses LightGBM probability vectors as activations (not DNN logits)\n")
        f.write("- EFC: diagonal-covariance Gaussian energy (NOT the inverse-Potts DCA model)\n")
        f.write("- OCN: NCM prototype distance in existing feature space (not Fisher/MMD-trained embedding)\n\n")
        f.write("| Family | OpenMax DR | OpenMax FPR | OpenMax AUC | EFC DR | EFC FPR | EFC AUC | OCN DR | OCN FPR | OCN AUC |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            om, efc, ocn = r["openmax"], r["efc"], r["ocn"]
            f.write(f"| {r['family']} | {om[0]:.4f} | {om[1]:.4f} | {om[2]:.4f} | "
                    f"{efc[0]:.4f} | {efc[1]:.4f} | {efc[2]:.4f} | {ocn[0]:.4f} | {ocn[1]:.4f} | {ocn[2]:.4f} |\n")
        f.write("\nCompare against Mondrian/marginal-CP/MSP in coverage_validation.py's output "
                "(run_analysis.py) for the full 6-way comparison the paper's Table 1 describes.\n")

    print(f"\nSaved: {OUT_DIR}/result.md + plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
