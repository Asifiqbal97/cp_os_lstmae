"""Evaluation metrics for the rebuilt pipeline. 'novel' now covers two
concrete Gate 3 outcomes (Zero-day-Unclustered / Candidate-Class-N) rather
than one flat 'novel' string -- both count as "correctly flagged novel"
for zero-day recall, since paper credit for Path A doesn't require the
cluster to have been promoted yet.
"""

import numpy as np
from sklearn.metrics import f1_score


def _is_novel_verdict(verdict) -> bool:
    return verdict == "Zero-day-Unclustered" or (
        isinstance(verdict, str) and verdict.startswith("Candidate-Class-")
    )


def evaluate_mvp(result: dict, held_out_family: str) -> dict:
    family = np.asarray(result["family"])
    verdict = np.asarray(result["verdict"])
    path = np.asarray(result["path"])

    scored_mask = path != "buffering_incomplete"
    family_s, verdict_s = family[scored_mask], verdict[scored_mask]

    known_mask = family_s != held_out_family
    zero_day_mask = family_s == held_out_family
    benign_mask = family_s == "Benign"

    known_correct = (verdict_s[known_mask] == family_s[known_mask]).mean() if known_mask.any() else float("nan")

    # FIX: restrict the label universe to the real known classes (the true
    # labels actually present), not sklearn's default of every distinct
    # value in true-or-predicted. Without this, spurious Gate-3 cluster
    # labels ("Candidate-Class-N", "Zero-day-Unclustered") that a known-class
    # row was wrongly routed to get counted as extra "classes" with F1=0
    # (they never appear as a true label), dragging the macro average down
    # even when the real known classes each score well individually.
    known_labels = np.unique(family_s[known_mask]) if known_mask.any() else np.array([])
    macro_f1 = f1_score(
        family_s[known_mask], verdict_s[known_mask], labels=known_labels,
        average="macro", zero_division=0,
    ) if known_mask.any() else float("nan")

    is_novel = np.vectorize(_is_novel_verdict)(verdict_s)
    zero_day_recall = is_novel[zero_day_mask].mean() if zero_day_mask.any() else float("nan")
    benign_far = is_novel[benign_mask].mean() if benign_mask.any() else float("nan")

    path_counts = dict(zip(*np.unique(path, return_counts=True)))
    n_buffering_dropped = int((path == "buffering_incomplete").sum())

    return {
        "known_class_accuracy": known_correct,
        "known_class_macro_f1": macro_f1,
        "zero_day_recall": zero_day_recall,
        "benign_false_alarm_rate": benign_far,
        "path_counts": path_counts,
        "n_rows_excluded_incomplete_buffer": n_buffering_dropped,
    }


def path_decomposition(result: dict, held_out_family: str) -> dict:
    """The paper's own open-set breakdown (Methodology, 'Evaluation design'):
    per zero-day flow, decomposed into four MUTUALLY EXCLUSIVE paths:
      A  -- correctly surfaced as novel
      B  -- confidently exited as a known ATTACK (Gate 1 singleton, wrong)
      B0 -- absorbed into Benign anywhere in the pipeline (the paper calls
            this the operationally most dangerous outcome)
      C  -- reverted to a known ATTACK label after deferral (Gate 2 said
            "not anomalous enough", but the fallback label happens to be an
            attack, not benign)
    """
    family = np.asarray(result["family"])
    verdict = np.asarray(result["verdict"])
    path = np.asarray(result["path"])

    scored = path != "buffering_incomplete"
    zd_mask = scored & (family == held_out_family)
    fam_zd, verdict_zd, path_zd = family[zd_mask], verdict[zd_mask], path[zd_mask]
    n = len(fam_zd)

    is_novel = np.vectorize(_is_novel_verdict)(verdict_zd)
    is_benign_verdict = verdict_zd == "Benign"

    A = is_novel
    B0 = (~is_novel) & is_benign_verdict
    B = (~is_novel) & (~is_benign_verdict) & (path_zd == "singleton_exit")
    C = (~is_novel) & (~is_benign_verdict) & (path_zd == "deferred_known_revert")

    counts = {
        "A_correctly_novel": int(A.sum()),
        "B_confident_known_attack": int(B.sum()),
        "B0_absorbed_benign": int(B0.sum()),
        "C_reverted_known_attack": int(C.sum()),
    }
    assert sum(counts.values()) == n, (
        f"path decomposition not exhaustive/mutually exclusive: "
        f"{sum(counts.values())} != {n} zero-day rows"
    )
    fractions = {k: (v / n if n else float("nan")) for k, v in counts.items()}

    return {"held_out_family": held_out_family, "n_zero_day_rows": n,
            "counts": counts, "fractions": fractions}


def gate1_raw_report(result: dict, held_out_family: str) -> dict:
    """Gate 1 / LightGBM alone, bypassing the conformal gate and Gate 2/3
    entirely -- uses stage1_argmax (the classifier's raw best guess for
    every row) instead of the pipeline's final verdict. This answers "how
    good is the underlying classifier", separate from how the conformal
    gate + Gate 2 revert logic affect the pipeline's final output."""
    family = np.asarray(result["family"])
    stage1_argmax = np.asarray(result["stage1_argmax"])
    path = np.asarray(result["path"])

    scored = path != "buffering_incomplete"
    known_mask = scored & (family != held_out_family)

    acc = (stage1_argmax[known_mask] == family[known_mask]).mean() if known_mask.any() else float("nan")
    labels = np.unique(family[known_mask]) if known_mask.any() else np.array([])
    f1 = f1_score(
        family[known_mask], stage1_argmax[known_mask], labels=labels,
        average="macro", zero_division=0,
    ) if known_mask.any() else float("nan")

    pred_set_size = np.asarray(result["gate1_pred_set_size"])[scored]
    singleton_rate = (pred_set_size == 1).mean()
    empty_rate = (pred_set_size == 0).mean()
    multi_rate = (pred_set_size > 1).mean()
    mean_set_size = pred_set_size.mean()

    return {
        "stage1_raw_accuracy": acc,
        "stage1_raw_macro_f1": f1,
        "singleton_rate": singleton_rate,
        "empty_set_rate": empty_rate,
        "multi_element_set_rate": multi_rate,
        "mean_prediction_set_size": mean_set_size,
    }


def gate1_coverage_report(result: dict, known_classes: list, alpha: float) -> list:
    """Coverage-validation check (paper: 'plots achieved per-class coverage
    against the nominal level for the Mondrian gate'). For each known class,
    the theoretical guarantee is: true label excluded from the prediction
    set with probability at most alpha, i.e. achieved coverage >= 1-alpha.
    """
    family = np.asarray(result["family"])
    true_in_set = np.asarray(result["gate1_true_in_set"])
    path = np.asarray(result["path"])
    scored = path != "buffering_incomplete"

    rows = []
    for c in list(known_classes) + ["Benign"]:
        mask = scored & (family == c)
        n = int(mask.sum())
        if n == 0:
            continue
        achieved = float(true_in_set[mask].mean())
        rows.append({
            "class": c, "n": n,
            "achieved_coverage": achieved,
            "target_coverage": 1 - alpha,
            "meets_target": achieved >= (1 - alpha),
        })
    return rows
