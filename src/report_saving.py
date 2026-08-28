"""Saves rotation results as .json (machine-readable) and .md (human-readable
report), so terminal output isn't the only record of a run.
"""

import json
from pathlib import Path

import numpy as np


def to_jsonable(obj):
    """Recursively converts numpy scalars/arrays (not JSON-serializable by
    default) to native Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def save_rotation_report(artifacts_dir: Path, held_out_family: str, gate1_raw: dict,
                          gate1_coverage: list, pipeline_metrics: dict, paths: dict, cfg: dict):
    bundle = {
        "held_out_family": held_out_family,
        "config": {k: cfg[k] for k in
                   ["alpha_stage1", "alpha_stage2", "score_alpha", "seq_len", "min_cluster_size"]},
        "gate1_raw": gate1_raw,
        "gate1_coverage": gate1_coverage,
        "pipeline_metrics": pipeline_metrics,
        "path_decomposition": paths,
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(to_jsonable(bundle), f, indent=2)

    md = _rotation_markdown(held_out_family, gate1_raw, gate1_coverage, pipeline_metrics, paths, cfg)
    with open(artifacts_dir / "report.md", "w") as f:
        f.write(md)


def _rotation_markdown(held_out_family, gate1_raw, gate1_coverage, pipeline_metrics, paths, cfg) -> str:
    lines = [f"# CP-OSR-LSTMAE — Rotation Report: held out `{held_out_family}`\n"]

    lines.append("## Gate 1 (LightGBM + Mondrian)\n")
    lines.append(f"- Raw classifier accuracy (bypassing conformal gate): "
                 f"**{gate1_raw['stage1_raw_accuracy']:.4f}**")
    lines.append(f"- Raw classifier macro-F1: **{gate1_raw['stage1_raw_macro_f1']:.4f}**")
    lines.append(f"- Singleton rate: {gate1_raw['singleton_rate']:.4f}")
    lines.append(f"- Empty-set rate: {gate1_raw['empty_set_rate']:.4f}")
    lines.append(f"- Multi-element rate: {gate1_raw['multi_element_set_rate']:.4f}")
    lines.append(f"- Mean prediction set size: {gate1_raw['mean_prediction_set_size']:.3f}\n")

    lines.append(f"### Per-class coverage validation (target = {1 - cfg['alpha_stage1']:.2f})\n")
    lines.append("| class | n | achieved | target | meets target? |")
    lines.append("|---|---|---|---|---|")
    for row in gate1_coverage:
        lines.append(f"| {row['class']} | {row['n']} | {row['achieved_coverage']:.4f} | "
                     f"{row['target_coverage']:.2f} | {'yes' if row['meets_target'] else '**NO**'} |")
    lines.append("")

    lines.append("## Gate 2 (LSTM-AE + split-conformal)\n")
    lines.append(f"- Benign false alarm rate: **{pipeline_metrics['benign_false_alarm_rate']:.4f}** "
                 f"(target alpha={cfg['alpha_stage2']:.2f})")
    lines.append(f"- Zero-day recall: **{pipeline_metrics['zero_day_recall']:.4f}**\n")

    lines.append(f"## Open-set path decomposition (n={paths['n_zero_day_rows']} zero-day rows)\n")
    lines.append("| path | count | fraction |")
    lines.append("|---|---|---|")
    label_map = {
        "A_correctly_novel": "A — correctly surfaced as novel",
        "B_confident_known_attack": "B — confidently exited as known attack",
        "B0_absorbed_benign": "B0 — absorbed into Benign (most dangerous)",
        "C_reverted_known_attack": "C — reverted to known attack after deferral",
    }
    for k, label in label_map.items():
        lines.append(f"| {label} | {paths['counts'][k]} | {paths['fractions'][k]:.4f} |")
    lines.append("")

    lines.append("## Full pipeline (final verdict, all gates combined)\n")
    lines.append(f"- Known-class accuracy: **{pipeline_metrics['known_class_accuracy']:.4f}**")
    lines.append(f"- Known-class macro-F1: **{pipeline_metrics['known_class_macro_f1']:.4f}**")
    lines.append(f"- Path counts: `{pipeline_metrics['path_counts']}`")

    return "\n".join(lines) + "\n"


def save_summary_report(artifacts_root: Path, all_results: dict):
    metrics_to_summarize = [
        ("known_class_accuracy", "pipeline_metrics"),
        ("known_class_macro_f1", "pipeline_metrics"),
        ("zero_day_recall", "pipeline_metrics"),
        ("benign_false_alarm_rate", "pipeline_metrics"),
    ]
    summary = {"rotations": {}, "mean": {}, "std": {}}
    values_by_metric = {m: [] for m, _ in metrics_to_summarize}

    for family, res in all_results.items():
        row = {}
        for m, section in metrics_to_summarize:
            v = res[section][m]
            values_by_metric[m].append(v)
            row[m] = v
        row["path_decomposition_fractions"] = res["path_decomposition"]["fractions"]
        summary["rotations"][family] = row

    for m, _ in metrics_to_summarize:
        vals = values_by_metric[m]
        summary["mean"][m] = float(np.mean(vals))
        summary["std"][m] = float(np.std(vals))

    with open(artifacts_root / "summary.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    lines = ["# CP-OSR-LSTMAE — Summary Across All Rotations\n"]
    lines.append("| family | accuracy | macro-F1 | zero-day recall | benign FAR |")
    lines.append("|---|---|---|---|---|")
    for family, row in summary["rotations"].items():
        lines.append(f"| {family} | {row['known_class_accuracy']:.4f} | "
                     f"{row['known_class_macro_f1']:.4f} | {row['zero_day_recall']:.4f} | "
                     f"{row['benign_false_alarm_rate']:.4f} |")
    lines.append(f"| **mean** | {summary['mean']['known_class_accuracy']:.4f} | "
                 f"{summary['mean']['known_class_macro_f1']:.4f} | "
                 f"{summary['mean']['zero_day_recall']:.4f} | "
                 f"{summary['mean']['benign_false_alarm_rate']:.4f} |")
    lines.append(f"| **std** | {summary['std']['known_class_accuracy']:.4f} | "
                 f"{summary['std']['known_class_macro_f1']:.4f} | "
                 f"{summary['std']['zero_day_recall']:.4f} | "
                 f"{summary['std']['benign_false_alarm_rate']:.4f} |\n")

    lines.append("## Open-set path decomposition per rotation (fractions)\n")
    lines.append("| family | A (novel) | B (known attack) | B0 (absorbed benign) | C (reverted) |")
    lines.append("|---|---|---|---|---|")
    for family, row in summary["rotations"].items():
        f = row["path_decomposition_fractions"]
        lines.append(f"| {family} | {f['A_correctly_novel']:.4f} | {f['B_confident_known_attack']:.4f} | "
                     f"{f['B0_absorbed_benign']:.4f} | {f['C_reverted_known_attack']:.4f} |")

    with open(artifacts_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def save_shared_ae_report(artifacts_dir: Path, loss_history: list, recon_errors, gate2_bundle: dict, cfg: dict):
    """M2/M3 report: loss curve, reconstruction-error distribution, Gate 2
    calibration bundle -- for the ONE shared AE, not per-rotation."""
    recon_errors = np.asarray(recon_errors)
    recon_stats = {
        "mean": float(recon_errors.mean()), "std": float(recon_errors.std()),
        "p50": float(np.percentile(recon_errors, 50)), "p95": float(np.percentile(recon_errors, 95)),
        "p99": float(np.percentile(recon_errors, 99)), "max": float(recon_errors.max()),
    }
    bundle = {
        "config": {k: cfg[k] for k in ["alpha_stage2", "score_alpha", "seq_len", "hidden_dim", "latent_dim"]},
        "loss_history": loss_history,
        "reconstruction_error_stats": recon_stats,
        "gate2_calibration": {k: v for k, v in gate2_bundle.items() if k != "centroid"},
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(to_jsonable(bundle), f, indent=2)

    with open(artifacts_dir / "loss_curve.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in loss_history:
            f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['val_loss']:.6f}\n")

    lines = ["# CP-OSR-LSTMAE — Shared AE (Gate 2) Report\n"]
    lines.append(f"Trained once on benign-only data, reused across all zero-day rotations.\n")
    lines.append("## Training\n")
    lines.append(f"- Epochs run: {len(loss_history)} (of {cfg['stage2_epochs']} max, early stopping may apply)")
    lines.append(f"- Final train loss: {loss_history[-1]['train_loss']:.5f}")
    lines.append(f"- Final val loss: {loss_history[-1]['val_loss']:.5f}")
    lines.append(f"- Full loss curve: see `loss_curve.csv`\n")
    lines.append("## Reconstruction error (on all benign training sequences)\n")
    lines.append("| stat | value |")
    lines.append("|---|---|")
    for k, v in recon_stats.items():
        lines.append(f"| {k} | {v:.5f} |")
    lines.append("")
    lines.append("## Gate 2 calibration (benign-only)\n")
    lines.append(f"- Split-conformal threshold: **{gate2_bundle['threshold']:.4f}**")
    lines.append(f"- Cluster radius (for Gate 3): {gate2_bundle['cluster_radius']:.4f}")
    lines.append(f"- MSE p99 (normalization): {gate2_bundle['mse_p99']:.4f}")
    lines.append(f"- Latent-deviation p99 (normalization): {gate2_bundle['ldev_p99']:.4f}")

    with open(artifacts_dir / "report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def save_closed_set_report(artifacts_dir: Path, report: dict, cm, class_names):
    """M6 report: closed-set LightGBM, all known families, no holdout.
    Plain classification_report + confusion matrix, no conformal gate."""
    bundle = {"classification_report": report, "confusion_matrix": cm.tolist(),
              "class_names": list(class_names)}
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(to_jsonable(bundle), f, indent=2)

    lines = ["# CP-OSR-LSTMAE — Closed-Set LightGBM Report (no zero-day holdout)\n"]
    lines.append("Plain classifier evaluation on the official test split, all 5 known "
                 "families present in training. No conformal gate applied.\n")
    lines.append("## Per-class metrics\n")
    lines.append("| class | precision | recall | f1-score | support |")
    lines.append("|---|---|---|---|---|")
    for cls in class_names:
        r = report[cls]
        lines.append(f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1-score']:.4f} | {int(r['support'])} |")
    lines.append(f"| **macro avg** | {report['macro avg']['precision']:.4f} | "
                 f"{report['macro avg']['recall']:.4f} | {report['macro avg']['f1-score']:.4f} | "
                 f"{int(report['macro avg']['support'])} |")
    lines.append(f"| **weighted avg** | {report['weighted avg']['precision']:.4f} | "
                 f"{report['weighted avg']['recall']:.4f} | {report['weighted avg']['f1-score']:.4f} | "
                 f"{int(report['weighted avg']['support'])} |")
    lines.append(f"\n**Overall accuracy: {report['accuracy']:.4f}**\n")

    lines.append("## Confusion matrix\n")
    lines.append("| true \\ pred | " + " | ".join(class_names) + " |")
    lines.append("|---" * (len(class_names) + 1) + "|")
    for i, cls in enumerate(class_names):
        lines.append(f"| **{cls}** | " + " | ".join(str(x) for x in cm[i]) + " |")

    with open(artifacts_dir / "report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def save_analysis_report(artifacts_dir: Path, held_out_family: str, coverage_result: dict,
                          gate1_curves: dict, gate2_roc: dict, alpha1: float):
    """M4-M6 report: coverage-validation table + ROC AUCs, for one rotation."""
    bundle = {
        "held_out_family": held_out_family,
        "coverage_validation": coverage_result,
        "gate1_roc_auc": {cls: c["auc"] for cls, c in gate1_curves.items()},
        "gate2_roc": {k: v for k, v in (gate2_roc or {}).items() if k not in ("fpr", "tpr")},
    }
    with open(artifacts_dir / "analysis_metrics.json", "w") as f:
        json.dump(to_jsonable(bundle), f, indent=2)

    lines = [f"# CP-OSR-LSTMAE — Coverage Validation & ROC: held out `{held_out_family}`\n"]

    lines.append("## Coverage validation (Mondrian vs marginal-CP vs MSP)\n")
    lines.append(f"MSP calibrated threshold: {coverage_result['t_msp']:.4f}  |  "
                 f"Marginal-CP calibrated threshold: {coverage_result['t_marginal']:.4f}\n")
    lines.append("| class | n_test | n_calib | Mondrian | Marginal-CP | MSP | target | Beta band (Mondrian) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in coverage_result["rows"]:
        lo, hi = r["beta_tolerance_band"]
        band_str = f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "n/a"
        lines.append(f"| {r['class']} | {r['n_test']} | {r['n_calib']} | "
                     f"{r['mondrian_coverage']:.4f} | {r['marginal_cp_coverage']:.4f} | "
                     f"{r['msp_coverage']:.4f} | {r['target_coverage']:.2f} | {band_str} |")
    lines.append("\nSee `plots/coverage_validation.png` for the visual comparison.\n")

    lines.append("## Gate 1 ROC (one-vs-rest per known class)\n")
    lines.append("| class | AUC |")
    lines.append("|---|---|")
    for cls, c in gate1_curves.items():
        lines.append(f"| {cls} | {c['auc']:.4f} |")
    lines.append("\nSee `plots/gate1_roc.png`.\n")

    lines.append("## Gate 2 ROC (Benign vs held-out zero-day family)\n")
    if gate2_roc:
        lines.append(f"- AUC: **{gate2_roc['auc']:.4f}**")
        lines.append(f"- Benign windows: {gate2_roc['n_benign_windows']}, "
                     f"zero-day windows: {gate2_roc['n_zeroday_windows']}")
        lines.append("\nSee `plots/gate2_roc.png`.\n")
    else:
        lines.append("- Not enough data to compute (insufficient benign or zero-day sequences).\n")

    with open(artifacts_dir / "analysis_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
