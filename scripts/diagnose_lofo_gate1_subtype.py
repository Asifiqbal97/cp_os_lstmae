"""Checks whether the DDoS/DoS Gate-1 escape asymmetry (found in
diagnose_lofo_gate1_routing.py) is concentrated in specific attack
SUBTYPES, and -- critically -- what label singleton-escaped rows actually
exit as (testing the hypothesis that DDoS subtypes escape specifically AS
"DoS", not as something else).

result.joblib doesn't store Attack_type (only family), so this recomputes
Gate 1's routing decision fresh -- READ-ONLY, no retraining: reloads the
already-trained LightGBM + Mondrian calibration for each rotation, rebuilds
the SAME deterministic test split (same seed -> identical rows), and since
the test set is never shuffled, aligns each row back to its raw Attack_type
from the CSV directly.

Writes to a NEW isolated folder (artifacts/diagnostics/lofo_gate1_subtype/)
-- does not touch summary.md, result.joblib, variance_table, or any other
script's output.

Usage:
    python -m scripts.diagnose_lofo_gate1_subtype --config configs/mvp.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.loader import load_csv
from src.data.family_map import KNOWN_FAMILIES
from src.data.zero_day_split import make_zero_day_split
from src.analysis import load_rotation_artifacts
from src.stage1_classifier.train import gate1_predict

OUT_DIR = Path("artifacts") / "diagnostics" / "lofo_gate1_subtype"


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data (read-only)...")
    train_pool = load_csv(cfg["train_csv"])
    test_pool = load_csv(cfg["test_csv"])
    alpha1 = cfg["alpha_stage1"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_family_rows = {}

    for family in KNOWN_FAMILIES:
        print(f"\n{family}: reloading Gate 1 artifacts (read-only, no retraining)...")
        booster, label_encoder, nonconf_by_class, _ = load_rotation_artifacts(family)

        print(f"  rebuilding deterministic split (same seed)...")
        split = make_zero_day_split(train_pool, test_pool, family, random_state=cfg["random_state"])

        # Test set is never shuffled (see zero_day_split.py) -- test_pool's
        # row order matches split.X_te_s exactly, so Attack_type aligns.
        family_mask_in_testpool = (test_pool["family"] == family).values
        subtypes = test_pool.loc[family_mask_in_testpool, "Attack_type"].values
        proba = booster.predict(split.X_te_s[family_mask_in_testpool],
                                 num_iteration=booster.best_iteration)

        subtype_stats = {}
        for i, subtype in enumerate(subtypes):
            pred_set, argmax_cls, _ = gate1_predict(proba[i], label_encoder, nonconf_by_class, alpha1)
            if subtype not in subtype_stats:
                subtype_stats[subtype] = {"null": 0, "singleton": 0, "multi": 0,
                                           "singleton_labels": Counter(), "n": 0}
            s = subtype_stats[subtype]
            s["n"] += 1
            if len(pred_set) == 0:
                s["null"] += 1
            elif len(pred_set) == 1:
                s["singleton"] += 1
                s["singleton_labels"][next(iter(pred_set))] += 1
            else:
                s["multi"] += 1

        all_family_rows[family] = subtype_stats
        print(f"  {len(subtype_stats)} subtype(s) found: {list(subtype_stats.keys())}")
        for subtype, s in subtype_stats.items():
            top_label = s["singleton_labels"].most_common(1)
            top_label_str = f"{top_label[0][0]} ({top_label[0][1]}/{s['singleton']})" if top_label else "n/a"
            print(f"    {subtype}: n={s['n']}  null={s['null']} ({s['null']/s['n']:.1%})  "
                  f"singleton={s['singleton']} ({s['singleton']/s['n']:.1%}, exits mostly as {top_label_str})  "
                  f"multi={s['multi']} ({s['multi']/s['n']:.1%})")

    # Plot + report only for families with >1 subtype (single-subtype
    # families like Spoofing have nothing to break down -- consistent with
    # earlier diagnose_subtype_scores.py behavior)
    md_lines = ["# LOFO Gate 1 Routing by Subtype -- which subtypes escape, and as what label\n"]
    for family, subtype_stats in all_family_rows.items():
        if len(subtype_stats) <= 1:
            md_lines.append(f"## {family}\nOnly one subtype -- no breakdown possible.\n")
            continue

        subtypes = list(subtype_stats.keys())
        null_frac = [subtype_stats[s]["null"] / subtype_stats[s]["n"] for s in subtypes]
        singleton_frac = [subtype_stats[s]["singleton"] / subtype_stats[s]["n"] for s in subtypes]
        multi_frac = [subtype_stats[s]["multi"] / subtype_stats[s]["n"] for s in subtypes]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(subtypes, null_frac, label="Null -- correctly deferred", color="#55A868",
               edgecolor="black", linewidth=0.5)
        ax.bar(subtypes, multi_frac, bottom=null_frac, label="Multi -- correctly deferred",
               color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax.bar(subtypes, singleton_frac, bottom=[n + m for n, m in zip(null_frac, multi_frac)],
               label="Singleton -- escapes Gate 1", color="#C44E52", edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"LOFO {family}: Gate 1 routing by subtype")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{family}_subtype_routing.png", dpi=120)
        plt.close(fig)

        md_lines.append(f"## {family}\n")
        md_lines.append("| Subtype | n | Null | Singleton | Multi | Most common singleton exit label |")
        md_lines.append("|---|---|---|---|---|---|")
        for s in subtypes:
            st = subtype_stats[s]
            top = st["singleton_labels"].most_common(1)
            top_str = f"{top[0][0]} ({top[0][1]}/{st['singleton']})" if top else "n/a"
            md_lines.append(f"| {s} | {st['n']} | {st['null']} ({st['null']/st['n']:.1%}) | "
                            f"{st['singleton']} ({st['singleton']/st['n']:.1%}) | "
                            f"{st['multi']} ({st['multi']/st['n']:.1%}) | {top_str} |")
        md_lines.append(f"\nPlot: `{family}_subtype_routing.png`\n")

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved: {OUT_DIR}/result.md + plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
