"""LOFO (Leave-One-Family-Out) DIAGNOSTIC 3 -- for each held-out family's
TRUE zero-day flows, how did Gate 1 (LightGBM + Mondrian) route them?

  - "null" (empty prediction set)      -> correctly deferred to LSTM-AE
  - "multi" (multi-element set)        -> correctly deferred to LSTM-AE
  - "singleton" (single confident class) -> WRONGLY exits at Gate 1,
                                             never reaches LSTM-AE at all
                                             (this is the paper's "Path B"
                                             evasion channel)

"Correctly passed to LSTM-AE" = null + multi (i.e. NOT singleton).

LIGHTER than the other two diagnostics: reads ONLY the already-saved
result.joblib per rotation (gate1_pred_set_size column) -- no model
reloading, no CSV loading, no inference at all. Pure read of existing
pipeline output.

Read-only: does not retrain, does not touch summary.md, variance_table, or
any other script's output. Writes to a NEW isolated folder
(artifacts/diagnostics/lofo_gate1_routing/).

Usage:
    python -m scripts.diagnose_lofo_gate1_routing --config configs/mvp.yaml
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.family_map import KNOWN_FAMILIES

OUT_DIR = Path("artifacts") / "diagnostics" / "lofo_gate1_routing"


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for family in KNOWN_FAMILIES:
        result_path = Path("artifacts") / family / "result.joblib"
        if not result_path.exists():
            print(f"{family}: skipped -- {result_path} not found")
            continue

        print(f"{family}: loading result.joblib (read-only)...")
        result = joblib.load(result_path)
        fam_arr = np.asarray(result["family"])
        pred_set_size = np.asarray(result["gate1_pred_set_size"])
        path_arr = np.asarray(result["path"])

        # Only this rotation's TRUE zero-day rows (the held-out family),
        # excluding incomplete-buffer leftovers (not a real routing outcome).
        zd_mask = (fam_arr == family) & (path_arr != "buffering_incomplete")
        sizes = pred_set_size[zd_mask]
        n = len(sizes)
        if n == 0:
            print(f"  skipped -- no zero-day rows found")
            continue

        n_null = int((sizes == 0).sum())
        n_singleton = int((sizes == 1).sum())
        n_multi = int((sizes > 1).sum())
        n_correctly_deferred = n_null + n_multi  # null + multi = passed to LSTM-AE

        rows.append({
            "family": family, "n_total": n,
            "n_null": n_null, "frac_null": n_null / n,
            "n_singleton": n_singleton, "frac_singleton": n_singleton / n,
            "n_multi": n_multi, "frac_multi": n_multi / n,
            "n_correctly_deferred": n_correctly_deferred,
            "frac_correctly_deferred": n_correctly_deferred / n,
        })
        print(f"  n={n}  null={n_null} ({n_null/n:.2%})  singleton={n_singleton} ({n_singleton/n:.2%})  "
              f"multi={n_multi} ({n_multi/n:.2%})  -> correctly passed to LSTM-AE: {n_correctly_deferred/n:.2%}")

    if not rows:
        print("\nNo data found -- has run_full_study.py been run yet?")
        return

    # Stacked bar chart: null + multi (correctly deferred) vs singleton (escaped)
    families = [r["family"] for r in rows]
    frac_null = [r["frac_null"] for r in rows]
    frac_multi = [r["frac_multi"] for r in rows]
    frac_singleton = [r["frac_singleton"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(families, frac_null, label="Null (empty set) -- correctly deferred",
           color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(families, frac_multi, bottom=frac_null, label="Multi-class confused -- correctly deferred",
           color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(families, frac_singleton, bottom=[n + m for n, m in zip(frac_null, frac_multi)],
           label="Singleton -- WRONGLY exits (Path B, never reaches LSTM-AE)",
           color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fraction of LOFO zero-day flows")
    ax.set_ylim(0, 1.05)
    ax.set_title("Gate 1 routing outcome for TRUE zero-day flows, per LOFO family\n"
                 "(green+blue = correctly passed to LSTM-AE; red = escaped Gate 1 undetected)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate1_routing_stacked_bar.png", dpi=120)
    plt.close(fig)

    with open(OUT_DIR / "result.md", "w") as f:
        f.write("# LOFO Gate 1 Routing Breakdown -- null/multi/singleton, per held-out family\n\n")
        f.write("For each family's TRUE zero-day flows only (not the other 4 known families, not benign). "
                "'Correctly passed to LSTM-AE' = null (empty set) + multi (ambiguous set). "
                "'Singleton' = Gate 1 wrongly confident, never reaches Gate 2 at all -- the paper's Path B.\n\n")
        f.write("| Family | n_total | Null (empty) | Singleton (escaped) | Multi (confused) | Correctly deferred |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['family']} | {r['n_total']} | "
                    f"{r['n_null']} ({r['frac_null']:.2%}) | "
                    f"{r['n_singleton']} ({r['frac_singleton']:.2%}) | "
                    f"{r['n_multi']} ({r['frac_multi']:.2%}) | "
                    f"{r['n_correctly_deferred']} ({r['frac_correctly_deferred']:.2%}) |\n")
        f.write("\nPlot: `gate1_routing_stacked_bar.png`.\n")
        f.write("\nNote: this reads only the already-saved result.joblib from run_full_study.py -- "
                "no retraining, no model reloading, no data loading.\n")

    print(f"\nSaved: {OUT_DIR}/result.md, {OUT_DIR}/gate1_routing_stacked_bar.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    main(args.config)
