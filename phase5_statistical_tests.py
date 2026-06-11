"""
Phase 5: Paired statistical tests of meta-feature sets.

For each (meta-learner × metric), runs Wilcoxon signed-rank tests comparing
paired scores across meta-feature sets. Two pairing scopes are reported:

  1. run:     pairs are (repetition, fold, dataset_id). This is useful as a
              descriptive repeated-CV analysis, but the pairs are not fully
              independent because the same datasets reappear across repetitions.
  2. dataset: scores are first averaged over repetitions/folds per dataset.
              This is the statistically more conservative analysis, with one
              paired observation per dataset.

Main question: is CaD-Chatterjee (the proposal) better than CaD-Spearman
(the Pimentel baseline)? CaD-Kendall is kept as a rank-based control.

Outputs (./meta_learning/):
  - statistical_tests.csv             combined run + dataset analyses
  - statistical_tests_by_run.csv      repeated-CV paired analysis
  - statistical_tests_by_dataset.csv  dataset-aggregated paired analysis
  - run_metadata_phase5.json

Usage:
  python phase5_statistical_tests.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COMPARISONS = [
    ("CaD-Sp", "CaD-Ch", "main: Chatterjee vs Spearman baseline"),
    ("CaD-Sp", "CaD-Ke", "control: Kendall vs Spearman baseline"),
]
METRICS = ["SRC", "WRC", "NDCG@3", "Top1Hit", "Regret@1", "ARI", "AMI"]
LOWER_IS_BETTER = {"Regret@1"}
LEARNERS = ["kNN", "RF"]
PAIR_KEY = ["repetition", "fold", "dataset_id"]


def run_wilcoxon(a, b, alpha, lower_is_better=False):
    """Return p-value and a conservative winner label based on mean and p-value.

    diff is always proposal - baseline. For metrics in which lower values are
    better, winner selection is inverted, but the reported delta is unchanged.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan, "tie", np.nan, np.nan
    diff = b - a
    try:
        _, p = wilcoxon(diff, zero_method="zsplit")
    except ValueError:
        p = 1.0
    p = float(p)
    mean_delta = float(np.mean(diff))
    median_delta = float(np.median(diff))
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    if p < alpha:
        if lower_is_better:
            winner = "proposal" if mean_b < mean_a else "baseline" if mean_a < mean_b else "tie"
        else:
            winner = "proposal" if mean_b > mean_a else "baseline" if mean_a > mean_b else "tie"
    else:
        winner = "tie"
    return p, winner, mean_delta, median_delta


def build_wide(full, learner, metric, scope):
    """Build a wide table with one row per paired unit and one column per MF set."""
    sub = full[full["learner"] == learner]
    if scope == "run":
        long = sub[PAIR_KEY + ["mf_set", metric]]
        return long.pivot_table(index=PAIR_KEY, columns="mf_set", values=metric, aggfunc="first")
    if scope == "dataset":
        long = (
            sub.groupby(["dataset_id", "mf_set"], as_index=False)[metric]
            .mean(numeric_only=True)
        )
        return long.pivot_table(index="dataset_id", columns="mf_set", values=metric, aggfunc="first")
    raise ValueError("scope must be 'run' or 'dataset'")


def test_scope(full, scope, alpha):
    rows = []
    for learner in LEARNERS:
        if learner not in set(full["learner"]):
            continue
        for metric in METRICS:
            if metric not in full.columns:
                continue
            wide = build_wide(full, learner, metric, scope)
            for baseline, proposal, desc in COMPARISONS:
                needed = [baseline, proposal]
                if not all(c in wide.columns for c in needed):
                    continue
                paired = wide[needed].dropna()
                a = paired[baseline].values
                b = paired[proposal].values
                p, winner_code, mean_delta, median_delta = run_wilcoxon(
                    a, b, alpha, lower_is_better=(metric in LOWER_IS_BETTER)
                )
                winner = (
                    proposal if winner_code == "proposal"
                    else baseline if winner_code == "baseline"
                    else "tie"
                )
                rows.append({
                    "scope": scope,
                    "learner": learner,
                    "metric": metric,
                    "comparison": f"{baseline} vs {proposal}",
                    "description": desc,
                    "n_pairs": int(len(paired)),
                    "mean_baseline": float(np.mean(a)) if len(a) else np.nan,
                    "mean_proposal": float(np.mean(b)) if len(b) else np.nan,
                    "mean_delta_proposal_minus_baseline": mean_delta,
                    "median_delta_proposal_minus_baseline": median_delta,
                    "p_value": p,
                    "significant": bool(np.isfinite(p) and p < alpha),
                    "winner": winner,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./meta_learning/results_full.csv")
    parser.add_argument("--output-dir", default="./meta_learning")
    parser.add_argument("--alpha",  type=float, default=0.05)
    args = parser.parse_args()

    full = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_run = test_scope(full, "run", args.alpha)
    by_dataset = test_scope(full, "dataset", args.alpha)
    combined = pd.concat([by_run, by_dataset], ignore_index=True)

    by_run.to_csv(out_dir / "statistical_tests_by_run.csv", index=False)
    by_dataset.to_csv(out_dir / "statistical_tests_by_dataset.csv", index=False)
    combined.to_csv(out_dir / "statistical_tests.csv", index=False)

    metadata = {
        "phase": 5,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "output_dir": str(out_dir),
        "alpha": args.alpha,
        "metrics": METRICS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "comparisons": [f"{a} vs {b}" for a, b, _ in COMPARISONS],
        "scopes": ["run", "dataset"],
        "note": "dataset scope aggregates repeated CV scores by dataset before Wilcoxon and is the conservative test.",
    }
    with open(out_dir / "run_metadata_phase5.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(combined.to_string(index=False, float_format="%.4f"))
    print(f"\nWrote {out_dir / 'statistical_tests.csv'}")
    print(f"Wrote {out_dir / 'statistical_tests_by_run.csv'}")
    print(f"Wrote {out_dir / 'statistical_tests_by_dataset.csv'}")


if __name__ == "__main__":
    main()
