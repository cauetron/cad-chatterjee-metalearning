"""
Phase 6: Generate tables and figures for the report.

Reads:
  meta_learning/results_full.csv
  meta_learning/results_summary.csv
  meta_learning/statistical_tests.csv

Writes (./report_assets/):
  table1_performance.csv/md             all metrics, mean ± std
  table1_meta_ranking_metrics.csv/md    SRC, WRC, NDCG@3, Top1Hit, Regret@1
  table1_external_metrics.csv/md        ARI, AMI
  table2_wilcoxon.csv/md                Wilcoxon tests, run and dataset scopes
  fig1_bar_src.png                      bar chart: SRC by MF set and learner
  fig2_box_src.png                      boxplot: SRC distribution
  fig3_bar_wrc.png                      bar chart: WRC by MF set and learner
  fig4_bar_ndcg3.png                    bar chart: NDCG@3 by MF set and learner

Usage:
  python phase6_figures_tables.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MF_ORDER = ["Distance", "CaD-Sp", "CaD-Ke", "CaD-Ch"]
SUMMARY_ORDER = ["MeanRank", "MajorityRank", "Distance", "CaD-Sp", "CaD-Ke", "CaD-Ch"]
LEARNERS = ["kNN", "RF"]
RANKING_METRICS = ["SRC", "WRC", "NDCG@3", "Top1Hit", "Regret@1"]
EXTERNAL_METRICS = ["ARI", "AMI"]
METRICS = RANKING_METRICS + EXTERNAL_METRICS
LOWER_IS_BETTER = {"Regret@1"}


# -- TABLE 1 -------------------------------------------------------------------

def build_table1(summary, metrics=METRICS):
    """Format mean ± std as a report-ready table."""
    rows = []
    ordered = []
    for mf in SUMMARY_ORDER:
        for learner in (["Baseline"] if mf in ["MeanRank", "MajorityRank"] else LEARNERS):
            ordered.append((mf, learner))

    seen = set()
    for mf, learner in ordered + list(summary[["mf_set", "learner"]].itertuples(index=False, name=None)):
        if (mf, learner) in seen:
            continue
        seen.add((mf, learner))
        sub = summary[(summary["mf_set"] == mf) & (summary["learner"] == learner)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        row = {"Method / MF set": mf, "Meta-learner": learner}
        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in r and std_col in r:
                m = r[mean_col]
                s = r[std_col]
                row[metric] = f"{m:.3f} ± {s:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_md_table(df, path, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


# -- TABLE 2 -------------------------------------------------------------------

def build_table2(tests):
    """Format the Wilcoxon table for the report."""
    out = tests.copy()
    out["p_value"] = out["p_value"].apply(
        lambda p: "NA" if pd.isna(p) else "<0.0001" if p < 0.0001 else f"{p:.4f}"
    )
    out["significant"] = out["significant"].map({True: "✓", False: " "})
    out = out.rename(columns={
        "scope": "Scope",
        "learner": "Meta-learner",
        "metric": "Metric",
        "comparison": "Comparison",
        "n_pairs": "N pairs",
        "mean_baseline": "Mean baseline",
        "mean_proposal": "Mean proposal",
        "mean_delta_proposal_minus_baseline": "Mean Δ",
        "median_delta_proposal_minus_baseline": "Median Δ",
        "p_value": "p-value",
        "significant": "Sig. (α=0.05)",
        "winner": "Winner",
    })
    keep = [
        "Scope", "Meta-learner", "Metric", "Comparison", "N pairs",
        "Mean baseline", "Mean proposal", "Mean Δ", "Median Δ",
        "p-value", "Sig. (α=0.05)", "Winner",
    ]
    out = out[[c for c in keep if c in out.columns]]
    for c in ["Mean baseline", "Mean proposal", "Mean Δ", "Median Δ"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
    return out


# -- FIGURES -------------------------------------------------------------------

def fig_bar_metric(summary, metric, output_path, ylabel=None, title=None, ylim=None):
    """Bar chart for a selected metric by MF set and learner."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    x = np.arange(len(MF_ORDER))
    width = 0.38

    for offset, learner in zip([-width/2, width/2], LEARNERS):
        means, stds = [], []
        for mf in MF_ORDER:
            sub = summary[(summary["mf_set"] == mf) & (summary["learner"] == learner)]
            means.append(sub[f"{metric}_mean"].iloc[0] if not sub.empty and f"{metric}_mean" in sub else np.nan)
            stds.append(sub[f"{metric}_std"].iloc[0] if not sub.empty and f"{metric}_std" in sub else np.nan)
        ax.bar(x + offset, means, width, yerr=stds, label=learner, capsize=3)

    ax.set_title(title or f"{metric} por conjunto de meta-features")
    ax.set_ylabel(ylabel or metric)
    ax.set_xticks(x)
    ax.set_xticklabels(MF_ORDER, rotation=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Meta-aprendiz")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_box_metric(full, metric, output_path):
    """Boxplot of a per-prediction metric by MF set, side-by-side for kNN and RF."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, learner in zip(axes, LEARNERS):
        data = [
            full[(full["learner"] == learner) & (full["mf_set"] == mf)][metric].dropna().values
            for mf in MF_ORDER
        ]
        ax.boxplot(data, tick_labels=MF_ORDER, widths=0.6)
        ax.set_title(f"Distribuição de {metric} — {learner}")
        ax.axhline(0, linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"Distribuição de {metric} por predição na validação cruzada repetida",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -- MAIN ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="./meta_learning")
    parser.add_argument("--output-dir", default="./report_assets")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(in_dir / "results_summary.csv")
    full    = pd.read_csv(in_dir / "results_full.csv")
    tests   = pd.read_csv(in_dir / "statistical_tests.csv")

    # Table 1: all metrics
    t1_all = build_table1(summary, METRICS)
    t1_all.to_csv(out_dir / "table1_performance.csv", index=False)
    write_md_table(t1_all, out_dir / "table1_performance.md",
                   "Table 1 — Meta-learning performance (mean ± std)")
    print("Wrote table1_performance.{csv,md}")

    # Table 1a: ranking-level metrics only
    t1_rank = build_table1(summary, RANKING_METRICS)
    t1_rank.to_csv(out_dir / "table1_meta_ranking_metrics.csv", index=False)
    write_md_table(t1_rank, out_dir / "table1_meta_ranking_metrics.md",
                   "Table 1a — Meta-level ranking metrics (mean ± std)")
    print("Wrote table1_meta_ranking_metrics.{csv,md}")

    # Table 1b: external partition metrics only
    t1_ext = build_table1(summary, EXTERNAL_METRICS)
    t1_ext.to_csv(out_dir / "table1_external_metrics.csv", index=False)
    write_md_table(t1_ext, out_dir / "table1_external_metrics.md",
                   "Table 1b — External partition metrics for top-1 recommendation (mean ± std)")
    print("Wrote table1_external_metrics.{csv,md}")

    # Table 2
    t2 = build_table2(tests)
    t2.to_csv(out_dir / "table2_wilcoxon.csv", index=False)
    write_md_table(t2, out_dir / "table2_wilcoxon.md",
                   "Table 2 — Paired Wilcoxon signed-rank tests (α = 0.05)")
    print("Wrote table2_wilcoxon.{csv,md}")

    # Figures
    fig_bar_metric(summary, "SRC", out_dir / "fig1_bar_src.png",
                   ylabel="SRC médio (folds × repetições)",
                   title="Similaridade de ranking (SRC) por conjunto de meta-features",
                   ylim=(0.0, 1.0))
    print("Wrote fig1_bar_src.png")

    fig_box_metric(full, "SRC", out_dir / "fig2_box_src.png")
    print("Wrote fig2_box_src.png")

    if "WRC_mean" in summary.columns:
        fig_bar_metric(summary, "WRC", out_dir / "fig3_bar_wrc.png",
                       ylabel="WRC médio (folds × repetições)",
                       title="Correlação de ranking ponderada no topo (WRC)",
                       ylim=(-1.0, 1.0))
        print("Wrote fig3_bar_wrc.png")

    if "NDCG@3_mean" in summary.columns:
        fig_bar_metric(summary, "NDCG@3", out_dir / "fig4_bar_ndcg3.png",
                       ylabel="NDCG@3 médio (folds × repetições)",
                       title="Qualidade do topo do ranking (NDCG@3)",
                       ylim=(0.0, 1.05))
        print("Wrote fig4_bar_ndcg3.png")

    print(f"\nAll assets in {out_dir}/")


if __name__ == "__main__":
    main()
