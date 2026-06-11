"""
Phase 4: Meta-learning experiments.

For each of the 4 meta-feature sets (Distance, CaD-Sp, CaD-Ke, CaD-Ch) and each
of the 2 meta-learners (k-NN with average ranking, Random Forest regression):
  - Run repeated K-fold cross-validation.
  - In each fold:
      * fit StandardScaler on train MFs only
      * fit meta-learner on (scaled train MFs, train rankings)
      * predict a ranking for each test dataset
      * compute SRC between predicted and true rankings
      * compute ARI and AMI of the predicted top-1 algorithm's partition
        against the true class partition

This phase also adds two simple fold-safe baselines trained only on the training
rankings of each fold:
  - MeanRank: average the training ranks per algorithm and re-rank the averages.
  - MajorityRank: use the most frequent complete ranking in the training fold.

Kendall tau was intentionally removed from the evaluation metrics to stay closer
to Pimentel (2019), but CaD-Ke is kept as a meta-feature set in Phase 3 as a
rank-based control.

Outputs (./meta_learning/):
  - results_full.csv             one row per prediction
  - results_summary.csv          mean ± std per (mf_set, learner, metric)
  - run_metadata_phase4.json     execution metadata

Usage:
  python phase4_meta_learning.py
  python phase4_meta_learning.py --repetitions 1   # quick test
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, weightedtau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

MF_FILES = {
    "Distance": "mf_distance.csv",
    "CaD-Sp":   "mf_cad_spearman.csv",
    "CaD-Ke":   "mf_cad_kendall.csv",
    "CaD-Ch":   "mf_cad_chatterjee.csv",
}
ALGO_COLS = ["KM", "WA", "AA", "GMd", "GMf"]
METRICS = ["SRC", "WRC", "NDCG@3", "Top1Hit", "Regret@1", "ARI", "AMI"]
K_NEIGHBORS = 5
N_TREES = 100
N_REPETITIONS = 20
N_FOLDS = 10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def rerank(values):
    """Convert scores/average positions to ranks 1..A; lower is better."""
    return pd.Series(values).rank(method="average").values


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------

def predict_knn_average_ranking(X_train, R_train, X_test, k):
    """For each test instance, find k nearest training instances in MF space,
    average their rankings, and reassign ranks 1..A.

    X_train, X_test: scaled MF matrices
    R_train: training rankings (n_train × n_algos)
    """
    diff = X_test[:, None, :] - X_train[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (n_test, n_train)
    preds = np.empty((X_test.shape[0], R_train.shape[1]))
    k_eff = min(k, X_train.shape[0])
    for i in range(X_test.shape[0]):
        nbr_idx = np.argsort(dists[i])[:k_eff]
        avg_ranks = R_train[nbr_idx].mean(axis=0)
        preds[i] = rerank(avg_ranks)
    return preds


def predict_rf(X_train, R_train, X_test, n_trees, seed):
    """Random Forest multi-target regression, then re-rank predicted values."""
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    )
    model.fit(X_train, R_train)
    raw_preds = model.predict(X_test)
    preds = np.empty_like(raw_preds)
    for i in range(raw_preds.shape[0]):
        preds[i] = rerank(raw_preds[i])
    return preds


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def predict_mean_rank_baseline(R_train, n_test):
    """Predict the average training ranking for every test dataset."""
    mean_rank = rerank(R_train.mean(axis=0))
    return np.tile(mean_rank, (n_test, 1))


def predict_majority_rank_baseline(R_train, n_test):
    """Predict the most frequent complete ranking in the training fold.

    Ties between equally frequent rankings are broken deterministically by
    pandas' sorted value_counts index.
    """
    counts = pd.DataFrame(R_train, columns=ALGO_COLS).value_counts(sort=True)
    majority_rank = np.asarray(counts.index[0], dtype=float)
    return np.tile(majority_rank, (n_test, 1))


# ---------------------------------------------------------------------------
# Per-test-instance metric computation
# ---------------------------------------------------------------------------

def safe_float(value, default=0.0):
    """Convert scipy/numpy scalar outputs to float, replacing NaN by default."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(value) else value


def src_score(true_rank, pred_rank):
    """Spearman rank correlation between true and predicted rankings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, _ = spearmanr(true_rank, pred_rank)
    return safe_float(stat)


def weighted_rank_correlation(true_rank, pred_rank):
    """Top-weighted rank correlation based on scipy.stats.weightedtau.

    The rank-importance array is defined from the true ranking: the best true
    algorithm has importance rank 0, the second-best rank 1, and so on. With
    scipy's default hyperbolic weigher, swaps involving top-ranked algorithms
    receive larger weights. This makes the metric more aligned with algorithm
    recommendation than plain SRC, which weights all positions uniformly.

    Range: approximately [-1, 1]. Higher is better.
    """
    true_rank = np.asarray(true_rank, dtype=float)
    pred_rank = np.asarray(pred_rank, dtype=float)
    importance = np.maximum(true_rank - 1, 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, _ = weightedtau(true_rank, pred_rank, rank=importance)
    return safe_float(stat)


def ndcg_at_k(true_rank, pred_rank, k=3):
    """NDCG@k for algorithm rankings.

    Relevance is derived from the true rank: rank 1 receives the largest
    relevance. The predicted ranking defines the retrieved order. Higher is
    better, with 1.0 representing an ideal top-k ordering.
    """
    true_rank = np.asarray(true_rank, dtype=float)
    pred_rank = np.asarray(pred_rank, dtype=float)
    n_algorithms = len(true_rank)
    if n_algorithms == 0:
        return np.nan

    relevance = n_algorithms - true_rank + 1
    pred_order = np.argsort(pred_rank, kind="mergesort")
    ideal_order = np.argsort(true_rank, kind="mergesort")
    k_eff = min(k, n_algorithms)

    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum(relevance[pred_order[:k_eff]] * discounts))
    idcg = float(np.sum(relevance[ideal_order[:k_eff]] * discounts))
    return dcg / idcg if idcg > 0 else np.nan


def top1_hit(true_rank, pred_rank):
    """Whether the predicted best algorithm is also a true best algorithm."""
    true_rank = np.asarray(true_rank, dtype=float)
    pred_rank = np.asarray(pred_rank, dtype=float)
    pred_best = int(np.argmin(pred_rank))
    true_best_rank = np.nanmin(true_rank)
    return float(true_rank[pred_best] == true_best_rank)


def regret_at_1(true_rank, pred_rank):
    """Normalized regret of the predicted top-1 algorithm.

    0.00 means the predicted top-1 is truly best. 1.00 means it is the worst
    possible algorithm among the candidates. Lower is better.
    """
    true_rank = np.asarray(true_rank, dtype=float)
    pred_rank = np.asarray(pred_rank, dtype=float)
    pred_best = int(np.argmin(pred_rank))
    denom = max(len(true_rank) - 1, 1)
    return float((true_rank[pred_best] - 1) / denom)


def compute_metrics(pred_rank, true_rank, dataset_stem, top1_algo, assignments_dir):
    """Ranking-level metrics plus ARI/AMI of the predicted top-1 partition."""
    metrics = {
        "SRC": src_score(true_rank, pred_rank),
        "WRC": weighted_rank_correlation(true_rank, pred_rank),
        "NDCG@3": ndcg_at_k(true_rank, pred_rank, k=3),
        "Top1Hit": top1_hit(true_rank, pred_rank),
        "Regret@1": regret_at_1(true_rank, pred_rank),
        "ARI": np.nan,
        "AMI": np.nan,
    }

    pred_path = assignments_dir / f"{dataset_stem}__{top1_algo}.npy"
    truth_path = assignments_dir / f"{dataset_stem}__y_true.npy"
    if pred_path.exists() and truth_path.exists():
        y_pred = np.load(pred_path)
        y_true = np.load(truth_path)
        if len(y_pred) == len(y_true):
            metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
            metrics["AMI"] = adjusted_mutual_info_score(y_true, y_pred)

    return metrics


def append_prediction_rows(rows, repetition, fold_i, mf_set, learner, preds,
                           test_idx, R_test, dataset_ids, dataset_names, stems,
                           assignments_dir):
    """Append one results row per test dataset for a set of predictions."""
    for ti, gi in enumerate(test_idx):
        pred = preds[ti]
        true = R_test[ti]
        top1 = ALGO_COLS[int(np.argmin(pred))]
        metrics = compute_metrics(pred, true, stems[gi], top1, assignments_dir)
        rows.append({
            "repetition": repetition,
            "fold": fold_i,
            "mf_set": mf_set,
            "learner": learner,
            "dataset_id": int(dataset_ids[gi]),
            "dataset_name": dataset_names[gi],
            "top1_predicted": top1,
            **metrics,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mf-dir",       default="./meta_features")
    parser.add_argument("--targets-dir",  default="./meta_targets")
    parser.add_argument("--datasets-dir", default="./datasets")
    parser.add_argument("--output-dir",   default="./meta_learning")
    parser.add_argument("--repetitions",  type=int, default=N_REPETITIONS)
    parser.add_argument("--folds",        type=int, default=N_FOLDS)
    parser.add_argument("--k-neighbors",  type=int, default=K_NEIGHBORS)
    parser.add_argument("--n-trees",      type=int, default=N_TREES)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    mf_dir       = Path(args.mf_dir)
    targets_dir  = Path(args.targets_dir)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = targets_dir / "assignments"

    rankings = pd.read_csv(targets_dir / "true_rankings.csv").sort_values("dataset_id").reset_index(drop=True)
    R = rankings[ALGO_COLS].values
    dataset_ids = rankings["dataset_id"].values
    dataset_names = rankings["name"].values

    manifest = pd.read_csv(Path(args.datasets_dir) / "manifest.csv")
    stem_by_id = {
        int(r["dataset_id"]): Path(r["filename"]).stem
        for _, r in manifest.iterrows()
    }
    stems = np.array([stem_by_id[int(did)] for did in dataset_ids])

    mf_sets = {}
    for name, fname in MF_FILES.items():
        df = pd.read_csv(mf_dir / fname).set_index("dataset_id").loc[dataset_ids]
        cols = [c for c in df.columns if c.startswith("MF")]
        mf_sets[name] = df[cols].values

    print(f"Datasets:          {len(rankings)}")
    print(f"MF sets:           {list(MF_FILES.keys())}")
    print(f"Meta-learners:     k-NN (k={args.k_neighbors}), RF (T={args.n_trees})")
    print("Baselines:         MeanRank, MajorityRank")
    print(f"Cross-validation:  {args.folds}-fold × {args.repetitions} repetitions")
    print(f"Metrics:           {', '.join(METRICS)}\n")

    rows = []
    t_total = time.time()

    for rep in range(args.repetitions):
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state + rep)
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(rankings)))):
            R_train, R_test = R[train_idx], R[test_idx]

            # Fold-safe baselines: use only training rankings.
            baseline_preds = {
                "MeanRank": predict_mean_rank_baseline(R_train, len(test_idx)),
                "MajorityRank": predict_majority_rank_baseline(R_train, len(test_idx)),
            }
            for baseline_name, preds in baseline_preds.items():
                append_prediction_rows(
                    rows, rep, fold_i, baseline_name, "Baseline", preds,
                    test_idx, R_test, dataset_ids, dataset_names, stems, assignments_dir
                )

            # Meta-feature-based methods.
            for mf_name, MF in mf_sets.items():
                MF_train, MF_test = MF[train_idx], MF[test_idx]

                scaler = StandardScaler().fit(MF_train)
                X_train = scaler.transform(MF_train)
                X_test  = scaler.transform(MF_test)

                preds_by_learner = {
                    "kNN": predict_knn_average_ranking(X_train, R_train, X_test, args.k_neighbors),
                    "RF":  predict_rf(X_train, R_train, X_test, args.n_trees, args.random_state + rep),
                }

                for learner, preds in preds_by_learner.items():
                    append_prediction_rows(
                        rows, rep, fold_i, mf_name, learner, preds,
                        test_idx, R_test, dataset_ids, dataset_names, stems, assignments_dir
                    )

        elapsed = time.time() - t_total
        print(f"  rep {rep+1}/{args.repetitions} done  ({elapsed:.0f}s elapsed)")

    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "results_full.csv", index=False)

    summary_rows = []
    for (mf_set, learner), sub in full.groupby(["mf_set", "learner"], sort=False):
        row = {"mf_set": mf_set, "learner": learner}
        for metric in METRICS:
            vals = sub[metric].dropna()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"]  = float(vals.std())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "results_summary.csv", index=False)

    metadata = {
        "phase": 4,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mf_dir": str(mf_dir),
        "targets_dir": str(targets_dir),
        "datasets_dir": args.datasets_dir,
        "output_dir": str(out_dir),
        "repetitions": args.repetitions,
        "folds": args.folds,
        "k_neighbors": args.k_neighbors,
        "n_trees": args.n_trees,
        "random_state": args.random_state,
        "mf_sets": list(MF_FILES.keys()),
        "baselines": ["MeanRank", "MajorityRank"],
        "metrics": METRICS,
    }
    with open(out_dir / "run_metadata_phase4.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    total = time.time() - t_total
    print(f"\nDone in {total:.1f}s.")
    print("\n--- Summary (mean ± std across all folds and repetitions) ---")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(summary.round(4).to_string(index=False))
    print(f"\nOutputs in {out_dir}/")


if __name__ == "__main__":
    main()
