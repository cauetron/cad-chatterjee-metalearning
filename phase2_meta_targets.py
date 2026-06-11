"""
Phase 2: Generate true rankings of clustering algorithms per dataset.

For each dataset produced by phase1_datasets.py:
  - subsample to MAX_N instances if larger (fixed seed)
  - run 5 clustering algorithms with K = true number of classes
  - score each partition with 10 internal indices (Pimentel 2019, Table 6):
      CH, Sil, Dunn, Gamma, Tau    (higher is better)
      DB, XB, SD-Scat, SD-Dis, RT  (lower is better)

Correção aplicada: XB usa a separação mínima entre pontos de clusters
diferentes (δ1), enquanto RT usa a separação mínima entre centróides.
Assim, os dois índices não ficam algebricamente duplicados no caso crisp.
  - rank algorithms per index, average ranks, reassign final ranks 1..A

Outputs (./meta_targets/):
  - true_rankings.csv     one row per dataset, columns = 5 algorithms
  - raw_indices.csv       raw index values per (dataset, algorithm, index)
  - timings.csv           per-algorithm runtime per dataset
  - assignments/          cluster labels (.npy) for phase 4 ARI/AMI

Usage:
  python phase2_meta_targets.py --max-datasets 5 --sort-by-size   # quick test
  python phase2_meta_targets.py                                   # full run
"""

import argparse
import json
import time
from datetime import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

MAX_N = 5000
RANDOM_SEED = 42

# Pimentel (2019) Table 6: 10 internal indices.
INDEX_HIGHER_IS_BETTER = {
    "CH":      True,   # Calinski-Harabasz
    "Sil":     True,   # Silhouette
    "Dunn":    True,   # Dunn (min between / max within)
    "Gamma":   True,   # Baker-Hubert Gamma
    "Tau":     True,   # Tau simplificado: (s+ - s-) / (n_within * n_between)
    "DB":      False,  # Davies-Bouldin
    "XB":      False,  # Xie-Beni (crisp variant)
    "SD_Scat": False,  # Halkidi SD scattering component
    "SD_Dis":  False,  # Halkidi SD dissimilarity component
    "RT":      False,  # Ray-Turi
}
INDEX_NAMES = list(INDEX_HIGHER_IS_BETTER.keys())


def make_algorithms(seed):
    return {
        "KM":  lambda k: KMeans(n_clusters=k, n_init=10, random_state=seed),
        "WA":  lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "AA":  lambda k: AgglomerativeClustering(n_clusters=k, linkage="average"),
        "GMd": lambda k: GaussianMixture(n_components=k, covariance_type="diag",
                                         n_init=3, reg_covar=1e-4, random_state=seed),
        "GMf": lambda k: GaussianMixture(n_components=k, covariance_type="full",
                                         n_init=3, reg_covar=1e-4, random_state=seed),
    }


def compute_indices(X, labels, distances, i_idx, j_idx):
    """Compute all 10 indices for a single partition.

    distances: pdist(X) 1D array (shared across algorithm calls for same X)
    i_idx, j_idx: np.triu_indices(n, k=1) (shared)
    """
    out = {k: np.nan for k in INDEX_NAMES}
    unique = np.unique(labels)
    if len(unique) < 2:
        return out

    # --- sklearn indices ---
    try: out["CH"] = float(calinski_harabasz_score(X, labels))
    except Exception: pass
    try: out["Sil"] = float(silhouette_score(X, labels))
    except Exception: pass
    try: out["DB"] = float(davies_bouldin_score(X, labels))
    except Exception: pass

    # --- Pairwise-distance-based: Dunn, Gamma, Tau ---
    try:
        is_within = labels[i_idx] == labels[j_idx]
        within = distances[is_within]
        between = distances[~is_within]
        if len(within) > 0 and len(between) > 0:
            # Dunn: min between-pair distance / max within-pair distance.
            if within.max() > 0:
                out["Dunn"] = float(between.min() / within.max())
            # Gamma & Tau: concordance/discordance via sorted between-distances.
            between_sorted = np.sort(between)
            n_b = len(between)
            # For each w in `within`:
            #   # of b > w = n_b - searchsorted(side='right')   --> s+ (concordant)
            #   # of b < w = searchsorted(side='left')          --> s- (discordant)
            s_plus = int((n_b - np.searchsorted(between_sorted, within, side="right")).sum())
            s_minus = int(np.searchsorted(between_sorted, within, side="left").sum())
            if s_plus + s_minus > 0:
                out["Gamma"] = (s_plus - s_minus) / (s_plus + s_minus)
            # Tau: bounded [-1, +1] by dividing by total cross-pair count.
            out["Tau"] = (s_plus - s_minus) / (len(within) * n_b)
    except Exception: pass

    # --- Compactness and separation indices: XB, RT, SD-Scat, SD-Dis ---
    try:
        centroids = np.array([X[labels == k].mean(axis=0) for k in unique])
        n = X.shape[0]

        # Sum of squared distances of each point to its own cluster centroid.
        intra_sq = 0.0
        for ci, k in enumerate(unique):
            pts = X[labels == k]
            intra_sq += float(np.sum((pts - centroids[ci]) ** 2))

        # Xie-Beni (crisp variant used by clusterCrit):
        # numerator = mean squared error to assigned centroid;
        # denominator = minimum squared distance between points in different clusters.
        # This avoids duplicating Ray-Turi, whose denominator is centroid separation.
        between_mask = labels[i_idx] != labels[j_idx]
        between_dist = distances[between_mask]
        if between_dist.size > 0:
            min_between = float(np.min(between_dist))
            out["XB"] = np.inf if min_between <= 1e-12 else (intra_sq / n) / (min_between ** 2)

        # Pairwise centroid distances.
        cent_pd = pdist(centroids)
        if cent_pd.size > 0:
            min_cent = float(cent_pd.min())
            if min_cent <= 1e-12:
                out["RT"] = np.inf
            else:
                min_inter_sq = min_cent ** 2
                out["RT"] = (intra_sq / n) / min_inter_sq

                # SD-Dis: (Dmax/Dmin) * sum_k 1/(sum_{z!=k} d(v_k, v_z))
                dmax, dmin = float(cent_pd.max()), min_cent
                cent_sq = squareform(cent_pd)
                inv_sums = [1.0 / s for s in cent_sq.sum(axis=1) if s > 0]
                if inv_sums:
                    out["SD_Dis"] = (dmax / dmin) * sum(inv_sums)

        # SD-Scat: average per-cluster variance norm / overall variance norm.
        sigma_X_norm = float(np.linalg.norm(np.var(X, axis=0)))
        if sigma_X_norm > 0:
            norms = [
                float(np.linalg.norm(np.var(X[labels == k], axis=0)))
                for k in unique if (labels == k).sum() > 1
            ]
            if norms:
                out["SD_Scat"] = float(np.mean(norms)) / sigma_X_norm
    except Exception: pass

    return out


def rank_algorithms(scores_df):
    """Rank per index (1=best, NaN=last), average ranks, reassign final 1..A."""
    ranks = pd.DataFrame(index=scores_df.index, columns=scores_df.columns, dtype=float)
    for col in scores_df.columns:
        ascending = not INDEX_HIGHER_IS_BETTER[col]
        ranks[col] = scores_df[col].rank(method="average", ascending=ascending, na_option="bottom")
    avg = ranks.mean(axis=1)
    return avg.rank(method="average")


def process_dataset(filepath, K, algorithms, assignments_dir, max_n, seed):
    df = pd.read_csv(filepath)
    X = df.drop(columns="__target__").values.astype(float)
    y_true = df["__target__"].values

    n = X.shape[0]
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        X = X[idx]
        y_true = y_true[idx]
        print(f"    subsampled {n} -> {max_n}")

    # Compute pdist ONCE per dataset, share across all algorithms.
    t0 = time.time()
    distances = pdist(X)
    i_idx, j_idx = np.triu_indices(X.shape[0], k=1)
    pdist_time = time.time() - t0
    if pdist_time > 1.0:
        print(f"    pdist: {pdist_time:.1f}s  (n={X.shape[0]}, p={X.shape[1]})")

    scores, timings, assignments = {}, {}, {}
    for name, factory in algorithms.items():
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = factory(K).fit_predict(X)
            idx_scores = compute_indices(X, labels, distances, i_idx, j_idx)
            status = "OK"
        except Exception as e:
            labels = None
            idx_scores = {k: np.nan for k in INDEX_NAMES}
            status = f"FAIL({type(e).__name__})"
        dt = time.time() - t0
        scores[name] = idx_scores
        timings[name] = dt
        assignments[name] = labels
        n_valid = sum(1 for v in idx_scores.values() if not np.isnan(v))
        print(f"    {name:>3} {dt:6.1f}s  {status:<14}  {n_valid}/{len(INDEX_NAMES)} indices")

    stem = filepath.stem
    for name, labels in assignments.items():
        if labels is not None:
            np.save(assignments_dir / f"{stem}__{name}.npy", labels)
    np.save(assignments_dir / f"{stem}__y_true.npy", y_true)

    scores_df = pd.DataFrame(scores).T[INDEX_NAMES]
    final_rank = rank_algorithms(scores_df)
    return scores_df, final_rank, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="./datasets")
    parser.add_argument("--output-dir", default="./meta_targets")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-n", type=int, default=MAX_N,
                        help=f"Subsample to this many instances if larger (default {MAX_N})")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--sort-by-size", action="store_true")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = out_dir / "assignments"
    assignments_dir.mkdir(exist_ok=True)

    manifest = pd.read_csv(datasets_dir / "manifest.csv")
    if args.sort_by_size:
        manifest = manifest.sort_values("n").reset_index(drop=True)
    if args.max_datasets is not None:
        manifest = manifest.head(args.max_datasets).reset_index(drop=True)

    algorithms = make_algorithms(args.random_state)
    print(f"Algorithms: {list(algorithms.keys())}")
    print(f"Indices:    {INDEX_NAMES}")
    print(f"Datasets:   {len(manifest)}")
    print(f"Max N:      {args.max_n}\n")

    rankings_rows, scores_rows, timings_rows = [], [], []
    t_total = time.time()

    for i, row in manifest.iterrows():
        K = int(row["K"])
        print(f"[{i+1}/{len(manifest)}] {row['name']}  n={row['n']} p={row['p']} K={K}")
        filepath = datasets_dir / row["filename"]
        try:
            scores_df, final_rank, timings = process_dataset(
                filepath, K, algorithms, assignments_dir, args.max_n, args.random_state
            )
        except Exception as e:
            print(f"  DATASET FAILED: {type(e).__name__}: {e}")
            continue

        rankings_rows.append({"dataset_id": row["dataset_id"], "name": row["name"],
                              **final_rank.to_dict()})
        flat = {"dataset_id": row["dataset_id"]}
        for algo in scores_df.index:
            for idx in scores_df.columns:
                flat[f"{algo}_{idx}"] = scores_df.loc[algo, idx]
        scores_rows.append(flat)
        timings_rows.append({"dataset_id": row["dataset_id"], "n": int(row["n"]), **timings})

    pd.DataFrame(rankings_rows).to_csv(out_dir / "true_rankings.csv", index=False)
    pd.DataFrame(scores_rows).to_csv(out_dir / "raw_indices.csv", index=False)
    timings_df = pd.DataFrame(timings_rows)
    timings_df.to_csv(out_dir / "timings.csv", index=False)

    metadata = {
        "phase": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_dir": str(datasets_dir),
        "output_dir": str(out_dir),
        "max_n": args.max_n,
        "random_state": args.random_state,
        "algorithms": list(algorithms.keys()),
        "indices": INDEX_NAMES,
        "xb_formula": "mean squared error divided by minimum squared point-to-point distance between different clusters",
        "rt_formula": "mean squared error divided by minimum squared centroid-to-centroid distance",
    }
    with open(out_dir / "run_metadata_phase2.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    total = time.time() - t_total
    n_ok = len(rankings_rows)
    print(f"\nDone. {n_ok}/{len(manifest)} datasets processed in {total:.1f}s "
          f"({total/max(n_ok,1):.1f}s avg/dataset).")
    if n_ok > 0:
        print("Per-algorithm cumulative time:")
        for algo in ["KM", "WA", "AA", "GMd", "GMf"]:
            if algo in timings_df.columns:
                print(f"  {algo:>3}: total {timings_df[algo].sum():7.1f}s, "
                      f"mean {timings_df[algo].mean():5.2f}s, max {timings_df[algo].max():5.2f}s")
    print(f"Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
