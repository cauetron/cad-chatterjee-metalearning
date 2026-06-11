"""
Phase 3: Extract 4 sets of meta-features per dataset.

For each dataset in manifest.csv (output of phase1):
  - subsample to MAX_N=5000 instances (same seed as phase 2)
  - sample MAX_PAIRS=50000 pairs of instances (same seed for all 4 sets)
  - compute pairwise Euclidean distance d
  - compute pairwise correlations:
      c_sp = Spearman      (Pimentel 2019 baseline, vectorized via pre-ranking)
      c_ke = Kendall tau   (proposal 1, scipy per-pair)
      c_ch = Chatterjee xi simétrico (proposal 2; média de xi(x_k,x_l) e xi(x_l,x_k))
  - build 4 vectors and normalize each to [0, 1]:
      Distance: m = d
      CaD-Sp:   m = [c_sp, d]
      CaD-Ke:   m = [c_ke, d]
      CaD-Ch:   m = [c_ch, d]
  - extract 19 meta-features per Pimentel 2019, Table 2:
      MF1-5:   mean, variance, std, skewness, kurtosis
      MF6-15:  histogram bins in [0, 1] with bin width 0.1
      MF16-19: |Z-score| histogram in [0,1), [1,2), [2,3), [3,inf)

Outputs (./meta_features/):
  - mf_distance.csv          one row per dataset, columns = dataset_id + 19 MFs
  - mf_cad_spearman.csv
  - mf_cad_kendall.csv
  - mf_cad_chatterjee.csv
  - timings.csv              per-step runtime per dataset

Usage:
  python phase3_meta_features.py --max-datasets 5 --sort-by-size
  python phase3_meta_features.py
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau, skew, kurtosis

MAX_N = 5000
MAX_PAIRS = 50000
RANDOM_SEED = 42

CONJUNTOS = ["Distance", "CaD-Sp", "CaD-Ke", "CaD-Ch"]
OUTPUT_FILES = {
    "Distance": "mf_distance.csv",
    "CaD-Sp":   "mf_cad_spearman.csv",
    "CaD-Ke":   "mf_cad_kendall.csv",
    "CaD-Ch":   "mf_cad_chatterjee.csv",
}


# ---------------------------------------------------------------------------
# Meta-feature extraction (Algorithm 1 of Pimentel 2019)
# ---------------------------------------------------------------------------

def normalize_minmax(v):
    """Min-max normalize a 1D array to [0, 1]. Constant input -> all zeros."""
    v_min, v_max = float(v.min()), float(v.max())
    if v_max == v_min:
        return np.zeros_like(v)
    return (v - v_min) / (v_max - v_min)


def extract_19_mfs(m):
    """Compute MF1..MF19 from a 1D array m already normalized to [0, 1]."""
    out = {}
    out["MF1_mean"] = float(np.mean(m))
    out["MF2_var"]  = float(np.var(m, ddof=0))
    out["MF3_std"]  = float(np.std(m, ddof=0))
    skew_val = float(skew(m, bias=False))
    kurt_val = float(kurtosis(m, bias=False))
    out["MF4_skew"] = 0.0 if np.isnan(skew_val) else skew_val
    out["MF5_kurt"] = 0.0 if np.isnan(kurt_val) else kurt_val

    bin_edges = np.linspace(0, 1, 11)
    bin_edges[-1] += 1e-9  # include 1.0 in last bin
    hist, _ = np.histogram(m, bins=bin_edges)
    for i in range(10):
        out[f"MF{6+i}_h{i}"] = float(hist[i] / len(m))

    mean = float(np.mean(m))
    std  = float(np.std(m, ddof=0))
    z = np.zeros_like(m) if std < 1e-12 else np.abs((m - mean) / std)
    hist_z, _ = np.histogram(z, bins=[0, 1, 2, 3, np.inf])
    for i in range(4):
        out[f"MF{16+i}_z{i}"] = float(hist_z[i] / len(m))
    return out


# ---------------------------------------------------------------------------
# Pair sampling and pairwise computations
# ---------------------------------------------------------------------------

def sample_pairs(n, n_pairs, rng):
    """Sample n_pairs distinct (i, j) with i < j from {0..n-1}."""
    i_all, j_all = np.triu_indices(n, k=1)
    total = len(i_all)
    if n_pairs >= total:
        return np.column_stack([i_all, j_all])
    sel = rng.choice(total, size=n_pairs, replace=False)
    return np.column_stack([i_all[sel], j_all[sel]])


def pairwise_distance(X, pair_idx):
    diff = X[pair_idx[:, 0]] - X[pair_idx[:, 1]]
    return np.sqrt(np.sum(diff ** 2, axis=1))


def pairwise_spearman(X, pair_idx):
    """Spearman between rows of X, vectorized via pre-ranking + standardization."""
    R = rankdata(X, axis=1).astype(np.float64)
    R = R - R.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    R = R / norms
    return np.sum(R[pair_idx[:, 0]] * R[pair_idx[:, 1]], axis=1)


def pairwise_kendall(X, pair_idx):
    """Kendall tau between rows of X (scipy, per-pair)."""
    out = np.empty(len(pair_idx))
    for k in range(len(pair_idx)):
        i, j = pair_idx[k]
        tau, _ = kendalltau(X[i], X[j])
        out[k] = 0.0 if np.isnan(tau) else tau
    return out


def chatterjee_xi(x, y):
    """Chatterjee xi(X, Y); asymmetric dependence of Y on X.

    Uses the tie-aware version from Chatterjee's coefficient. For vectors with
    no ties in y, this reduces to 1 - 3 * sum(|diff(rank_y)|) / (n^2 - 1).
    Ties in x are resolved deterministically by a stable sort for reproducibility.
    """
    n = len(x)
    if n < 2:
        return 0.0

    order = np.argsort(x, kind="stable")
    y_sorted = y[order]

    # r_i = #{j: y_j <= y_i}; l_i = #{j: y_j >= y_i}
    r = rankdata(y_sorted, method="max").astype(float)
    l = (n - rankdata(y_sorted, method="min") + 1).astype(float)

    numerator = n * np.sum(np.abs(np.diff(r)))
    denominator = 2.0 * np.sum(l * (n - l))
    if denominator <= 1e-12:
        return 0.0
    return 1.0 - numerator / denominator


def pairwise_chatterjee(X, pair_idx, sym_mode="mean"):
    """Symmetric Chatterjee between rows of X.

    Chatterjee's coefficient is directed: xi(x, y) can differ from xi(y, x).
    Instance pairs in clustering have no natural direction, so the default is
    the mean of both directions. Use sym_mode='max' to keep the stronger
    directed dependence instead.
    """
    out = np.empty(len(pair_idx))
    for k in range(len(pair_idx)):
        i, j = pair_idx[k]
        xy = chatterjee_xi(X[i], X[j])
        yx = chatterjee_xi(X[j], X[i])
        if sym_mode == "max":
            out[k] = max(xy, yx)
        elif sym_mode == "mean":
            out[k] = 0.5 * (xy + yx)
        else:
            raise ValueError("sym_mode must be 'mean' or 'max'")
    return out


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def process_dataset(filepath, max_n, max_pairs, seed, chatterjee_sym, verbose=True):
    df = pd.read_csv(filepath)
    X = df.drop(columns="__target__").values.astype(float)

    if X.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X = X[idx]
    n = X.shape[0]

    rng_pairs = np.random.default_rng(seed + 1)
    pair_idx = sample_pairs(n, max_pairs, rng_pairs)

    t0 = time.time();  d   = pairwise_distance(X, pair_idx);    t_d  = time.time() - t0
    t0 = time.time();  csp = pairwise_spearman(X, pair_idx);    t_sp = time.time() - t0
    t0 = time.time();  cke = pairwise_kendall(X, pair_idx);     t_ke = time.time() - t0
    t0 = time.time();  cch = pairwise_chatterjee(X, pair_idx, sym_mode=chatterjee_sym);  t_ch = time.time() - t0

    if verbose:
        print(f"    pairs={len(pair_idx)}  d={t_d:5.1f}s  sp={t_sp:5.1f}s  "
              f"ke={t_ke:5.1f}s  ch={t_ch:5.1f}s")

    vectors = {
        "Distance": d,
        "CaD-Sp":   np.concatenate([csp, d]),
        "CaD-Ke":   np.concatenate([cke, d]),
        "CaD-Ch":   np.concatenate([cch, d]),
    }

    mfs = {name: extract_19_mfs(normalize_minmax(m)) for name, m in vectors.items()}
    timings = {"d": t_d, "sp": t_sp, "ke": t_ke, "ch": t_ch}
    return mfs, timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="./datasets")
    parser.add_argument("--output-dir",   default="./meta_features")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-n",        type=int, default=MAX_N)
    parser.add_argument("--max-pairs",    type=int, default=MAX_PAIRS)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--chatterjee-sym", choices=["mean", "max"], default="mean",
                        help="How to symmetrize Chatterjee xi for unordered instance pairs (default: mean)")
    parser.add_argument("--sort-by-size", action="store_true")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(datasets_dir / "manifest.csv")
    if args.sort_by_size:
        manifest = manifest.sort_values("n").reset_index(drop=True)
    if args.max_datasets is not None:
        manifest = manifest.head(args.max_datasets).reset_index(drop=True)

    print(f"Conjuntos:  {CONJUNTOS}")
    print(f"Datasets:   {len(manifest)}")
    print(f"Max N:      {args.max_n}, Max pairs: {args.max_pairs}")
    print(f"Chatterjee: symmetric {args.chatterjee_sym} of both directions\n")

    rows = {c: [] for c in CONJUNTOS}
    all_timings = []
    t_total = time.time()

    for i, row in manifest.iterrows():
        print(f"[{i+1}/{len(manifest)}] {row['name']}  n={row['n']} p={row['p']}")
        try:
            mfs, timings = process_dataset(
                datasets_dir / row["filename"],
                args.max_n, args.max_pairs, args.random_state, args.chatterjee_sym
            )
        except Exception as e:
            print(f"  DATASET FAILED: {type(e).__name__}: {e}")
            continue

        for c in CONJUNTOS:
            rows[c].append({"dataset_id": row["dataset_id"], "name": row["name"],
                            **mfs[c]})
        all_timings.append({
            "dataset_id": row["dataset_id"],
            "n": int(row["n"]), "p": int(row["p"]),
            **timings,
        })

    for c in CONJUNTOS:
        df = pd.DataFrame(rows[c])
        df.to_csv(out_dir / OUTPUT_FILES[c], index=False)
        print(f"  Wrote {OUTPUT_FILES[c]}  ({len(df)} rows × {len(df.columns)} cols)")

    timings_df = pd.DataFrame(all_timings)
    timings_df.to_csv(out_dir / "timings.csv", index=False)

    metadata = {
        "phase": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_dir": str(datasets_dir),
        "output_dir": str(out_dir),
        "max_n": args.max_n,
        "max_pairs": args.max_pairs,
        "random_state": args.random_state,
        "chatterjee_sym": args.chatterjee_sym,
        "mf_sets": CONJUNTOS,
    }
    with open(out_dir / "run_metadata_phase3.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    total = time.time() - t_total
    print(f"\nDone in {total:.1f}s.")
    print("Cumulative time per step:")
    for col in ["d", "sp", "ke", "ch"]:
        if col in timings_df.columns:
            print(f"  {col:>3}: total {timings_df[col].sum():7.1f}s,  "
                  f"mean {timings_df[col].mean():5.2f}s,  max {timings_df[col].max():5.2f}s")


if __name__ == "__main__":
    main()
