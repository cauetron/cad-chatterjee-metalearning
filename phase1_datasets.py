"""
Phase 1: Download CC-18 datasets and apply standard preprocessing.

For each dataset in OpenML benchmark suite CC-18 (suite id=99):
  - keep only numerical attributes (drop categorical)
  - drop rows containing NaN or inf
  - min-max normalize each attribute to [0, 1]
  - keep only datasets with >= MIN_INSTANCES, >= MIN_FEATURES, >= MIN_CLASSES

Outputs:
  <output-dir>/<id>_<name>.csv   one per kept dataset (last column = "__target__")
  <output-dir>/manifest.csv      summary table: dataset_id, name, n, p, K, filename

Usage:
  python phase1_datasets.py                       # full CC-18
  python phase1_datasets.py --max-datasets 5      # quick test
"""

import argparse
from pathlib import Path

import numpy as np
import openml
import pandas as pd

MIN_INSTANCES = 100
MIN_FEATURES = 2
MIN_CLASSES = 2


def get_cc18_dataset_ids():
    """Return sorted unique dataset IDs in OpenML CC-18 benchmark suite (id 99)."""
    suite = openml.study.get_suite(99)
    tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
    return sorted(set(int(d) for d in tasks["did"]))


def preprocess(dataset_id):
    """Download and preprocess one dataset.

    Returns (DataFrame, info_dict) on success, or (None, reason_string) on skip.
    """
    try:
        ds = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, cat_mask, attr_names = ds.get_data(target=ds.default_target_attribute)
    except Exception as e:
        return None, f"download/parse failed: {type(e).__name__}: {e}"

    if y is None:
        return None, "no default target"

    # Keep only numerical attributes.
    num_cols = [n for n, is_cat in zip(attr_names, cat_mask) if not is_cat and n in X.columns]
    if len(num_cols) < MIN_FEATURES:
        return None, f"only {len(num_cols)} numerical features"

    # Drop rows with NaN/inf in features or target.
    combined = pd.concat([X[num_cols], y.rename("__target__")], axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(combined) < MIN_INSTANCES:
        return None, f"{len(combined)} rows after cleaning"

    # Encode target as 0..K-1.
    y_enc = pd.factorize(combined["__target__"])[0]
    K = int(np.unique(y_enc).size)
    if K < MIN_CLASSES:
        return None, f"only {K} class(es)"

    # Min-max normalize per column; constant columns become 0.
    X_arr = combined.drop(columns="__target__").values.astype(float)
    col_min = X_arr.min(axis=0)
    col_max = X_arr.max(axis=0)
    constant = col_max == col_min
    col_range = np.where(constant, 1.0, col_max - col_min)
    X_norm = (X_arr - col_min) / col_range
    X_norm[:, constant] = 0.0

    out_df = pd.DataFrame(X_norm, columns=num_cols)
    out_df["__target__"] = y_enc

    info = {
        "dataset_id": dataset_id,
        "name": ds.name,
        "n": len(out_df),
        "p": len(num_cols),
        "K": K,
    }
    return out_df, info


def safe_filename(name, dataset_id):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:50]
    return f"{dataset_id}_{safe}.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./datasets")
    parser.add_argument(
        "--max-datasets", type=int, default=None,
        help="Optional cap on number of datasets (for quick test)."
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching CC-18 (OpenML benchmark suite 99)...")
    ids = get_cc18_dataset_ids()
    print(f"  Suite contains {len(ids)} datasets.")

    if args.max_datasets is not None:
        ids = ids[: args.max_datasets]
        print(f"  Capped at {args.max_datasets} for this run.")

    manifest = []
    n_skipped = 0
    for i, did in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] id={did}", end=" ... ", flush=True)
        df, info = preprocess(did)
        if df is None:
            print(f"SKIP ({info})")
            n_skipped += 1
            continue
        fname = safe_filename(info["name"], info["dataset_id"])
        df.to_csv(out_dir / fname, index=False)
        info["filename"] = fname
        manifest.append(info)
        print(f"OK n={info['n']} p={info['p']} K={info['K']}")

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print(f"\nDone. {len(manifest)} datasets saved, {n_skipped} skipped.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
