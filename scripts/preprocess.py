"""
Convert a TOXRIC CSV (or merged multi-task CSV) into the KPGT cache format.

What this does:
    1. Copies/renames the source CSV to {data_root}/{dataset}/{dataset}.csv
       with the SMILES column normalized to 'smiles' (lowercase, required by KPGT).
    2. Generates a scaffold-like random split file at
       {data_root}/{dataset}/splits/{split_name}.npy
       (object array of 3: [train_idx, val_idx, test_idx]).
    3. Invokes KPGT's preprocess_downstream_dataset.py via subprocess to
       produce the .pkl graphs, fingerprints, and descriptors.

Usage from labfiles/toxicity-kpgt/:
    python scripts/preprocess.py \\
        --source data/toxric/toxric_30_datasets/<some_csv>.csv \\
        --dataset toxric_hepatotoxicity \\
        --smiles-col "Canonical SMILES" \\
        --label-cols col1 col2 col3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Path to the TOXRIC CSV (or merged multi-task CSV).")
    p.add_argument("--dataset", required=True, help="Name for the prepared dataset (also the output dirname).")
    p.add_argument("--data-root", default="data/kpgt-cache", help="Where KPGT-format datasets live.")
    p.add_argument("--smiles-col", default="Canonical SMILES", help="SMILES column name in the source CSV.")
    p.add_argument("--label-cols", nargs="+", required=True, help="Target columns to keep as labels.")
    p.add_argument("--split-name", default="random_0")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kpgt-dir", default="external/KPGT", help="Path to the cloned KPGT repo.")
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def normalize_csv(args) -> Path:
    """Read the source, keep smiles + label columns, write to KPGT-expected path."""
    df = pd.read_csv(args.source)

    if args.smiles_col not in df.columns:
        raise SystemExit(
            f"--smiles-col {args.smiles_col!r} not found. Available columns: {list(df.columns)}"
        )
    missing = [c for c in args.label_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Label columns not found: {missing}")

    df = df[[args.smiles_col, *args.label_cols]].rename(columns={args.smiles_col: "smiles"})
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    out_dir = Path(args.data_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.dataset}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[csv] wrote {len(df)} rows × {len(df.columns)} cols to {out_csv}")
    return out_csv


def write_split(args, n_rows: int) -> Path:
    """Random split → object array of 3 index arrays, saved at splits/{name}.npy."""
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_rows)
    n_test = int(n_rows * args.test_frac)
    n_val = int(n_rows * args.val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    splits_dir = Path(args.data_root) / args.dataset / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{args.split_name}.npy"
    np.save(split_path, np.array([train_idx, val_idx, test_idx], dtype=object),
            allow_pickle=True)
    print(f"[split] train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}  -> {split_path}")
    return split_path


def run_kpgt_preprocess(args) -> None:
    script = Path(args.kpgt_dir) / "scripts" / "preprocess_downstream_dataset.py"
    if not script.exists():
        raise SystemExit(f"Cannot find {script}. Did you run scripts/setup.py first?")

    kpgt_root = str(Path(args.kpgt_dir).resolve())
    cmd = [
        sys.executable, str(script),
        "--data_path", str(Path(args.data_root).resolve()),
        "--dataset", args.dataset,
        "--n_jobs", str(args.n_jobs),
    ]
    print(f"[kpgt-preprocess] {' '.join(cmd)}")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = kpgt_root
    subprocess.run(cmd, check=True, cwd=kpgt_root, env=env)


def _preprocessed_count(args) -> int:
    """Read the actual number of valid compounds from the fps file produced by KPGT."""
    import scipy.sparse as sp
    fps_path = Path(args.data_root) / args.dataset / "rdkfp1-7_512.npz"
    return sp.load_npz(fps_path).shape[0]


def main() -> int:
    args = parse_args()
    normalize_csv(args)
    run_kpgt_preprocess(args)
    # Use the preprocessed fps count — KPGT filters out invalid SMILES during
    # graph building, so the actual row count may be less than the CSV row count.
    n_valid = _preprocessed_count(args)
    write_split(args, n_rows=n_valid)
    print("[done] dataset ready. Next: python scripts/train.py --dataset", args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
