"""
Merge the 30 single-endpoint TOXRIC CSVs into one wide multi-task CSV.

Each source CSV has columns:
    TAID, Name, IUPAC Name, PubChem CID, Canonical SMILES, InChIKey, Toxicity Value

We keep Canonical SMILES as the join key (renamed to 'smiles' for KPGT)
and turn each file's 'Toxicity Value' into its own column named after the
file (minus the .csv suffix). Compounds absent from a given endpoint get
NaN there — the trainer's masked loss handles that.

Usage from labfiles/toxicity-kpgt/:
    python scripts/merge_toxric.py \\
        --source-dir data/toxric/toxric_30_datasets/toxric_30_datasets \\
        --output data/toxric_merged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SMILES_COL = "Canonical SMILES"
LABEL_COL = "Toxicity Value"


def endpoint_name_from_filename(path: Path) -> str:
    """'Hepatotoxicity_Hepatotoxicity.csv' -> 'Hepatotoxicity_Hepatotoxicity'."""
    return path.stem.replace(" ", "_")


def merge_toxric_csvs(source_dir: str | Path, key: str = SMILES_COL) -> pd.DataFrame:
    """Load all 30 TOXRIC endpoint CSVs from source_dir and outer-join on `key`.

    Returns a wide DataFrame with `smiles` (or `key`) plus one column per endpoint.
    Compounds with duplicate keys within a single endpoint file are max-aggregated
    (positive wins, since labels are binary 0/1).

    Walks `source_dir` recursively, so it works whether you pass the parent
    `data/toxric/toxric_30_datasets/` (the extract dir from download_toxric)
    or the nested `data/toxric/toxric_30_datasets/toxric_30_datasets/`.
    Skips macOS metadata (`__MACOSX/`, `._*`).
    """
    src = Path(source_dir)
    csvs = sorted(
        p for p in src.rglob("*.csv")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )
    if not csvs:
        raise FileNotFoundError(f"No CSVs under {src}")

    merged: pd.DataFrame | None = None
    for csv in csvs:
        df = pd.read_csv(csv)
        if SMILES_COL not in df.columns or LABEL_COL not in df.columns:
            continue
        endpoint = endpoint_name_from_filename(csv)
        slim = (df[[key, LABEL_COL]]
                .dropna(subset=[key])
                .groupby(key, as_index=False)[LABEL_COL]
                .max()
                .rename(columns={LABEL_COL: endpoint}))
        merged = slim if merged is None else merged.merge(slim, on=key, how="outer")

    if merged is None:
        raise RuntimeError("No endpoint CSVs matched the expected schema.")

    # KPGT expects the smiles column to be named 'smiles' (lowercase)
    merged = merged.rename(columns={key: "smiles"})
    cols = ["smiles"] + [c for c in merged.columns if c != "smiles"]
    return merged[cols]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True,
                   help="Directory containing the 30 TOXRIC CSVs.")
    p.add_argument("--output", required=True, help="Output merged CSV path.")
    p.add_argument("--key", default="Canonical SMILES",
                   choices=["Canonical SMILES", "InChIKey"],
                   help="Join key. InChIKey is more canonical but SMILES is what KPGT expects.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    merged = merge_toxric_csvs(args.source_dir, key=args.key)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    # Quick label-density report so the user knows what they got
    n_compounds = len(merged)
    label_cols = [c for c in merged.columns if c != "smiles"]
    cov = merged[label_cols].notna().sum() / n_compounds
    print(f"\n[merge] wrote {out}")
    print(f"        compounds: {n_compounds}, endpoints: {len(label_cols)}")
    print(f"        avg label density per endpoint: {cov.mean()*100:.1f}%")
    print(f"        most-covered endpoints:")
    for c, v in cov.sort_values(ascending=False).head(5).items():
        print(f"          {c:<55} {v*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
