"""
Match a list of candidate SMILES against the TOXRIC merged dataset.

Uses RDKit canonical SMILES for exact matching (avoids kekulization differences).
If no candidates match, falls back to returning all candidates unfiltered.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem


def _canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def filter_by_toxric(
    candidates: list[dict],
    merged_csv_path: str = "data/toxric_merged.csv",
) -> tuple[list[dict], bool]:
    """Filter candidates to those present in the TOXRIC merged CSV.

    Args:
        candidates:       List of {smiles, similarity, chembl_id} dicts (from similarity.py).
        merged_csv_path:  Path to toxric_merged.csv produced by scripts/merge_toxric.py.

    Returns:
        (filtered, is_fallback)
        - filtered:     Matched candidates with an extra 'toxric_labels' key containing
                        the endpoint binary labels from the CSV row, OR all candidates
                        if no TOXRIC match was found (fallback).
        - is_fallback:  True when no TOXRIC match was found and all candidates are returned.
    """
    df = pd.read_csv(merged_csv_path, low_memory=False)

    tox_index: dict[str, int] = {}
    for i, raw in enumerate(df["smiles"]):
        canon = _canon(str(raw))
        if canon:
            tox_index[canon] = i

    matched: list[dict] = []
    for cand in candidates:
        canon = _canon(cand["smiles"])
        if canon and canon in tox_index:
            row = df.iloc[tox_index[canon]]
            matched.append({**cand, "toxric_labels": row.drop("smiles").to_dict()})

    if matched:
        return matched, False

    # Fallback: no TOXRIC overlap — return all candidates without labels
    return candidates, True
