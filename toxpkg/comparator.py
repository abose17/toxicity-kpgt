"""
Rank molecules by predicted toxicity.

Toxicity score = sum of sigmoid probabilities across all classification endpoints.
Lower score = safer molecule.
"""

from __future__ import annotations

import pandas as pd

_META_COLS = {"smiles", "name", "similarity", "chembl_id", "toxric_match", "toxicity_score", "role"}


def rank_by_toxicity(
    records: list[dict],
    scores_list: list[dict],
    is_fallback: bool = False,
    top_k: int = 5,
) -> pd.DataFrame:
    """Build a ranked comparison DataFrame.

    Args:
        records:     List of candidate dicts {smiles, similarity, chembl_id, ...}.
                     The first record must be the original query molecule (role='original').
        scores_list: Parallel list of {endpoint_name: sigmoid_prob} dicts from predict.py.
        is_fallback: If True (no TOXRIC matches), cap output at top_k alternatives.
        top_k:       Number of alternatives to keep in fallback mode.

    Returns:
        DataFrame with columns: role, smiles, similarity, chembl_id, toxric_match,
        toxicity_score, <endpoint columns...>, sorted ascending by toxicity_score.
        The original query row retains its position for easy comparison.
    """
    rows = []
    for i, (rec, scores) in enumerate(zip(records, scores_list)):
        toxicity_score = sum(scores.values())
        row = {
            "role": rec.get("role", "original" if i == 0 else "alternative"),
            "name": rec.get("name", ""),
            "smiles": rec["smiles"],
            "similarity": rec.get("similarity"),
            "chembl_id": rec.get("chembl_id", ""),
            "toxric_match": "toxric_labels" in rec,
            "toxicity_score": round(toxicity_score, 4),
        }
        row.update({k: round(v, 4) for k, v in scores.items()})
        rows.append(row)

    df = pd.DataFrame(rows)

    original = df[df["role"] == "original"]
    alternatives = df[df["role"] == "alternative"].sort_values("toxicity_score", ascending=True)

    if is_fallback:
        alternatives = alternatives.head(top_k)

    return pd.concat([original, alternatives], ignore_index=True)
