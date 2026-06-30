"""
Standalone PNG visualizations for the safer-alternatives pipeline.

All functions write to a file path — no display calls — so output
can be loaded directly into Streamlit via st.image().
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_META_COLS = {"role", "name", "similarity", "chembl_id", "toxric_match", "toxicity_score", "source"}


def draw_molecule_grid(
    smiles_list: list[str],
    labels: list[str],
    scores: list[float],
    output_path: str,
    names: list[str] | None = None,
    mol_size: tuple[int, int] = (300, 220),
    mols_per_row: int = 4,
) -> None:
    """RDKit grid image of molecules with drug name and toxicity score in the legend.

    Args:
        smiles_list:  SMILES for each molecule (original first).
        labels:       Short role label per molecule (e.g. "ORIGINAL", "Alt 1").
        scores:       Toxicity score per molecule.
        output_path:  Destination PNG path.
        names:        Optional drug/compound name per molecule. Shown above the score.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    names = names or [""] * len(smiles_list)
    mols, legends = [], []
    for s, lbl, sc, nm in zip(smiles_list, labels, scores, names):
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols.append(mol)
            name_line = f"{nm}\n" if nm else ""
            legends.append(f"{lbl}\n{name_line}Tox score: {sc:.3f}")

    if not mols:
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        legends=legends,
    )
    img.save(output_path)


def draw_toxicity_heatmap(
    comparison_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Seaborn heatmap: rows = molecules, columns = 30 toxicity endpoints.

    Args:
        comparison_df: DataFrame from comparator.rank_by_toxicity().
        output_path:   Destination PNG path.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    endpoint_cols = [c for c in comparison_df.columns if c not in _META_COLS | {"smiles"}]
    heat_data = comparison_df[endpoint_cols].astype(float)

    row_labels = []
    for _, row in comparison_df.iterrows():
        nm = str(row.get("name", "") or "")
        cid = str(row.get("chembl_id", "") or "")
        tag = nm if nm else (cid if cid else str(row["smiles"])[:18])
        prefix = "[ORIG] " if row["role"] == "original" else "[ALT] "
        row_labels.append(prefix + (tag[:28] + "…" if len(tag) > 28 else tag))
    heat_data.index = row_labels

    fig_w = max(14, len(endpoint_cols) * 0.45)
    fig_h = max(3, len(comparison_df) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        heat_data,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="RdYlGn_r",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Toxicity Probability", "shrink": 0.6},
    )
    ax.set_title("Toxicity Endpoint Comparison", fontsize=13, pad=10)
    ax.set_xlabel("Endpoint", fontsize=9)
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_mcs_highlight(
    smiles_query: str,
    smiles_best: str,
    output_path: str,
    name_query: str = "",
    name_best: str = "",
    mol_size: tuple[int, int] = (420, 300),
) -> None:
    """Two-panel image highlighting the Maximum Common Substructure (MCS)
    between the original query and the best (safest) alternative.

    Args:
        smiles_query: Original input SMILES.
        smiles_best:  Safest alternative SMILES.
        output_path:  Destination PNG path.
        name_query:   Common name of the original molecule (shown in legend).
        name_best:    Common name of the best alternative (shown in legend).
    """
    from rdkit import Chem
    from rdkit.Chem import Draw, rdFMCS

    mol_q = Chem.MolFromSmiles(smiles_query)
    mol_b = Chem.MolFromSmiles(smiles_best)
    if mol_q is None or mol_b is None:
        return

    mcs_result = rdFMCS.FindMCS([mol_q, mol_b], timeout=5, completeRingsOnly=True)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString) if mcs_result.numAtoms > 0 else None

    query_match = list(mol_q.GetSubstructMatch(mcs_mol)) if mcs_mol else []
    best_match = list(mol_b.GetSubstructMatch(mcs_mol)) if mcs_mol else []

    lbl_q = f"Query: {name_query}" if name_query else "Query (Original)"
    lbl_b = f"Best Alt: {name_best}" if name_best else "Best Alternative (Lowest Toxicity)"

    img = Draw.MolsToGridImage(
        [mol_q, mol_b],
        molsPerRow=2,
        subImgSize=mol_size,
        highlightAtomLists=[query_match, best_match],
        legends=[lbl_q, lbl_b],
    )
    img.save(output_path)
