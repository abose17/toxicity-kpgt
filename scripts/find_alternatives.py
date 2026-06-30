"""
Find safer structural alternatives for a given drug molecule.

Usage:
    python scripts/find_alternatives.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \\
        --checkpoint checkpoints/best.pt

    python scripts/find_alternatives.py --smiles "CC(=O)Nc1ccc(O)cc1" \\
        --checkpoint checkpoints/best.pt \\
        --threshold 0.6 \\
        --output-dir results/paracetamol/

Pipeline:
    1. ChEMBL similarity search  → up to 20 structurally similar SMILES
    2. TOXRIC matching           → keep only TOXRIC-present molecules
                                   (fallback: all 20 if none match, capped at 5)
    3. KPGT prediction           → toxicity scores across 30 endpoints
    4. Rank + compare            → original vs alternatives, sorted safest-first
    5. Save PNG plots            → molecule_grid.png, toxicity_heatmap.png, mcs_highlight.png
    6. Save comparison table     → comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toxpkg.comparator import rank_by_toxicity
from toxpkg.predict import load_finetuned_model, predict_smiles, scores_per_endpoint
from toxpkg.similarity import fetch_similar_chembl, lookup_name_pubchem
from toxpkg.toxric_matcher import filter_by_toxric
from toxpkg.visualizer import draw_mcs_highlight, draw_molecule_grid, draw_toxicity_heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", required=True, help="Input drug molecule SMILES.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt",
                   help="Fine-tuned KPGT checkpoint (from scripts/train.py).")
    p.add_argument("--merged-csv", default="data/toxric_merged.csv",
                   help="TOXRIC merged CSV (from scripts/merge_toxric.py).")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="ChEMBL Tanimoto similarity threshold [0-1]. Default 0.7.")
    p.add_argument("--n-similar", type=int, default=20,
                   help="Number of similar molecules to fetch from ChEMBL.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="results/alternatives",
                   help="Directory to save comparison.csv and PNG plots.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train a model first with scripts/train.py (or run Phase D on Azure ML)."
        )

    # ── 0. Look up original molecule name ───────────────────────────────────
    print("[0/5] Looking up original molecule name via PubChem…")
    original_name = lookup_name_pubchem(args.smiles)
    if original_name:
        print(f"      Name: {original_name}")
    else:
        print("      Name not found — will display SMILES only.")

    # ── 1. ChEMBL similarity search ─────────────────────────────────────────
    print(f"[1/5] Searching ChEMBL for molecules similar to input (threshold={args.threshold})…")
    candidates = fetch_similar_chembl(args.smiles, threshold=args.threshold, n=args.n_similar)
    if not candidates:
        print("[warn] ChEMBL returned no results. Try lowering --threshold.")
        return 1
    print(f"      Found {len(candidates)} candidates.")

    # ── 2. TOXRIC matching ───────────────────────────────────────────────────
    print(f"[2/5] Matching candidates against TOXRIC ({args.merged_csv})…")
    filtered, is_fallback = filter_by_toxric(candidates, merged_csv_path=args.merged_csv)
    if is_fallback:
        print(f"      [fallback] No TOXRIC matches — running KPGT on all {len(filtered)} "
              "candidates, returning top-5 by lowest toxicity.")
    else:
        print(f"      {len(filtered)} TOXRIC match(es) found.")

    # ── 3. KPGT prediction ───────────────────────────────────────────────────
    print("[3/5] Loading model and running predictions…")
    model, cfg = load_finetuned_model(str(ckpt_path), device=args.device)
    task_names: list[str] = cfg["task_names"]
    task_types: list[str] = cfg["task_types"]

    # Prepend original molecule as first record
    all_records = [{"smiles": args.smiles, "role": "original", "name": original_name}] + [
        {**r, "role": "alternative"} for r in filtered
    ]
    all_smiles = [r["smiles"] for r in all_records]

    logits, valid_mask = predict_smiles(model, all_smiles, device=args.device)
    if logits.shape[0] == 0:
        print("[error] All SMILES failed to featurize.")
        return 1

    all_scores = scores_per_endpoint(logits, task_names, task_types)

    # Re-align records to valid SMILES only
    valid_records = [r for r, ok in zip(all_records, valid_mask) if ok]

    # ── 4. Rank and compare ──────────────────────────────────────────────────
    print("[4/5] Ranking by toxicity score (sum of sigmoid probabilities)…")
    comparison_df = rank_by_toxicity(
        valid_records,
        all_scores,
        is_fallback=is_fallback,
        top_k=5,
    )

    csv_path = out_dir / "comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"      Saved: {csv_path}")

    _print_summary(comparison_df)

    # ── 5. Generate PNG plots ────────────────────────────────────────────────
    print("[5/5] Generating plots…")

    smiles_col = list(comparison_df["smiles"])
    names_col = list(comparison_df["name"])
    roles = list(comparison_df["role"])
    tox_scores = list(comparison_df["toxicity_score"])
    labels = [
        "ORIGINAL" if r == "original" else f"Alt {i}"
        for i, r in enumerate(roles)
    ]

    grid_path = out_dir / "molecule_grid.png"
    draw_molecule_grid(smiles_col, labels, tox_scores, str(grid_path), names=names_col)
    print(f"      Saved: {grid_path}")

    heatmap_path = out_dir / "toxicity_heatmap.png"
    draw_toxicity_heatmap(comparison_df, str(heatmap_path))
    print(f"      Saved: {heatmap_path}")

    alternatives = comparison_df[comparison_df["role"] == "alternative"]
    if not alternatives.empty:
        best_smiles = alternatives.iloc[0]["smiles"]
        best_name = str(alternatives.iloc[0].get("name", "") or "")
        mcs_path = out_dir / "mcs_highlight.png"
        draw_mcs_highlight(
            args.smiles, best_smiles, str(mcs_path),
            name_query=original_name, name_best=best_name,
        )
        print(f"      Saved: {mcs_path}")

    print(f"\nDone. All outputs in: {out_dir.resolve()}")
    return 0


def _print_summary(df) -> None:
    print(f"\n{'─'*82}")
    print(f"  {'ROLE':<12} {'NAME':<22} {'TOX_SCORE':>10}  {'SMILES'}")
    print(f"{'─'*82}")
    for _, row in df.iterrows():
        role = row["role"].upper()
        nm = str(row.get("name", "") or row.get("chembl_id", ""))[:21]
        score = f"{row['toxicity_score']:.4f}"
        smiles = str(row["smiles"])[:34] + ("…" if len(str(row["smiles"])) > 34 else "")
        marker = " ◄ best" if row.name == 1 else ""
        print(f"  {role:<12} {nm:<22} {score:>10}  {smiles}{marker}")
    print(f"{'─'*82}\n")


if __name__ == "__main__":
    sys.exit(main())
