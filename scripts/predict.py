"""
CLI: SMILES → toxicity predictions + plain-English explanation.

Usage:
    python scripts/predict.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
    python scripts/predict.py --smiles "CCO" "c1ccccc1" --top-k 10
    python scripts/predict.py --smiles "CCO" --no-explain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toxpkg.explainer import explain_predictions
from toxpkg.predict import load_finetuned_model, predict_smiles, scores_per_endpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", nargs="+", required=True, help="One or more SMILES strings.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to fine-tuned model.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=10, help="How many top-scoring endpoints to print.")
    p.add_argument("--no-explain", action="store_true",
                   help="Skip Claude explanation (raw scores only).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"{ckpt_path} not found. Train a model first with scripts/train.py."
        )

    print(f"[load] {ckpt_path}")
    model, cfg = load_finetuned_model(str(ckpt_path), device=args.device)
    task_names = cfg["task_names"]
    task_types = cfg["task_types"]
    task_type_map = dict(zip(task_names, task_types))

    logits, valid_mask = predict_smiles(model, args.smiles, device=args.device)
    if logits.shape[0] == 0:
        print("[error] no valid SMILES — all inputs failed to parse.")
        return 1

    all_scores = scores_per_endpoint(logits, task_names, task_types)
    score_iter = iter(all_scores)

    for smiles, ok in zip(args.smiles, valid_mask):
        print(f"\n{'='*70}\n{smiles}\n{'='*70}")
        if not ok:
            print("  [error] invalid SMILES — RDKit could not parse it")
            continue

        scores = next(score_iter)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        print(f"\nTop {args.top_k} endpoint scores:")
        for name, val in ranked[: args.top_k]:
            kind = task_type_map.get(name, "classification")
            label = "probability" if kind == "classification" else "value"
            print(f"  {name:<55} {label}={val:.3f}")

        if not args.no_explain:
            print("\n--- Plain-English explanation (via Claude) ---")
            try:
                explanation = explain_predictions(smiles, scores, task_types=task_type_map)
                print(explanation)
            except Exception as e:
                print(f"[explainer error] {e}")
                print("(Run with --no-explain to skip this step.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
