"""
Agentic toxicity prediction for a single drug molecule SMILES.

Usage:
    python scripts/predict_agentic.py --smiles "CC(=O)Nc1ccc(O)cc1"  \\
        --checkpoint checkpoints/best.pt

    python scripts/predict_agentic.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \\
        --checkpoint checkpoints/best.pt \\
        --threshold 7 \\
        --max-iter 3 \\
        --no-explain \\
        --output result.json

Pipeline (non-TOXRIC path):
    KPGT prediction → LLM validate (Inf 1) → if unsatisfactory:
    LLM suggest (Inf 2) → fp+md+graph_embed similarity → best candidate →
    repeat up to --max-iter times.

    TOXRIC-matched molecules skip straight to the explainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toxpkg.agentic_pipeline import run_agentic_pipeline
from toxpkg.model import build_pretrained_predictor
from toxpkg.predict import load_finetuned_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", required=True, help="Input drug molecule SMILES.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt",
                   help="Fine-tuned KPGT checkpoint (from scripts/train.py).")
    p.add_argument("--pretrained", default="external/KPGT/models/pretrained/base/base.pth",
                   help="Pretrained KPGT base.pth for graph embeddings.")
    p.add_argument("--merged-csv", default="data/toxric_merged.csv")
    p.add_argument("--threshold", type=float, default=6.0,
                   help="LLM confidence score threshold to accept prediction (0-10). Default 6.")
    p.add_argument("--max-iter", type=int, default=3,
                   help="Maximum self-correction iterations. Default 3.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-explain", action="store_true",
                   help="Skip Claude plain-English explanation.")
    p.add_argument("--output", default=None,
                   help="Save result JSON to this path (optional).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train a model first with scripts/train.py (or run Phase D on Azure ML)."
        )

    print(f"[load] fine-tuned model from {ckpt_path}")
    model, cfg = load_finetuned_model(str(ckpt_path), device=args.device)

    print(f"[load] pretrained backbone from {args.pretrained}")
    backbone = build_pretrained_predictor(pretrained_path=args.pretrained)
    backbone.to(args.device).eval()

    print(f"\n[run] agentic pipeline  threshold={args.threshold}  max_iter={args.max_iter}")
    result = run_agentic_pipeline(
        smiles=args.smiles,
        model=model,
        backbone=backbone,
        cfg=cfg,
        merged_csv_path=args.merged_csv,
        satisfactory_threshold=args.threshold,
        max_iter=args.max_iter,
        kpgt_dir="external/KPGT",
        device=args.device,
        explain=not args.no_explain,
    )

    _print_result(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[saved] {out_path}")

    return 0


def _print_result(r: dict) -> None:
    print(f"\n{'═' * 72}")
    print(f"  STATUS        : {r['status'].upper()}")
    print(f"  SOURCE        : {r['source'].upper()}")
    print(f"  ORIGINAL      : {r['original_smiles']}")
    if r['final_smiles'] != r['original_smiles']:
        print(f"  FINAL SMILES  : {r['final_smiles']}")
    if r['final_name']:
        print(f"  DRUG NAME     : {r['final_name']}")
    print(f"{'─' * 72}")

    print(f"\n  Iterations: {len(r['iterations'])}")
    for it in r['iterations']:
        print(f"    [{it['iteration']}] {it['name'] or it['smiles'][:40]:<40} "
              f"conf={it['confidence']:.0f}/10")
        if it['suggestions']:
            print(f"        suggestions: {', '.join(it['suggestions'])}")

    print(f"\n  Top 10 toxicity scores (final):")
    ranked = sorted(r['scores'].items(), key=lambda kv: -kv[1])
    for ep, sc in ranked[:10]:
        bar = "█" * int(sc * 20)
        print(f"    {ep:<50} {sc:.3f} {bar}")

    if r.get('explanation'):
        print(f"\n{'─' * 72}")
        print(r['explanation'])

    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    sys.exit(main())
