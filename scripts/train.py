"""
Entry point for fine-tuning. Inspects the prepared dataset to infer
n_tasks and task_types, then runs the training loop.

Usage:
    python scripts/train.py --dataset toxric_hepatotoxicity --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make src importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toxpkg.config import TrainConfig
from toxpkg.trainer import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True, help="Name of a prepared dataset under data/kpgt-cache/")
    p.add_argument("--data-root", default="data/kpgt-cache")
    p.add_argument("--split-name", default="random_0")
    p.add_argument("--pretrained", default="external/KPGT/models/pretrained/base/base.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Where to save best.pt. On Azure ML, pass ${{outputs.model_dir}}.")
    p.add_argument("--task-types", nargs="+", default=None,
                   help="Per-column task type: 'classification' or 'regression'. "
                        "If omitted, inferred from dtype + unique value count.")
    return p.parse_args()


def infer_task_columns(csv_path: Path) -> tuple[list[str], list[str]]:
    """Return (task_names, task_types) from CSV column order + value sniff."""
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c != "smiles"]
    types = []
    for c in label_cols:
        s = df[c].dropna()
        unique = s.unique()
        if set(unique).issubset({0, 1, 0.0, 1.0, True, False}):
            types.append("classification")
        else:
            types.append("regression")
    return label_cols, types


def main() -> int:
    args = parse_args()
    csv_path = Path(args.data_root) / args.dataset / f"{args.dataset}.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Run `python scripts/preprocess.py --source ... --dataset {args.dataset}` first."
        )

    task_names, inferred_types = infer_task_columns(csv_path)
    task_types = args.task_types or inferred_types
    print(f"[tasks] {len(task_names)} endpoints: {task_names[:5]}{'...' if len(task_names) > 5 else ''}")
    print(f"[tasks] types: {task_types[:5]}{'...' if len(task_types) > 5 else ''}")

    cfg = TrainConfig(
        data_root=args.data_root,
        dataset_name=args.dataset,
        split_name=args.split_name,
        pretrained_path=args.pretrained,
        n_tasks=len(task_names),
        task_names=task_names,
        task_types=task_types,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    best_metrics = train(cfg)
    print(f"\n[done] best val metrics: {best_metrics}")
    print(f"[done] checkpoint saved to {cfg.checkpoint_dir}/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
