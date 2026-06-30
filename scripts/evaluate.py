"""
Evaluate a fine-tuned KPGT checkpoint on the held-out test split.

Reports per-task ROC-AUC (classification) or RMSE (regression),
mean AUC across all classification tasks, and saves a ranked CSV.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --dataset toxric_multitask \
        --data-root data/kpgt-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toxpkg.model import KPGTMultiTaskFineTuner, build_pretrained_predictor
from toxpkg.config import TrainConfig
from toxpkg.trainer import _ensure_kpgt_on_path, _compute_metric, build_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--dataset", default="toxric_multitask")
    p.add_argument("--data-root", default="data/kpgt-cache")
    p.add_argument("--split-name", default="random_0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def infer_task_columns(csv_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c != "smiles"]
    types = []
    for c in label_cols:
        unique = df[c].dropna().unique()
        if set(unique).issubset({0, 1, 0.0, 1.0, True, False}):
            types.append("classification")
        else:
            types.append("regression")
    return label_cols, types


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    csv_path = Path(args.data_root) / args.dataset / f"{args.dataset}.csv"
    task_names, task_types = infer_task_columns(csv_path)
    print(f"[tasks] {len(task_names)} endpoints")

    # Load checkpoint — contains state_dict + the cfg dict used during training
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = ckpt.get("cfg", {})
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"[loaded] {ckpt_path}  (epoch {epoch + 1 if isinstance(epoch, int) else epoch}, val_loss={val_loss:.4f})")

    # Rebuild the same architecture that produced this checkpoint
    # (PyTorch saves weights only — the model class is needed to load them into)
    cfg = TrainConfig(
        data_root=args.data_root,
        dataset_name=args.dataset,
        split_name=args.split_name,
        n_tasks=len(task_names),
        task_names=task_names,
        task_types=task_types,
        batch_size=args.batch_size,
        device=args.device,
        pretrained_path=saved_cfg.get("pretrained_path", "external/KPGT/models/pretrained/base/base.pth"),
        head_hidden_dim=saved_cfg.get("head_hidden_dim", 256),
        head_dropout=saved_cfg.get("head_dropout", 0.15),
    )

    _ensure_kpgt_on_path()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    base = build_pretrained_predictor(pretrained_path=cfg.pretrained_path)
    model = KPGTMultiTaskFineTuner(
        base, n_tasks=cfg.n_tasks,
        head_hidden_dim=cfg.head_hidden_dim,
        head_dropout=cfg.head_dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Run inference on the test split
    test_loader = build_dataloader(cfg, split="test")
    print(f"[eval] running inference on test split ...")

    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            _, g, fp, md, y = batch
            g, fp, md, y = g.to(device), fp.to(device), md.to(device), y.to(device)
            logits = model(g, fp, md)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    metrics = _compute_metric(
        np.concatenate(all_logits),
        np.concatenate(all_targets),
        task_types,
    )

    # Build results table
    rows = []
    for i, name in enumerate(task_names):
        if f"task{i}_auc" in metrics:
            rows.append({"endpoint": name, "metric": "AUC", "value": metrics[f"task{i}_auc"]})
        elif f"task{i}_rmse" in metrics:
            rows.append({"endpoint": name, "metric": "RMSE", "value": metrics[f"task{i}_rmse"]})

    df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    auc_vals = [r["value"] for r in rows if r["metric"] == "AUC"]
    mean_auc = float(np.mean(auc_vals)) if auc_vals else float("nan")

    print("\n" + "=" * 58)
    print(f"  TEST RESULTS — {args.dataset}")
    print("=" * 58)
    print(f"  Mean AUC across {len(auc_vals)} classification tasks: {mean_auc:.4f}")
    print("=" * 58)
    print(df.to_string(index=True))
    print("=" * 58)

    out_path = Path("results") / f"{args.dataset}_test_metrics.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
