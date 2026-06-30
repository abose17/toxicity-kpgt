"""
Multi-task fine-tuning loop with masked loss.

Designed to handle TOXRIC's natural sparsity: any compound may have
labels for some endpoints and NaN for others. We compute per-task
loss only where the label is finite, so NaN entries contribute
zero gradient and don't poison training.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import DataLoader

from .config import TrainConfig
from .model import KPGTMultiTaskFineTuner, build_pretrained_predictor


def _ensure_kpgt_on_path(kpgt_dir: str = "external/KPGT") -> None:
    p = str(Path(kpgt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_types: list[str],
) -> torch.Tensor:
    """Per-task BCE/MSE loss, summed over tasks, masking NaN targets.

    logits, targets: (batch_size, n_tasks). task_types[i] is 'classification' or 'regression'.
    """
    total = logits.new_tensor(0.0)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    for i, kind in enumerate(task_types):
        col_logits = logits[:, i]
        col_target = targets[:, i]
        mask = torch.isfinite(col_target)
        if mask.sum() == 0:
            continue
        if kind == "classification":
            l = bce(col_logits[mask], col_target[mask].float())
        else:
            l = mse(col_logits[mask], col_target[mask].float())
        total = total + l.mean()

    return total


def _compute_metric(
    all_logits: np.ndarray, all_targets: np.ndarray, task_types: list[str]
) -> dict:
    """Per-task ROC-AUC for classification, RMSE for regression. NaN-aware."""
    out: dict[str, float] = {}
    for i, kind in enumerate(task_types):
        y = all_targets[:, i]
        p = all_logits[:, i]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        if kind == "classification":
            try:
                out[f"task{i}_auc"] = float(roc_auc_score(y[mask], p[mask]))
            except ValueError:
                pass  # only one class present in batch
        else:
            out[f"task{i}_rmse"] = float(np.sqrt(mean_squared_error(y[mask], p[mask])))
    return out


def build_dataloader(cfg: TrainConfig, split: str) -> DataLoader:
    """Construct a DataLoader using KPGT's MoleculeDataset + Collator_tune."""
    _ensure_kpgt_on_path()
    from src.data.finetune_dataset import MoleculeDataset  # noqa: E402
    from src.data.collator import Collator_tune  # noqa: E402

    dataset = MoleculeDataset(
        root_path=cfg.data_root,
        dataset=cfg.dataset_name,
        dataset_type="multi-task",
        path_length=5,
        n_virtual_nodes=2,
        split_name=cfg.split_name,
        split=split,
    )
    collator = Collator_tune(max_length=5, n_virtual_nodes=2, add_self_loop=True)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )


def train(cfg: TrainConfig) -> dict:
    """Run a full fine-tuning loop. Returns the best val metrics dict."""
    _seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    base = build_pretrained_predictor(pretrained_path=cfg.pretrained_path)
    model = KPGTMultiTaskFineTuner(
        base, n_tasks=cfg.n_tasks,
        head_hidden_dim=cfg.head_hidden_dim, head_dropout=cfg.head_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.backbone_lr, cfg.head_lr, cfg.weight_decay)
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(cfg.n_epochs):
        # --- train ---
        model.train()
        running = 0.0
        for batch in train_loader:
            # KPGT's Collator_tune returns: (smiles, graphs, fps, mds, labels)
            _, g, fp, md, y = batch
            g, fp, md, y = g.to(device), fp.to(device), md.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(g, fp, md)
            loss = masked_loss(logits, y, cfg.task_types)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item()

        # --- validate ---
        model.eval()
        v_loss = 0.0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                _, g, fp, md, y = batch
                g, fp, md, y = g.to(device), fp.to(device), md.to(device), y.to(device)
                logits = model(g, fp, md)
                v_loss += masked_loss(logits, y, cfg.task_types).item()
                all_logits.append(logits.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        metrics = _compute_metric(
            np.concatenate(all_logits), np.concatenate(all_targets), cfg.task_types
        )
        print(
            f"epoch {epoch+1:>3}: train_loss={running/len(train_loader):.4f}  "
            f"val_loss={v_loss/len(val_loader):.4f}  metrics={metrics}"
        )

        # --- checkpoint best ---
        if v_loss < best_val:
            best_val = v_loss
            best_metrics = metrics
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                 "epoch": epoch, "val_loss": v_loss},
                ckpt_dir / "best.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stop_patience:
                print(f"early stop at epoch {epoch+1}")
                break

    return best_metrics
