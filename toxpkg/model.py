"""
Fine-tuning model: pretrained KPGT backbone + a custom multi-task head.

This mirrors KPGT's own scripts/finetune.py exactly:
  1. Instantiate LiGhTPredictor with the base config.
  2. Load `base.pth` into it (strict=False; aux pretraining heads load too).
  3. Replace `predictor` (a new attribute) with a task-sized MLP.
  4. Delete the three aux pretraining heads (md/fp/node_predictor).
  5. Call `forward_tune(g, fp, md)` instead of `forward()`, which concatenates
     [fp_virtual_node, md_virtual_node, atom_readout] → 3 × d_g_feats = 2304
     and runs it through the new predictor.

The earlier Phase B version of this file used `triplet_h[indicators==1]`
directly as the graph embedding (768 dims) — that produced the wrong shape
and would have given poor results. Use `forward_tune` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .config import KPGT_BASE_CONFIG


def _ensure_kpgt_on_path(kpgt_dir: str = "external/KPGT") -> None:
    p = str(Path(kpgt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_predictor(d_input: int, n_tasks: int, n_layers: int,
                     dropout: float, hidden_dim: int) -> nn.Module:
    """Same construction pattern as KPGT/scripts/finetune.py:get_predictor."""
    if n_layers == 1:
        return nn.Linear(d_input, n_tasks)
    layers: list[nn.Module] = [
        nn.Linear(d_input, hidden_dim),
        nn.Dropout(dropout),
        nn.GELU(),
    ]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, n_tasks))
    return nn.Sequential(*layers)


def build_pretrained_predictor(
    kpgt_dir: str = "external/KPGT",
    pretrained_path: str | None = None,
):
    """Instantiate KPGT's LiGhTPredictor with the base config and load weights.

    Pass pretrained_path=None to build an empty backbone (used by predict.py
    when loading a fine-tuned checkpoint).
    """
    _ensure_kpgt_on_path(kpgt_dir)
    from src.model.light import LiGhTPredictor  # noqa: E402

    model = LiGhTPredictor(**KPGT_BASE_CONFIG)

    if pretrained_path is not None and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu")
        # base.pth is a raw state_dict; finetune.py strips DataParallel's 'module.' prefix
        if isinstance(state, dict) and "model" in state and not any(
            k.startswith(("node_emb", "model")) for k in state.keys()
        ):
            state = state["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # Drop any keys whose tensor shape doesn't match the current model
        # (e.g. node_predictor head has 25856 outputs in pretraining vs 1 here).
        model_state = model.state_dict()
        state = {k: v for k, v in state.items()
                 if k not in model_state or v.shape == model_state[k].shape}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[load_pretrained] missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[load_pretrained] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[load_pretrained] loaded {pretrained_path}")
    elif pretrained_path is not None:
        print(
            f"[warn] pretrained path missing — training from random init. "
            f"Did you place base.pth at {pretrained_path!r}?"
        )

    return model


class KPGTMultiTaskFineTuner(nn.Module):
    """Wraps a pretrained LiGhTPredictor and adds a custom multi-task head.

    Follows KPGT's own finetune.py pattern exactly: sets `backbone.predictor`,
    deletes aux pretraining heads, calls `backbone.forward_tune` in forward.
    """

    def __init__(
        self,
        pretrained_predictor: nn.Module,
        n_tasks: int,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.15,
        n_predictor_layers: int = 2,
    ):
        super().__init__()
        self.backbone = pretrained_predictor
        d_g = KPGT_BASE_CONFIG["d_g_feats"]
        d_input = d_g * 3   # forward_tune concatenates [fp_vn, md_vn, atom_readout]

        self.backbone.predictor = _build_predictor(
            d_input, n_tasks, n_predictor_layers, head_dropout, head_hidden_dim
        )
        # Mirror finetune.py: drop the three pretraining-only heads.
        for attr in ("md_predictor", "fp_predictor", "node_predictor"):
            if hasattr(self.backbone, attr):
                delattr(self.backbone, attr)

    def forward(self, g, fp, md):
        # forward_tune calls g.remove_nodes() in-place, so clone to avoid
        # corrupting the caller's graph between multiple forward passes.
        return self.backbone.forward_tune(g.clone(), fp, md)

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float):
        backbone_params, head_params = [], []
        for name, p in self.named_parameters():
            (head_params if "predictor" in name else backbone_params).append(p)
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
