"""
Inference: SMILES → predictions using a fine-tuned KPGT checkpoint.

End-to-end: featurize SMILES (graph + fp + md) → batch via KPGT's
Collator_tune → run model.forward (which dispatches to LiGhT.forward_tune)
→ apply sigmoid for classification tasks → return per-endpoint scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .featurizer import featurize_smiles
from .model import KPGTMultiTaskFineTuner, build_pretrained_predictor


def _ensure_kpgt_on_path(kpgt_dir: str = "external/KPGT") -> None:
    p = str(Path(kpgt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_finetuned_model(
    checkpoint_path: str,
    kpgt_dir: str = "external/KPGT",
    device: str = "cpu",
) -> tuple[KPGTMultiTaskFineTuner, dict]:
    """Reconstruct a fine-tuned model from a Phase B checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]
    base = build_pretrained_predictor(kpgt_dir=kpgt_dir, pretrained_path=None)
    model = KPGTMultiTaskFineTuner(
        base,
        n_tasks=cfg["n_tasks"],
        head_hidden_dim=cfg.get("head_hidden_dim", 256),
        head_dropout=cfg.get("head_dropout", 0.15),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg


def predict_smiles(
    model: KPGTMultiTaskFineTuner,
    smiles_list: list[str],
    kpgt_dir: str = "external/KPGT",
    device: str = "cpu",
) -> tuple[np.ndarray, list[bool]]:
    """Run prediction on a list of SMILES.

    Returns:
        logits: (n_valid, n_tasks) numpy array of raw model outputs
        valid:  list[bool] of length len(smiles_list) — False = invalid SMILES
    """
    _ensure_kpgt_on_path(kpgt_dir)
    from src.data.collator import Collator_tune  # noqa: E402

    samples = []
    valid: list[bool] = []
    for s in smiles_list:
        feat = featurize_smiles(s, kpgt_dir=kpgt_dir)
        if feat is None:
            valid.append(False)
            continue
        g, fp, md = feat
        # Collator_tune unpacks 5 items per sample; a dummy label tensor is fine.
        samples.append((s, g, fp, md, torch.zeros(model.backbone.predictor[-1].out_features)))
        valid.append(True)

    if not samples:
        return np.zeros((0, 0), dtype=np.float32), valid

    collator = Collator_tune(max_length=5, n_virtual_nodes=2, add_self_loop=True)
    _, batched_g, fps, mds, _ = collator(samples)
    batched_g = batched_g.to(device)
    fps = fps.to(device)
    mds = mds.to(device)

    with torch.no_grad():
        logits = model(batched_g, fps, mds)
    return logits.cpu().numpy().astype(np.float32), valid


def scores_per_endpoint(
    logits: np.ndarray,
    task_names: list[str],
    task_types: list[str],
) -> list[dict[str, float]]:
    """Convert (n_samples, n_tasks) logits → list of {endpoint: score} dicts.

    For classification, score is sigmoid(logit) ∈ [0, 1] (probability).
    For regression, score is the raw logit value.
    """
    out: list[dict[str, float]] = []
    for row in logits:
        d: dict[str, float] = {}
        for i, (name, kind) in enumerate(zip(task_names, task_types)):
            v = float(row[i])
            if kind == "classification":
                d[name] = float(1.0 / (1.0 + np.exp(-v)))
            else:
                d[name] = v
        out.append(d)
    return out
