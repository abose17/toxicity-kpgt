"""
Central config for the fine-tuning pipeline.

KPGT_BASE_CONFIG mirrors the 'base' config in `KPGT/src/model_config.py` —
the dims MUST match those values or the pretrained `base.pth` weights
will not load. Do not change KPGT_BASE_CONFIG fields unless you also
re-pretrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Verified 2026-06-09 against KPGT/src/model_config.py 'base' entry.
# Do not modify — these dims are baked into base.pth.
KPGT_BASE_CONFIG: dict = {
    "d_node_feats": 137,
    "d_edge_feats": 14,
    "d_g_feats": 768,
    "d_fp_feats": 512,
    "d_md_feats": 200,
    "d_hpath_ratio": 12,
    "n_mol_layers": 12,
    "path_length": 5,
    "n_heads": 12,
    "n_ffn_dense_layers": 2,
    "input_drop": 0.0,
    "attn_drop": 0.1,
    "feat_drop": 0.1,
    "n_node_types": 1,
    "readout_mode": "mean",
}


@dataclass
class TrainConfig:
    """Hyperparameters and paths for one fine-tuning run."""

    # Data
    data_root: str = "data/kpgt-cache"          # KPGT-cache root (one subdir per dataset)
    dataset_name: str = "toxric_multitask"      # subfolder + filename stem
    split_name: str = "random_0"                # splits/random_0.npy under dataset dir

    # Where the LiGhT pretrained weights live (Phase A's `check_kpgt_pretrained` checks this)
    pretrained_path: str = "external/KPGT/models/pretrained/base/base.pth"

    # Output
    checkpoint_dir: str = "checkpoints"

    # Architecture (must match KPGT_BASE_CONFIG for the backbone)
    head_hidden_dim: int = 256
    head_dropout: float = 0.15

    # Tasks — populated from the CSV's non-SMILES columns at runtime
    n_tasks: int = 0
    task_names: list[str] = field(default_factory=list)  # endpoint names in column order
    task_types: list[str] = field(default_factory=list)  # 'classification' or 'regression' per task

    # Optimizer
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-6

    # Loop
    n_epochs: int = 50
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"        # falls back to cpu if unavailable
    early_stop_patience: int = 10
