"""
Single-SMILES featurization for inference.

Produces the exact same three inputs that KPGT's preprocess pipeline
produces in batch (graph, fingerprint, descriptor), so a fine-tuned
model can be fed individual SMILES at predict time.

Verified against KPGT's `scripts/preprocess_downstream_dataset.py` and
`src/data/finetune_dataset.py`:
    - graph:  smiles_to_graph_tune(smiles, max_length=5, n_virtual_nodes=2)
    - fp:     Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)
    - md:     RDKit2DNormalized().process(smiles)[1:]   # drop the leading flag column
    - NaN md values are replaced with 0 (the dataset does this on load)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem


def _ensure_kpgt_on_path(kpgt_dir: str = "external/KPGT") -> None:
    """KPGT isn't pip-installable — it's source-only. Add it to sys.path."""
    p = str(Path(kpgt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def featurize_smiles(
    smiles: str,
    kpgt_dir: str = "external/KPGT",
    path_length: int = 5,
    n_virtual_nodes: int = 2,
):
    """Featurize a single SMILES into (dgl_graph, fp_tensor, md_tensor).

    Returns:
        graph: dgl.DGLGraph with the same node/edge features KPGT expects
        fp:    torch.FloatTensor of shape (512,)
        md:    torch.FloatTensor of shape (200,)
    Or None if the SMILES is invalid.
    """
    _ensure_kpgt_on_path(kpgt_dir)
    from src.data.featurizer import smiles_to_graph_tune
    from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    graph = smiles_to_graph_tune(
        smiles, max_length=path_length, n_virtual_nodes=n_virtual_nodes, add_self_loop=True
    )
    if graph is None:
        return None

    fp_bits = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)
    fp = torch.tensor(list(fp_bits), dtype=torch.float)

    generator = RDKit2DNormalized()
    md_full = np.asarray(generator.process(smiles), dtype=np.float32)
    md_arr = md_full[1:]  # drop the leading flag column (KPGT preprocess does the same)
    md_arr = np.where(np.isnan(md_arr), 0, md_arr)
    md = torch.tensor(md_arr, dtype=torch.float)

    return graph, fp, md
