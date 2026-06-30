"""
Generate `pipeline.ipynb` — a step-by-step notebook that walks through
the entire TOXRIC + KPGT pipeline by calling functions from src/ and
scripts/. Re-run this script any time you want to regenerate the notebook.

Usage from labfiles/toxicity-kpgt/:
    python scripts/_build_pipeline_notebook.py

Output:
    pipeline.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path


# --- tiny cell-builder helpers (so cell sources stay readable below) -----

CELLS: list[dict] = []


def md(*lines: str) -> None:
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
    })


def code(*lines: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    })


# =========================================================================
# Notebook content
# =========================================================================

md(
    "# TOXRIC + KPGT — End-to-End Pipeline Notebook",
    "",
    "Walks through every stage of the pipeline by calling functions from this project's `src/` and `scripts/`. ",
    "Each section is self-describing and can be run in order from top to bottom.",
    "",
    "**Sections:**",
    "1. Single-target TOXRIC data — what one of the 30 source CSVs looks like",
    "2. Convert single-target → multi-target (the merge step)",
    "3. KPGT preprocessing (graphs + fingerprints + descriptors)",
    "4. Featurize one SMILES (inspect what the model actually sees)",
    "5. Build the fine-tuning model (pretrained KPGT + multi-task head)",
    "6. Training smoke-test (one batch, one optimizer step)",
    "7. Run prediction on a few SMILES",
    "8. Generate plain-English explanation via Claude",
    "",
    "**Dependency tiers** (each section's first cell prints what's missing):",
    "- 1 & 2: `pandas`, `numpy`",
    "- 3, 4, 5, 6, 7: `rdkit`, `torch`, `dgl`, `dgllife`, `scipy`, `scikit-learn`",
    "- 8: `anthropic`, `azure-identity` (+ a populated `.env`)",
)

md("## Setup")

code(
    "import os, sys",
    "from pathlib import Path",
    "",
    "# Ensure we run from the toxicity-kpgt directory so src/ imports + relative paths work.",
    "ROOT = Path.cwd()",
    "if ROOT.name != 'toxicity-kpgt':",
    "    candidate = ROOT / 'labfiles' / 'toxicity-kpgt'",
    "    if candidate.exists():",
    "        os.chdir(candidate)",
    "        ROOT = Path.cwd()",
    "print('cwd:', ROOT)",
    "",
    "# Make src/ importable",
    "sys.path.insert(0, str(ROOT))",
    "print('python:', sys.version.split()[0])",
)

# -------- Step 1 --------
md(
    "## Step 1 — Single-target TOXRIC data",
    "",
    "TOXRIC's `toxric_30_datasets.zip` extracts to 30 CSVs, each covering ONE toxicity endpoint. ",
    "Schema is the same across all 30: `TAID, Name, IUPAC Name, PubChem CID, Canonical SMILES, InChIKey, Toxicity Value`. ",
    "The `Toxicity Value` column is binary (0 = inactive, 1 = active for that endpoint).",
)

code(
    "import pandas as pd",
    "from toxpkg.data_utils import list_toxric_subdatasets, inspect_csv, print_inspection",
    "",
    "TOXRIC_DIR = ROOT / 'data' / 'toxric' / 'toxric_30_datasets' / 'toxric_30_datasets'",
    "csvs = list_toxric_subdatasets(TOXRIC_DIR)",
    "print(f'Found {len(csvs)} endpoint CSVs.')",
    "for p in csvs[:6]:",
    "    print(' -', p.name)",
    "print(' ...')",
)

code(
    "# Inspect one endpoint to see the source schema",
    "info = inspect_csv(csvs[csvs.index([p for p in csvs if 'Hepatotoxicity' in p.name][0])])",
    "print_inspection(info)",
)

code(
    "# Class distribution for this single endpoint",
    "df_one = pd.read_csv([p for p in csvs if 'Hepatotoxicity' in p.name][0])",
    "print('Rows:', len(df_one))",
    "print('Label distribution:')",
    "print(df_one['Toxicity Value'].value_counts())",
    "df_one.head()",
)

# -------- Step 2 --------
md(
    "## Step 2 — Convert single-target → multi-target",
    "",
    "Each TOXRIC CSV alone is one endpoint. To do multi-task learning we outer-join all 30 on `Canonical SMILES`. ",
    "Compounds tested in some endpoints but not others get `NaN` for the missing columns. ",
    "The masked loss in `src/trainer.py` ignores `NaN` per task during training.",
)

code(
    "from scripts.merge_toxric import merge_toxric_csvs",
    "",
    "merged = merge_toxric_csvs(TOXRIC_DIR)",
    "print('shape:', merged.shape)",
    "print('first 3 columns:', merged.columns.tolist()[:3])",
    "print('all endpoint columns:')",
    "for c in merged.columns[1:]:",
    "    print(' -', c)",
)

code(
    "# How sparse is the multi-target matrix? Per-endpoint label density.",
    "endpoint_cols = [c for c in merged.columns if c != 'smiles']",
    "density = (merged[endpoint_cols].notna().sum() / len(merged) * 100).sort_values(ascending=False)",
    "print('Compounds:', len(merged))",
    "print('Endpoints:', len(endpoint_cols))",
    "print(f'Average label density per endpoint: {density.mean():.1f}%')",
    "print()",
    "print('Per-endpoint coverage (% of compounds with a label):')",
    "for name, pct in density.head(10).items():",
    "    print(f'  {name:<55} {pct:>5.1f}%')",
    "print('  ...')",
)

code(
    "# How many endpoints does each compound have a label for?",
    "labels_per_compound = merged[endpoint_cols].notna().sum(axis=1)",
    "print('Labels per compound (distribution):')",
    "print(labels_per_compound.describe())",
    "print()",
    "print('Histogram (bin = number of endpoints labelled):')",
    "print(labels_per_compound.value_counts().sort_index().head(15))",
)

code(
    "# Save the merged CSV in KPGT's expected location.",
    "OUT_DIR = ROOT / 'data' / 'kpgt-cache' / 'toxric_multitask'",
    "OUT_DIR.mkdir(parents=True, exist_ok=True)",
    "out_csv = OUT_DIR / 'toxric_multitask.csv'",
    "merged.to_csv(out_csv, index=False)",
    "print(f'wrote {out_csv} ({out_csv.stat().st_size/1e6:.2f} MB)')",
)

# -------- Step 3 --------
md(
    "## Step 3 — KPGT preprocessing",
    "",
    "KPGT's `MoleculeDataset` expects three cached files per dataset directory:",
    "- `{dataset}_5.pkl` — DGL graph objects (built from SMILES)",
    "- `rdkfp1-7_512.npz` — RDKit fingerprints, 512 bits, path lengths 1–7 (sparse)",
    "- `molecular_descriptors.npz` — 200-dim normalized 2D descriptors",
    "",
    "Plus a split index file at `splits/{name}.npy` for train/val/test partition.",
    "",
    "Our `scripts/preprocess.py` writes the split, then subprocesses KPGT's own `preprocess_downstream_dataset.py` to produce the three cache files. ",
    "**Heavy deps required:** `rdkit`, `dgl`, `dgllife`, `scipy`, `numpy`.",
)

code(
    "# Sanity check that heavy deps are installed BEFORE we try preprocessing.",
    "missing = []",
    "for m in ('rdkit', 'dgl', 'dgllife', 'scipy', 'numpy'):",
    "    try:",
    "        __import__(m)",
    "    except ImportError:",
    "        missing.append(m)",
    "if missing:",
    "    print('[skip] missing deps:', missing)",
    "    print('Install them, then re-run this cell. The remaining cells in this section will not work without them.')",
    "else:",
    "    print('[ok] all preprocess deps importable')",
)

code(
    "# Generate the split file (random 80/10/10).",
    "import numpy as np",
    "rng = np.random.default_rng(42)",
    "idx = rng.permutation(len(merged))",
    "n_test = int(len(merged) * 0.1)",
    "n_val  = int(len(merged) * 0.1)",
    "test_idx = idx[:n_test]",
    "val_idx  = idx[n_test:n_test+n_val]",
    "train_idx = idx[n_test+n_val:]",
    "splits_dir = OUT_DIR / 'splits'",
    "splits_dir.mkdir(exist_ok=True)",
    "split_path = splits_dir / 'random_0.npy'",
    "np.save(split_path, np.array([train_idx, val_idx, test_idx], dtype=object), allow_pickle=True)",
    "print(f'splits  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}  -> {split_path}')",
)

code(
    "# Run KPGT's preprocess as a subprocess. Takes a few minutes for ~15k compounds.",
    "import subprocess",
    "kpgt_preprocess = ROOT / 'external' / 'KPGT' / 'scripts' / 'preprocess_downstream_dataset.py'",
    "if kpgt_preprocess.exists():",
    "    cmd = [sys.executable, str(kpgt_preprocess),",
    "           '--data_path', str((ROOT / 'data' / 'kpgt-cache').resolve()),",
    "           '--dataset', 'toxric_multitask',",
    "           '--n_jobs', '4']",
    "    print('running:', ' '.join(cmd))",
    "    print('(this can take several minutes)')",
    "    result = subprocess.run(cmd, capture_output=True, text=True)",
    "    print(result.stdout[-2000:])",
    "    if result.returncode != 0:",
    "        print('STDERR:', result.stderr[-2000:])",
    "else:",
    "    print(f'[skip] {kpgt_preprocess} not found. Did you run scripts/setup.py?')",
)

code(
    "# Confirm the three cache files now exist",
    "for fname in ('toxric_multitask_5.pkl', 'rdkfp1-7_512.npz', 'molecular_descriptors.npz'):",
    "    p = OUT_DIR / fname",
    "    if p.exists():",
    "        print(f'  [ok] {fname:<35} ({p.stat().st_size/1e6:.2f} MB)')",
    "    else:",
    "        print(f'  [missing] {fname}')",
)

# -------- Step 4 --------
md(
    "## Step 4 — Featurize one SMILES",
    "",
    "Inspect exactly what the model sees for a single molecule. Uses `src/featurizer.py` which produces the same (graph, fp, md) triple KPGT's preprocess builds at scale.",
    "",
    "Demo molecule: **aspirin** (`CC(=O)OC1=CC=CC=C1C(=O)O`).",
)

code(
    "from toxpkg.featurizer import featurize_smiles",
    "",
    "smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'   # aspirin",
    "result = featurize_smiles(smiles)",
    "if result is None:",
    "    print('SMILES failed to featurize.')",
    "else:",
    "    graph, fp, md = result",
    "    print(f'Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')",
    "    print(f'Graph node features stored under ndata keys: {list(graph.ndata.keys())}')",
    "    print(f'Fingerprint: shape {tuple(fp.shape)}, dtype {fp.dtype}, sum(bits)={int(fp.sum())}')",
    "    print(f'Descriptors: shape {tuple(md.shape)}, dtype {md.dtype}, first 5 values: {md[:5].tolist()}')",
)

# -------- Step 5 --------
md(
    "## Step 5 — Build the fine-tuning model",
    "",
    "Loads KPGT's `LiGhTPredictor` with the base config (`d_g_feats=768, n_mol_layers=12, n_heads=12`), loads the pretrained `base.pth` weights, then wraps it in `KPGTMultiTaskFineTuner` which:",
    "- Sets `backbone.predictor` to a new MLP sized for our `n_tasks` (one per TOXRIC endpoint)",
    "- Deletes the three pretraining aux heads (`md_predictor`, `fp_predictor`, `node_predictor`)",
    "- Routes `forward()` to `LiGhT.forward_tune(g, fp, md)`, which concatenates `[fp_vn, md_vn, atom_readout]` → 3 × 768 = 2304-dim graph representation",
)

code(
    "import torch",
    "from toxpkg.model import build_pretrained_predictor, KPGTMultiTaskFineTuner",
    "from toxpkg.config import KPGT_BASE_CONFIG",
    "",
    "n_tasks = len(endpoint_cols)",
    "print(f'n_tasks = {n_tasks}')",
    "print()",
    "",
    "pretrained_path = ROOT / 'external' / 'KPGT' / 'models' / 'pretrained' / 'base' / 'base.pth'",
    "base = build_pretrained_predictor(pretrained_path=str(pretrained_path) if pretrained_path.exists() else None)",
    "model = KPGTMultiTaskFineTuner(base, n_tasks=n_tasks, head_hidden_dim=256, head_dropout=0.15)",
    "",
    "n_total = sum(p.numel() for p in model.parameters())",
    "n_head  = sum(p.numel() for n, p in model.named_parameters() if 'predictor' in n)",
    "n_back  = n_total - n_head",
    "print(f'Backbone params: {n_back/1e6:.1f}M (frozen-ish, lr=1e-5)')",
    "print(f'Head     params: {n_head/1e6:.2f}M  (lr=1e-3)')",
    "print(f'Total:           {n_total/1e6:.1f}M')",
    "print()",
    "print(model.backbone.predictor)",
)

# -------- Step 6 --------
md(
    "## Step 6 — Training smoke-test (one batch + one optimizer step)",
    "",
    "Builds a DataLoader using KPGT's `MoleculeDataset` + `Collator_tune`, pulls one batch, prints all the tensor shapes, runs one forward + masked-loss + backward + optimizer step. ",
    "Lets you confirm the wiring before committing to a multi-hour training run.",
)

code(
    "from toxpkg.trainer import build_dataloader, masked_loss",
    "from toxpkg.config import TrainConfig",
    "",
    "# Infer task types from CSV column dtypes (all 30 TOXRIC endpoints are binary)",
    "task_types = ['classification'] * len(endpoint_cols)",
    "cfg = TrainConfig(",
    "    data_root='data/kpgt-cache',",
    "    dataset_name='toxric_multitask',",
    "    split_name='random_0',",
    "    pretrained_path=str(pretrained_path),",
    "    n_tasks=len(endpoint_cols),",
    "    task_names=endpoint_cols,",
    "    task_types=task_types,",
    "    batch_size=8,        # small for smoke test",
    "    device='cpu',",
    ")",
    "",
    "train_loader = build_dataloader(cfg, split='train')",
    "print(f'train dataset: {len(train_loader.dataset)} compounds')",
    "print(f'batches at batch_size={cfg.batch_size}: {len(train_loader)}')",
)

code(
    "# Pull ONE batch and inspect every tensor.",
    "smiles_list, g, fp, md, y = next(iter(train_loader))",
    "print('smiles in batch:', len(smiles_list))",
    "print(f'graph (batched): {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')",
    "print(f'fp shape : {tuple(fp.shape)} dtype {fp.dtype}')",
    "print(f'md shape : {tuple(md.shape)} dtype {md.dtype}')",
    "print(f'y shape  : {tuple(y.shape)} dtype {y.dtype}  (NaN entries: {int(y.isnan().sum())})')",
)

code(
    "# Run ONE training step. Watch the loss and confirm gradients flow.",
    "optimizer = torch.optim.AdamW(model.param_groups(cfg.backbone_lr, cfg.head_lr, cfg.weight_decay))",
    "model.train()",
    "",
    "logits = model(g, fp, md)",
    "print(f'logits shape: {tuple(logits.shape)}  (expect ({cfg.batch_size}, {cfg.n_tasks}))')",
    "loss = masked_loss(logits, y, cfg.task_types)",
    "print(f'loss before step: {loss.item():.4f}')",
    "",
    "loss.backward()",
    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)",
    "optimizer.step()",
    "optimizer.zero_grad()",
    "",
    "with torch.no_grad():",
    "    logits2 = model(g, fp, md)",
    "    loss2 = masked_loss(logits2, y, cfg.task_types)",
    "print(f'loss after  step: {loss2.item():.4f}')",
    "print('[ok] forward + masked loss + backward + step all work end-to-end.')",
)

# -------- Step 7 --------
md(
    "## Step 7 — Prediction on a few SMILES",
    "",
    "Runs the SMILES → toxicity scores pipeline using whichever model state you have in memory. ",
    "After a real training run you'd load `checkpoints/best.pt` via `predict.load_finetuned_model(...)`; here we just use the model object from Step 6.",
)

code(
    "from toxpkg.predict import predict_smiles, scores_per_endpoint",
    "",
    "demo = [",
    "    'CC(=O)OC1=CC=CC=C1C(=O)O',        # aspirin",
    "    'CC(=O)Nc1ccc(O)cc1',                # acetaminophen / paracetamol",
    "    'CCO',                                # ethanol",
    "    'C(C(=O)O)N',                         # glycine",
    "]",
    "model.eval()",
    "logits, valid = predict_smiles(model, demo, device='cpu')",
    "scores = scores_per_endpoint(logits, endpoint_cols, task_types)",
    "",
    "for s, ok, sc in zip(demo, valid, scores if valid else []):",
    "    if not ok:",
    "        print(f'\\n=== {s} ===  INVALID')",
    "        continue",
    "    print(f'\\n=== {s} ===')",
    "    top = sorted(sc.items(), key=lambda kv: -kv[1])[:5]",
    "    for name, val in top:",
    "        print(f'  {name:<55} prob={val:.3f}')",
)

# -------- Step 8 --------
md(
    "## Step 8 — Plain-English explanation via Claude",
    "",
    "Sends the SMILES + per-endpoint scores to Claude via your Foundry endpoint and asks for a Markdown-formatted health summary. ",
    "Reuses the same Anthropic + DefaultAzureCredential auth pattern as `labfiles/foundry-chat/python/chat-app/chat-app.py`.",
    "",
    "**Requires** a `.env` with `FOUNDRY_BASE_URL` and `CLAUDE_MODEL` set, plus working Azure credentials.",
)

code(
    "from toxpkg.explainer import explain_predictions",
    "",
    "# Use aspirin's scores (first row from Step 7)",
    "if scores:",
    "    aspirin_scores = scores[0]",
    "    task_type_map = dict(zip(endpoint_cols, task_types))",
    "    try:",
    "        text = explain_predictions(",
    "            smiles=demo[0],",
    "            predictions=aspirin_scores,",
    "            task_types=task_type_map,",
    "        )",
    "        from IPython.display import Markdown, display",
    "        display(Markdown(text))",
    "    except Exception as e:",
    "        print(f'[explainer error] {type(e).__name__}: {e}')",
    "        print('Check .env (FOUNDRY_BASE_URL, CLAUDE_MODEL) and Azure credentials.')",
)

md(
    "## Done",
    "",
    "If everything ran without error: the merged multi-target CSV is at `data/kpgt-cache/toxric_multitask/toxric_multitask.csv`, the KPGT caches are alongside it, the model class loads, a training step works, predictions emerge from any SMILES, and Claude wraps them in plain English.",
    "",
    "Phase D (Azure ML submission) builds on top of this — taking the same `src/trainer.py::train()` function and running it on a GPU compute cluster.",
)


# =========================================================================
# Write notebook
# =========================================================================

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "pipeline.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
