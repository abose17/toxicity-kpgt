"""
LLM-based validation and suggestion for the agentic prediction pipeline.

LLM Inference 1 — validate_prediction:
    Scores KPGT output against the LLM's parametric knowledge of the drug.
    Returns a confidence score (0–10).

LLM Inference 2 — suggest_similar_drugs:
    Proposes 5 drugs with a similar known toxicity profile when predictions
    are unreliable.

Combined structural similarity — compute_combined_similarity:
    Combines all three KPGT feature types:
      (1/3) fp_tanimoto + (1/3) md_cosine + (1/3) graph_embed_cosine
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from pathlib import Path

import requests
import torch

from .explainer import build_azure_openai_client
from .featurizer import featurize_smiles


# ── System prompts ────────────────────────────────────────────────────────────

_VALIDATE_SYSTEM = """You are a computational toxicologist with deep knowledge of drug pharmacology.

You will receive a drug molecule (SMILES + name if known) and its predicted toxicity scores
across multiple endpoints from a fine-tuned graph neural network trained on TOXRIC data.

Assess whether these predictions are chemically and pharmacologically plausible based on
your knowledge of this molecule's known properties, structural class, and mechanism of action.

Respond with ONLY valid JSON in this exact format:
{
  "confidence": <integer 0-10>,
  "reasoning": "<1-2 sentences explaining your assessment>"
}

Confidence scale:
  0-3:  Predictions contradict well-known properties of this molecule
  4-5:  Insufficient knowledge to assess confidently
  6-8:  Predictions are plausible and consistent with known properties
  9-10: Predictions strongly match well-established toxicity data

Do not add any text outside the JSON."""


_SUGGEST_SYSTEM = """You are a computational toxicologist and medicinal chemist.

You will receive a drug molecule (SMILES + name) whose predicted toxicity scores may be
unreliable. Suggest exactly 5 drugs known to have a SIMILAR toxicity profile across the
same endpoints, based on your pharmacological knowledge.

Choose well-studied compounds with known clinical toxicology data.

Respond with ONLY valid JSON in this exact format:
[
  {"name": "<common drug name>", "smiles": "<canonical SMILES>"},
  {"name": "<common drug name>", "smiles": "<canonical SMILES>"},
  {"name": "<common drug name>", "smiles": "<canonical SMILES>"},
  {"name": "<common drug name>", "smiles": "<canonical SMILES>"},
  {"name": "<common drug name>", "smiles": "<canonical SMILES>"}
]

Only include SMILES you are confident are correct. Do not add any text outside the JSON."""


# ── PubChem fallback ──────────────────────────────────────────────────────────

_PUBCHEM_SMILES_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    "/{name}/property/CanonicalSMILES/JSON"
)


def _pubchem_smiles_by_name(name: str, timeout: int = 10) -> str:
    """Resolve a drug name to canonical SMILES via PubChem. Returns '' on failure."""
    try:
        url = _PUBCHEM_SMILES_URL.format(name=urllib.parse.quote(name, safe=""))
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return ""
        props = resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
        return props.get("CanonicalSMILES", "")
    except Exception:
        return ""


# ── LLM Inference 1 ───────────────────────────────────────────────────────────

def validate_prediction(
    smiles: str,
    name: str,
    scores: dict[str, float],
    llm_client=None,
    claude_client=None,     # backward-compat alias
    model: str | None = None,
) -> dict:
    """Assess whether KPGT predictions are plausible for this molecule.

    Returns:
        {"confidence": int 0-10, "reasoning": str}
    """
    client = llm_client or claude_client or build_azure_openai_client()
    if model is None:
        model = os.getenv("MODEL_DEPLOYMENT", "gpt-4o")

    name_line = f"Name: {name}\n" if name else ""
    score_lines = "\n".join(
        f"- {ep}: {v:.3f}" for ep, v in sorted(scores.items(), key=lambda kv: -kv[1])
    )
    user_msg = f"SMILES: {smiles}\n{name_line}\nPredicted toxicity scores:\n{score_lines}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": _VALIDATE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        result = json.loads(raw)
        return {
            "confidence": max(0, min(10, int(result.get("confidence", 5)))),
            "reasoning": str(result.get("reasoning", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return {"confidence": 5, "reasoning": raw[:300]}


# ── LLM Inference 2 ───────────────────────────────────────────────────────────

def suggest_similar_drugs(
    smiles: str,
    name: str,
    scores: dict[str, float],
    llm_client=None,
    claude_client=None,     # backward-compat alias
    model: str | None = None,
) -> list[dict]:
    """Suggest 5 drugs with a similar known toxicity profile.

    Invalid SMILES from the LLM are resolved via PubChem name lookup.
    Unresolvable entries are dropped.

    Returns:
        List of {name, smiles} dicts (up to 5, all with valid RDKit-parseable SMILES).
    """
    from rdkit import Chem

    client = llm_client or claude_client or build_azure_openai_client()
    if model is None:
        model = os.getenv("MODEL_DEPLOYMENT", "gpt-4o")

    name_line = f"Name: {name}\n" if name else ""
    score_lines = "\n".join(
        f"- {ep}: {v:.3f}" for ep, v in sorted(scores.items(), key=lambda kv: -kv[1])
    )
    user_msg = (
        f"SMILES: {smiles}\n{name_line}\n"
        f"Toxicity scores (KPGT prediction — may be unreliable):\n{score_lines}"
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SUGGEST_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "[]"
    try:
        suggestions = json.loads(raw)
    except json.JSONDecodeError:
        return []

    validated: list[dict] = []
    for s in suggestions[:5]:
        drug_name = str(s.get("name", ""))
        drug_smiles = str(s.get("smiles", ""))

        if drug_smiles and Chem.MolFromSmiles(drug_smiles):
            validated.append({"name": drug_name, "smiles": drug_smiles})
        elif drug_name:
            canonical = _pubchem_smiles_by_name(drug_name)
            if canonical and Chem.MolFromSmiles(canonical):
                validated.append({"name": drug_name, "smiles": canonical})

    return validated


# ── Combined structural similarity ────────────────────────────────────────────

def compute_combined_similarity(
    smiles_a: str,
    smiles_b: str,
    backbone,
    kpgt_dir: str = "external/KPGT",
    device: str = "cpu",
) -> float:
    """Combined structural similarity using all three KPGT feature types.

    combined = (1/3) fp_tanimoto + (1/3) md_cosine + (1/3) graph_embed_cosine

    backbone must be the pretrained LiGhTPredictor (not fine-tuned) so that
    embeddings reflect general molecular structure, not task-specific toxicity.

    Returns float in [0, 1]. Returns 0.0 if either SMILES fails featurization.
    """
    _ensure_kpgt_on_path(kpgt_dir)
    from src.data.collator import Collator_tune

    feat_a = featurize_smiles(smiles_a, kpgt_dir=kpgt_dir)
    feat_b = featurize_smiles(smiles_b, kpgt_dir=kpgt_dir)
    if feat_a is None or feat_b is None:
        return 0.0

    g_a, fp_a, md_a = feat_a
    g_b, fp_b, md_b = feat_b

    # ── 1. Fingerprint Tanimoto (512-dim binary vectors) ────────────────────
    fp_sim = _tanimoto(fp_a, fp_b)

    # ── 2. Descriptor cosine (200-dim float vectors) ────────────────────────
    md_sim = _cosine(md_a, md_b)

    # ── 3. Graph embedding cosine (pretrained KPGT backbone, 2304-dim) ──────
    collator = Collator_tune(max_length=5, n_virtual_nodes=2, add_self_loop=True)
    dummy = torch.zeros(1)

    _, bg_a, fps_a, mds_a, _ = collator([(smiles_a, g_a, fp_a, md_a, dummy)])
    _, bg_b, fps_b, mds_b, _ = collator([(smiles_b, g_b, fp_b, md_b, dummy)])

    backbone.eval()
    with torch.no_grad():
        emb_a = backbone.forward_tune(bg_a.to(device).clone(), fps_a.to(device), mds_a.to(device))
        emb_b = backbone.forward_tune(bg_b.to(device).clone(), fps_b.to(device), mds_b.to(device))

    graph_sim = _cosine(emb_a.squeeze(0).cpu(), emb_b.squeeze(0).cpu())

    return round((fp_sim + md_sim + graph_sim) / 3.0, 4)


def _tanimoto(a: torch.Tensor, b: torch.Tensor) -> float:
    """Tanimoto similarity for binary bit vectors. Range [0, 1]."""
    dot = float((a * b).sum())
    denom = float(a.sum() + b.sum() - dot)
    return dot / denom if denom > 0 else 0.0


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity mapped from [-1, 1] to [0, 1]."""
    norm_a = float(a.norm())
    norm_b = float(b.norm())
    if norm_a == 0 or norm_b == 0:
        return 0.0
    raw = float((a * b).sum()) / (norm_a * norm_b)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _ensure_kpgt_on_path(kpgt_dir: str = "external/KPGT") -> None:
    p = str(Path(kpgt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
