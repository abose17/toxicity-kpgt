"""
ToxNav — Drug Toxicity & Safer Alternatives
Streamlit UI for the TOXRIC + KPGT pipeline.

Run:
    cd labfiles/toxicity-kpgt
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toxpkg.rate_limiter import (
    EXPIRY_DATE, is_expired, days_remaining,
    check_and_increment, real_mode_remaining_today, DAILY_REAL_MODE_LIMIT,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToxNav",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Service expiry gate ───────────────────────────────────────────────────────
if is_expired():
    st.title("ToxNav — Service Ended")
    st.markdown(
        f"""
This demo was available for **60 days** (until **{EXPIRY_DATE.strftime('%B %d, %Y')}**)
as a personal research project. The hosted service is no longer active.

**To run ToxNav locally:**
```bash
git clone https://github.com/abose17/mslearn-ai-studio
cd mslearn-ai-studio/labfiles/toxicity-kpgt
pip install -r requirements.txt
streamlit run app.py
```

Source code and model weights are available in the repository.
        """
    )
    st.stop()

# ── Endpoint names (read once from CSV if available) ─────────────────────────
_MERGED_CSV = Path("data/toxric_merged.csv")
_ENDPOINT_NAMES: list[str] = []
if _MERGED_CSV.exists():
    try:
        _ENDPOINT_NAMES = [
            c for c in pd.read_csv(_MERGED_CSV, nrows=0).columns if c != "smiles"
        ]
    except Exception:
        pass

if not _ENDPOINT_NAMES:
    # Fallback hardcoded names (matches TOXRIC 30-dataset structure)
    _ENDPOINT_NAMES = [
        "CYP450_CYP1A2", "CYP450_CYP2C19", "CYP450_CYP2C9", "CYP450_CYP2D6",
        "CYP450_CYP3A4", "Carcinogenicity_Carcinogenicity",
        "Cardiotoxicity_Cardiotoxicity-1", "Cardiotoxicity_Cardiotoxicity-10",
        "Cardiotoxicity_Cardiotoxicity-30", "Cardiotoxicity_Cardiotoxicity-5",
        "Clinical_Toxicity_Clinical_toxicity",
        "Developmental_and_Reproductive_Toxicity_Developmental_Toxicity",
        "Developmental_and_Reproductive_Toxicity_Reproductive_Toxicity",
        "Endocrine_Disruption_NR-AR-LBD", "Endocrine_Disruption_NR-AR",
        "Endocrine_Disruption_NR-AhR", "Endocrine_Disruption_NR-ER-LBD",
        "Endocrine_Disruption_NR-ER", "Endocrine_Disruption_NR-PPAR-gamma",
        "Endocrine_Disruption_NR-aromatase", "Endocrine_Disruption_SR-ARE",
        "Endocrine_Disruption_SR-ATAD5", "Endocrine_Disruption_SR-HSE",
        "Endocrine_Disruption_SR-MMP", "Endocrine_Disruption_SR-p53",
        "Hepatotoxicity_Hepatotoxicity", "Irritation_and_Corrosion_Eye_Corrosion",
        "Irritation_and_Corrosion_Eye_Irritation", "Mutagenicity_Ames_Mutagenicity",
        "Respiratory_Toxicity_Respiratory_Toxicity",
    ]


# ── Demo data ─────────────────────────────────────────────────────────────────

_DEMO_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"   # Aspirin

_DEMO_MOLECULES = [
    {
        "role": "original",
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "similarity": None,
        "chembl_id": "CHEMBL25",
        "toxric_match": False,
        "source": "kpgt",
        # Higher toxicity — known CYP450 inhibitor, GI irritant
        "raw_scores": {
            "CYP450_CYP1A2": 0.72, "CYP450_CYP2C19": 0.68, "CYP450_CYP2C9": 0.81,
            "CYP450_CYP2D6": 0.44, "CYP450_CYP3A4": 0.38,
            "Carcinogenicity_Carcinogenicity": 0.22,
            "Cardiotoxicity_Cardiotoxicity-1": 0.55, "Cardiotoxicity_Cardiotoxicity-10": 0.48,
            "Cardiotoxicity_Cardiotoxicity-30": 0.41, "Cardiotoxicity_Cardiotoxicity-5": 0.52,
            "Clinical_Toxicity_Clinical_toxicity": 0.61,
            "Developmental_and_Reproductive_Toxicity_Developmental_Toxicity": 0.34,
            "Developmental_and_Reproductive_Toxicity_Reproductive_Toxicity": 0.29,
            "Endocrine_Disruption_NR-AR-LBD": 0.18, "Endocrine_Disruption_NR-AR": 0.21,
            "Endocrine_Disruption_NR-AhR": 0.35, "Endocrine_Disruption_NR-ER-LBD": 0.19,
            "Endocrine_Disruption_NR-ER": 0.22, "Endocrine_Disruption_NR-PPAR-gamma": 0.28,
            "Endocrine_Disruption_NR-aromatase": 0.31, "Endocrine_Disruption_SR-ARE": 0.45,
            "Endocrine_Disruption_SR-ATAD5": 0.27, "Endocrine_Disruption_SR-HSE": 0.33,
            "Endocrine_Disruption_SR-MMP": 0.41, "Endocrine_Disruption_SR-p53": 0.38,
            "Hepatotoxicity_Hepatotoxicity": 0.57,
            "Irritation_and_Corrosion_Eye_Corrosion": 0.44,
            "Irritation_and_Corrosion_Eye_Irritation": 0.51,
            "Mutagenicity_Ames_Mutagenicity": 0.29,
            "Respiratory_Toxicity_Respiratory_Toxicity": 0.36,
        },
    },
    {
        "role": "alternative",
        "name": "Salicylic acid",
        "smiles": "OC(=O)c1ccccc1O",
        "similarity": 0.83,
        "chembl_id": "CHEMBL20",
        "toxric_match": True,
        "source": "toxric",
        "raw_scores": {
            "CYP450_CYP1A2": 0.51, "CYP450_CYP2C19": 0.47, "CYP450_CYP2C9": 0.62,
            "CYP450_CYP2D6": 0.31, "CYP450_CYP3A4": 0.28,
            "Carcinogenicity_Carcinogenicity": 0.15,
            "Cardiotoxicity_Cardiotoxicity-1": 0.38, "Cardiotoxicity_Cardiotoxicity-10": 0.33,
            "Cardiotoxicity_Cardiotoxicity-30": 0.29, "Cardiotoxicity_Cardiotoxicity-5": 0.36,
            "Clinical_Toxicity_Clinical_toxicity": 0.44,
            "Developmental_and_Reproductive_Toxicity_Developmental_Toxicity": 0.24,
            "Developmental_and_Reproductive_Toxicity_Reproductive_Toxicity": 0.21,
            "Endocrine_Disruption_NR-AR-LBD": 0.12, "Endocrine_Disruption_NR-AR": 0.15,
            "Endocrine_Disruption_NR-AhR": 0.26, "Endocrine_Disruption_NR-ER-LBD": 0.13,
            "Endocrine_Disruption_NR-ER": 0.16, "Endocrine_Disruption_NR-PPAR-gamma": 0.21,
            "Endocrine_Disruption_NR-aromatase": 0.23, "Endocrine_Disruption_SR-ARE": 0.32,
            "Endocrine_Disruption_SR-ATAD5": 0.19, "Endocrine_Disruption_SR-HSE": 0.24,
            "Endocrine_Disruption_SR-MMP": 0.30, "Endocrine_Disruption_SR-p53": 0.27,
            "Hepatotoxicity_Hepatotoxicity": 0.38,
            "Irritation_and_Corrosion_Eye_Corrosion": 0.29,
            "Irritation_and_Corrosion_Eye_Irritation": 0.35,
            "Mutagenicity_Ames_Mutagenicity": 0.19,
            "Respiratory_Toxicity_Respiratory_Toxicity": 0.24,
        },
    },
    {
        "role": "alternative",
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "similarity": 0.74,
        "chembl_id": "CHEMBL521",
        "toxric_match": False,
        "source": "kpgt",
        "raw_scores": {
            "CYP450_CYP1A2": 0.43, "CYP450_CYP2C19": 0.39, "CYP450_CYP2C9": 0.58,
            "CYP450_CYP2D6": 0.22, "CYP450_CYP3A4": 0.31,
            "Carcinogenicity_Carcinogenicity": 0.11,
            "Cardiotoxicity_Cardiotoxicity-1": 0.29, "Cardiotoxicity_Cardiotoxicity-10": 0.25,
            "Cardiotoxicity_Cardiotoxicity-30": 0.22, "Cardiotoxicity_Cardiotoxicity-5": 0.27,
            "Clinical_Toxicity_Clinical_toxicity": 0.38,
            "Developmental_and_Reproductive_Toxicity_Developmental_Toxicity": 0.19,
            "Developmental_and_Reproductive_Toxicity_Reproductive_Toxicity": 0.17,
            "Endocrine_Disruption_NR-AR-LBD": 0.09, "Endocrine_Disruption_NR-AR": 0.11,
            "Endocrine_Disruption_NR-AhR": 0.21, "Endocrine_Disruption_NR-ER-LBD": 0.10,
            "Endocrine_Disruption_NR-ER": 0.13, "Endocrine_Disruption_NR-PPAR-gamma": 0.17,
            "Endocrine_Disruption_NR-aromatase": 0.18, "Endocrine_Disruption_SR-ARE": 0.25,
            "Endocrine_Disruption_SR-ATAD5": 0.14, "Endocrine_Disruption_SR-HSE": 0.19,
            "Endocrine_Disruption_SR-MMP": 0.23, "Endocrine_Disruption_SR-p53": 0.21,
            "Hepatotoxicity_Hepatotoxicity": 0.28,
            "Irritation_and_Corrosion_Eye_Corrosion": 0.22,
            "Irritation_and_Corrosion_Eye_Irritation": 0.27,
            "Mutagenicity_Ames_Mutagenicity": 0.13,
            "Respiratory_Toxicity_Respiratory_Toxicity": 0.18,
        },
    },
    {
        "role": "alternative",
        "name": "Naproxen",
        "smiles": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "similarity": 0.71,
        "chembl_id": "CHEMBL154",
        "toxric_match": False,
        "source": "kpgt",
        "raw_scores": {
            "CYP450_CYP1A2": 0.31, "CYP450_CYP2C19": 0.28, "CYP450_CYP2C9": 0.44,
            "CYP450_CYP2D6": 0.17, "CYP450_CYP3A4": 0.22,
            "Carcinogenicity_Carcinogenicity": 0.09,
            "Cardiotoxicity_Cardiotoxicity-1": 0.21, "Cardiotoxicity_Cardiotoxicity-10": 0.18,
            "Cardiotoxicity_Cardiotoxicity-30": 0.15, "Cardiotoxicity_Cardiotoxicity-5": 0.19,
            "Clinical_Toxicity_Clinical_toxicity": 0.27,
            "Developmental_and_Reproductive_Toxicity_Developmental_Toxicity": 0.14,
            "Developmental_and_Reproductive_Toxicity_Reproductive_Toxicity": 0.12,
            "Endocrine_Disruption_NR-AR-LBD": 0.07, "Endocrine_Disruption_NR-AR": 0.08,
            "Endocrine_Disruption_NR-AhR": 0.16, "Endocrine_Disruption_NR-ER-LBD": 0.08,
            "Endocrine_Disruption_NR-ER": 0.10, "Endocrine_Disruption_NR-PPAR-gamma": 0.13,
            "Endocrine_Disruption_NR-aromatase": 0.14, "Endocrine_Disruption_SR-ARE": 0.19,
            "Endocrine_Disruption_SR-ATAD5": 0.11, "Endocrine_Disruption_SR-HSE": 0.14,
            "Endocrine_Disruption_SR-MMP": 0.17, "Endocrine_Disruption_SR-p53": 0.16,
            "Hepatotoxicity_Hepatotoxicity": 0.21,
            "Irritation_and_Corrosion_Eye_Corrosion": 0.16,
            "Irritation_and_Corrosion_Eye_Irritation": 0.20,
            "Mutagenicity_Ames_Mutagenicity": 0.09,
            "Respiratory_Toxicity_Respiratory_Toxicity": 0.13,
        },
    },
]

_DEMO_ITERATIONS = [
    {
        "iteration": 1,
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "name": "Aspirin",
        "confidence": 4,
        "reasoning": (
            "KPGT predicts moderate CYP2C9 inhibition and hepatotoxicity, but underestimates "
            "known GI irritation and platelet inhibition effects of Aspirin."
        ),
        "suggestions": ["Salicylamide", "Methyl salicylate", "Diflunisal", "Salsalate", "Choline salicylate"],
        "best_candidate": "Salicylic acid",
        "best_similarity": 0.83,
        "path": "KPGT (agentic)",
    },
    {
        "iteration": 2,
        "smiles": "OC(=O)c1ccccc1O",
        "name": "Salicylic acid",
        "confidence": 8,
        "reasoning": (
            "Salicylic acid predictions align well with its known pharmacology — "
            "CYP2C9 inhibition and mild hepatotoxicity are consistent with literature."
        ),
        "suggestions": [],
        "best_candidate": None,
        "best_similarity": None,
        "path": "TOXRIC match",
    },
]

_DEMO_EXPLANATION = """### Compound
**Aspirin** (acetylsalicylic acid) — an NSAID and antiplatelet agent widely used for pain, fever, and cardiovascular protection.

### High-risk endpoints (probability ≥ 0.5)
- **CYP2C9 inhibition (0.81):** Aspirin and its metabolite salicylate inhibit CYP2C9, reducing metabolism of warfarin and other drugs. This is a well-known drug interaction risk.
- **CYP1A2 inhibition (0.72):** Moderate inhibition may affect clearance of theophylline and caffeine.
- **Clinical toxicity (0.61):** Consistent with known GI irritation, Reye's syndrome risk in children, and bleeding risk at therapeutic doses.

### Low-risk endpoints (probability < 0.5)
No predicted carcinogenicity, genotoxicity, or significant endocrine disruption. Hepatotoxicity is elevated (0.57) but below the threshold — consistent with rare DILI reports at high doses.

### Overall assessment
Aspirin carries moderate CYP450-mediated drug interaction risk and mild hepatotoxic potential. Its risk profile is well-characterized and manageable in most clinical settings.

### Caveat
These are computational predictions from a fine-tuned KPGT model on TOXRIC data — not clinical findings — and any decision-making should rely on actual toxicology studies."""


def _build_demo_results(tmp_dir: str) -> dict:
    """Generate demo results including PNG files."""
    from toxpkg.visualizer import draw_molecule_grid, draw_mcs_highlight

    mols = _DEMO_MOLECULES
    scores_list = [m["raw_scores"] for m in mols]

    # Build comparison DataFrame
    rows = []
    for m, sc in zip(mols, scores_list):
        row = {
            "role": m["role"],
            "name": m["name"],
            "smiles": m["smiles"],
            "similarity": m["similarity"],
            "chembl_id": m["chembl_id"],
            "toxric_match": m["toxric_match"],
            "source": m["source"],
            "toxicity_score": round(sum(sc.values()), 3),
        }
        row.update({k: round(v, 3) for k, v in sc.items()})
        rows.append(row)
    df = pd.DataFrame(rows)

    smiles_list = [m["smiles"] for m in mols]
    names = [m["name"] for m in mols]
    labels = ["ORIGINAL" if m["role"] == "original" else f"Alt {i}"
              for i, m in enumerate(mols)]
    tox_scores = [r["toxicity_score"] for r in rows]

    grid_path = os.path.join(tmp_dir, "molecule_grid.png")
    draw_molecule_grid(smiles_list, labels, tox_scores, grid_path, names=names)

    mcs_path = os.path.join(tmp_dir, "mcs_highlight.png")
    draw_mcs_highlight(
        mols[0]["smiles"], mols[-1]["smiles"], mcs_path,
        name_query=mols[0]["name"], name_best=mols[-1]["name"],
    )

    return {
        "status": "satisfactory",
        "source": "kpgt+agentic",
        "original_smiles": _DEMO_SMILES,
        "original_name": "Aspirin",
        "final_name": "Naproxen",
        "comparison_df": df,
        "grid_path": grid_path,
        "mcs_path": mcs_path,
        "explanation": _DEMO_EXPLANATION,
        "iterations": _DEMO_ITERATIONS,
        "demo": True,
    }


# ── Real pipeline ─────────────────────────────────────────────────────────────

def _run_real_pipeline(smiles: str, checkpoint: str, pretrained: str,
                       threshold: float, max_iter: int,
                       sim_threshold: float, tmp_dir: str) -> dict:
    """Run the actual pipeline and return results in the same shape as demo."""
    from toxpkg.agentic_pipeline import run_agentic_pipeline
    from toxpkg.model import build_pretrained_predictor
    from toxpkg.predict import load_finetuned_model
    from toxpkg.similarity import fetch_similar_chembl, lookup_name_pubchem
    from toxpkg.toxric_matcher import filter_by_toxric
    from toxpkg.comparator import rank_by_toxicity
    from toxpkg.predict import predict_smiles, scores_per_endpoint
    from toxpkg.visualizer import draw_molecule_grid, draw_mcs_highlight

    original_name = lookup_name_pubchem(smiles)

    # Step 1: ChEMBL similarity search
    candidates = fetch_similar_chembl(smiles, threshold=sim_threshold, n=20)
    if not candidates:
        st.warning("ChEMBL returned no results. Try lowering the similarity threshold.")
        return {}

    # Step 2: TOXRIC filter
    filtered, is_fallback = filter_by_toxric(candidates, merged_csv_path=str(_MERGED_CSV))

    # Step 3: Load model
    model, cfg = load_finetuned_model(checkpoint)
    backbone = build_pretrained_predictor(pretrained_path=pretrained)
    backbone.eval()

    task_names = cfg["task_names"]
    task_types = cfg["task_types"]

    # Step 4: Agentic prediction for each candidate
    all_records = [{"smiles": smiles, "role": "original", "name": original_name,
                    "similarity": None, "chembl_id": "", "source": "input"}]
    all_scores_list = []

    from toxpkg.predict import predict_smiles, scores_per_endpoint
    orig_logits, orig_valid = predict_smiles(model, [smiles])
    if orig_logits.shape[0] > 0:
        all_scores_list.append(scores_per_endpoint(orig_logits, task_names, task_types)[0])

    for cand in filtered[:5]:
        result = run_agentic_pipeline(
            smiles=cand["smiles"], model=model, backbone=backbone,
            cfg=cfg, satisfactory_threshold=threshold, max_iter=max_iter,
        )
        all_records.append({**cand, "role": "alternative",
                             "name": result.get("final_name", ""),
                             "source": result.get("source", "kpgt")})
        all_scores_list.append(result["scores"])

    df = rank_by_toxicity(all_records, all_scores_list, is_fallback=is_fallback)

    smiles_col = list(df["smiles"])
    names_col = list(df.get("name", [""] * len(df)))
    labels = ["ORIGINAL" if r == "original" else f"Alt {i}"
              for i, r in enumerate(df["role"])]
    tox_scores = list(df["toxicity_score"])

    grid_path = os.path.join(tmp_dir, "molecule_grid.png")
    draw_molecule_grid(smiles_col, labels, tox_scores, grid_path, names=names_col)

    alts = df[df["role"] == "alternative"]
    mcs_path = None
    if not alts.empty:
        best_smiles = alts.iloc[0]["smiles"]
        best_name = str(alts.iloc[0].get("name", ""))
        mcs_path = os.path.join(tmp_dir, "mcs_highlight.png")
        draw_mcs_highlight(smiles, best_smiles, mcs_path,
                           name_query=original_name, name_best=best_name)

    return {
        "status": "complete",
        "source": "pipeline",
        "original_smiles": smiles,
        "original_name": original_name,
        "final_name": alts.iloc[0].get("name", "") if not alts.empty else "",
        "comparison_df": df,
        "grid_path": grid_path,
        "mcs_path": mcs_path,
        "explanation": "",
        "iterations": [],
        "demo": False,
    }



# ── Display helpers ───────────────────────────────────────────────────────────

def _status_badge(status: str, source: str) -> None:
    colour = {"toxric_match": "green", "satisfactory": "blue",
              "degrading": "orange", "max_iterations": "red",
              "complete": "blue", "satisfactory+toxric": "green"}.get(status, "grey")
    source_label = {"toxric": "TOXRIC ground truth", "kpgt": "KPGT prediction",
                    "kpgt+agentic": "KPGT + agentic validation",
                    "pipeline": "Full pipeline"}.get(source, source)
    st.markdown(
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:0.85em">{status.replace("_"," ").upper()}</span>'
        f'&nbsp;&nbsp;<span style="color:grey;font-size:0.85em">Source: {source_label}</span>',
        unsafe_allow_html=True,
    )


def _show_comparison_table(df: pd.DataFrame) -> None:
    st.subheader("Comparison Table")

    meta_cols = ["role", "name", "smiles", "similarity", "chembl_id",
                 "toxric_match", "source", "toxicity_score"]
    display_cols = [c for c in meta_cols if c in df.columns]
    endpoint_cols = [c for c in df.columns if c not in set(meta_cols)]

    summary = df[display_cols].copy()
    if "toxicity_score" in summary.columns:
        summary = summary.sort_values("toxicity_score")

    def _highlight(row):
        if row.get("role") == "original":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        summary.style.apply(_highlight, axis=1).format(
            {"toxicity_score": "{:.3f}", "similarity": lambda v: f"{v:.2f}" if v else "—"}
        ),
        width="stretch",
    )

    with st.expander("Full endpoint scores"):
        ep_display = df[["name", "toxicity_score"] + endpoint_cols[:15]].copy()
        st.dataframe(ep_display.style.background_gradient(
            subset=endpoint_cols[:15], cmap="RdYlGn_r", vmin=0, vmax=1
        ), width="stretch")


def _show_visuals(grid_path, mcs_path) -> None:
    st.subheader("Molecular Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Molecule Grid** — all candidates ranked by toxicity score")
        if grid_path and Path(grid_path).exists():
            st.image(grid_path, width="stretch")
        else:
            st.info("Molecule grid not available.")
    with col2:
        st.markdown("**MCS Highlight** — shared scaffold between original and safest alternative")
        if mcs_path and Path(mcs_path).exists():
            st.image(mcs_path, width="stretch")
        else:
            st.info("MCS highlight not available.")


def _show_explanation(explanation: str) -> None:
    if not explanation:
        return
    with st.expander("Plain-English Explanation (Claude)", expanded=True):
        st.markdown(explanation)


def _show_iteration_trace(iterations: list[dict]) -> None:
    if not iterations:
        return
    with st.expander(f"Agentic Iteration Trace ({len(iterations)} iteration(s))", expanded=False):
        for it in iterations:
            conf = it["confidence"]
            colour = "#d4edda" if conf >= 6 else "#fff3cd" if conf >= 4 else "#f8d7da"
            st.markdown(
                f'<div style="border-left:4px solid #6c757d;padding:8px 12px;'
                f'background:{colour};margin-bottom:8px;border-radius:4px">'
                f'<b>Iteration {it["iteration"]}</b> — '
                f'<code>{it.get("name") or it["smiles"][:40]}</code>'
                f'&nbsp;&nbsp;Confidence: <b>{conf}/10</b>'
                f'<br><small>{it["reasoning"]}</small>'
                + (
                    f'<br><small>Suggestions: {", ".join(it["suggestions"])}</small>'
                    if it.get("suggestions") else ""
                )
                + (
                    f'<br><small>Best candidate: <b>{it["best_candidate"]}</b> '
                    f'(similarity={it["best_similarity"]:.3f})</small>'
                    if it.get("best_candidate") else ""
                )
                + f'</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 ToxNav")
    st.caption("Drug Toxicity & Safer Alternatives")
    st.divider()

    demo_mode = st.checkbox("Demo Mode", value=True,
                            help="Use pre-loaded Aspirin example. "
                                 "Uncheck to run the real pipeline (requires trained checkpoint).")

    smiles_input = st.text_area(
        "Input SMILES",
        value=_DEMO_SMILES if demo_mode else "",
        height=100,
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
        help="Paste a canonical SMILES string for your drug molecule.",
    )

    st.divider()
    st.subheader("Settings")
    if demo_mode:
        st.caption("Settings apply to real pipeline only — not used in demo mode.")
    sim_threshold = st.slider("ChEMBL similarity threshold", 0.4, 0.9, 0.7, 0.05,
                               disabled=demo_mode,
                               help="Tanimoto similarity cutoff for ChEMBL search.")
    conf_threshold = st.slider("LLM confidence threshold", 1, 10, 6, 1,
                                disabled=demo_mode,
                                help="Min. LLM confidence score to accept a KPGT prediction.")
    max_iter = st.number_input("Max agentic iterations", 1, 5, 3, 1,
                                disabled=demo_mode,
                                help="Hard cap on self-correction iterations.")

    if not demo_mode:
        st.divider()
        checkpoint = st.text_input("Checkpoint path", value="checkpoints/best.pt")
        pretrained = st.text_input("Pretrained backbone path",
                                   value="external/KPGT/models/pretrained/base/base.pth")

    st.divider()
    run_btn = st.button("Run Pipeline", type="primary")

    if demo_mode:
        st.info("Demo mode: showing Aspirin example with realistic dummy scores.")
    else:
        ckpt_exists = Path(checkpoint if not demo_mode else "").exists()
        if not ckpt_exists:
            st.warning("Checkpoint not found. Train a model first (Phase D).")
        remaining = real_mode_remaining_today()
        if remaining > 0:
            st.caption(f"Real-mode runs remaining today: **{remaining} / {DAILY_REAL_MODE_LIMIT}**")
        else:
            st.error("Daily limit reached. Real-mode runs reset at midnight UTC.")

    d = days_remaining()
    if d <= 7:
        st.sidebar.warning(f"This service ends in **{d} day(s)** ({EXPIRY_DATE.strftime('%b %d, %Y')}).")
    else:
        st.sidebar.caption(f"Service available until {EXPIRY_DATE.strftime('%b %d, %Y')} ({d} days).")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("ToxNav — Drug Toxicity & Safer Alternatives")
st.caption(
    "Enter a drug SMILES to find structurally similar molecules with lower predicted toxicity. "
    "Uses TOXRIC ground-truth labels when available; falls back to KPGT predictions "
    "with agentic self-correction when not."
)

if run_btn:
    if not smiles_input.strip():
        st.error("Please enter a SMILES string.")
        st.stop()

    tmp_dir = tempfile.mkdtemp(prefix="toxnav_")

    if demo_mode:
        with st.spinner("Generating demo results…"):
            try:
                results = _build_demo_results(tmp_dir)
                st.session_state["results"] = results
                st.session_state["tmp_dir"] = tmp_dir
            except Exception as e:
                st.error(f"Demo generation failed: {e}")
                st.stop()
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            st.error(f"Checkpoint not found: `{checkpoint}`. Train a model first (Phase D / AML).")
            st.stop()
        allowed, remaining_after = check_and_increment()
        if not allowed:
            st.error(
                f"Daily limit of {DAILY_REAL_MODE_LIMIT} real-mode runs reached. "
                "Resets at midnight UTC — or try Demo Mode."
            )
            st.stop()
        with st.spinner("Running pipeline — this may take a few minutes…"):
            try:
                results = _run_real_pipeline(
                    smiles_input.strip(), str(ckpt_path), pretrained,
                    conf_threshold, max_iter, sim_threshold, tmp_dir,
                )
                st.session_state["results"] = results
                st.session_state["tmp_dir"] = tmp_dir
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

# ── Results ───────────────────────────────────────────────────────────────────

if "results" in st.session_state:
    r = st.session_state["results"]
    if not r:
        st.stop()

    if r.get("demo"):
        st.info("Showing demo results for **Aspirin** (CC(=O)OC1=CC=CC=C1C(=O)O)")

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Molecule", r.get("original_name") or "Unknown")
    with col2:
        df = r.get("comparison_df", pd.DataFrame())
        alts = df[df["role"] == "alternative"] if not df.empty else pd.DataFrame()
        st.metric("Alternatives Found", len(alts))
    with col3:
        best_name = alts.iloc[0].get("name", "—") if not alts.empty else "—"
        st.metric("Safest Alternative", best_name)
    with col4:
        if not alts.empty and "toxicity_score" in alts.columns:
            orig_score = df[df["role"] == "original"]["toxicity_score"].values[0]
            best_score = alts.iloc[0]["toxicity_score"]
            reduction = round(((orig_score - best_score) / orig_score) * 100, 1)
            st.metric("Toxicity Reduction", f"{reduction}%",
                      delta=f"-{reduction}%", delta_color="inverse")

    _status_badge(r.get("status", ""), r.get("source", ""))
    st.divider()

    _show_comparison_table(df)
    st.divider()

    _show_visuals(r.get("grid_path"), r.get("mcs_path"))
    st.divider()

    _show_explanation(r.get("explanation", ""))
    _show_iteration_trace(r.get("iterations", []))

else:
    st.info("Enter a SMILES string in the sidebar and click **Run Pipeline** to start.")
    st.markdown("""
**How it works:**

1. **ChEMBL search** — finds up to 20 structurally similar molecules
2. **TOXRIC match** — molecules found in TOXRIC use ground-truth binary labels
3. **KPGT prediction** — unmatched molecules are scored by the fine-tuned GNN
4. **Agentic validation** — LLM checks if predictions align with known drug knowledge;
   self-corrects by suggesting better proxy molecules if confidence is low
5. **Ranking** — candidates sorted by total toxicity score (sum of sigmoid probabilities)
6. **Explanation** — Claude generates a plain-English health risk summary
""")
