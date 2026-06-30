"""
FastAPI inference service for the fine-tuned KPGT toxicity model.

Loads the model once at startup, exposes:
    GET  /health                 — liveness check
    POST /predict                — SMILES → per-endpoint scores (+ optional explanation)
    POST /explain                — predictions dict → plain English (Claude via Foundry)

Auth: API key via `X-API-Key` header. Set `API_KEY` in the VM's environment.

Run locally:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

# Make src/ importable when running from any cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from toxpkg.explainer import explain_predictions  # noqa: E402
from toxpkg.predict import load_finetuned_model, predict_smiles, scores_per_endpoint  # noqa: E402


# --- App state ---------------------------------------------------------------

load_dotenv()

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best.pt")
KPGT_DIR = os.environ.get("KPGT_DIR", "external/KPGT")
API_KEY = os.environ.get("API_KEY")  # required for non-localhost calls
SERVE_DEVICE = os.environ.get("SERVE_DEVICE", "cpu")

app = FastAPI(
    title="KPGT-TOXRIC inference",
    description="Multi-task molecular toxicity predictions via fine-tuned KPGT.",
    version="1.0",
)

# Loaded once at startup, used by every request.
MODEL = None
CFG: dict = {}
TASK_TYPE_MAP: dict = {}


@app.on_event("startup")
def _load_model() -> None:
    global MODEL, CFG, TASK_TYPE_MAP
    if not Path(MODEL_PATH).exists():
        # Don't crash — let /health surface this. Inference endpoints will 503.
        print(f"[warn] checkpoint not found at {MODEL_PATH} — endpoints will return 503")
        return
    print(f"[startup] loading {MODEL_PATH} on {SERVE_DEVICE}...")
    MODEL, CFG = load_finetuned_model(MODEL_PATH, kpgt_dir=KPGT_DIR, device=SERVE_DEVICE)
    TASK_TYPE_MAP = dict(zip(CFG["task_names"], CFG["task_types"]))
    print(f"[startup] model ready — {CFG['n_tasks']} tasks")


# --- Auth --------------------------------------------------------------------

def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if API_KEY is None:
        return  # auth disabled (only safe on a locked-down dev box)
    if x_api_key != API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


# --- Schemas -----------------------------------------------------------------

class PredictRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, max_length=64,
                              description="One or more SMILES strings.")
    explain: bool = Field(False, description="Also generate a Claude plain-English summary.")
    top_k: int = Field(10, ge=1, le=50, description="How many top-scoring endpoints to return.")


class PerMoleculeResult(BaseModel):
    smiles: str
    valid: bool
    scores: dict[str, float] = {}
    top: list[dict[str, float | str]] = []
    explanation: Optional[str] = None


class PredictResponse(BaseModel):
    n_tasks: int
    results: list[PerMoleculeResult]


class ExplainRequest(BaseModel):
    smiles: str
    predictions: dict[str, float]


class ExplainResponse(BaseModel):
    explanation: str


# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "device": SERVE_DEVICE,
        "n_tasks": CFG.get("n_tasks", 0),
    }


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"Model not loaded (checkpoint at {MODEL_PATH} missing?).")

    logits, valid = predict_smiles(MODEL, req.smiles, kpgt_dir=KPGT_DIR, device=SERVE_DEVICE)
    score_dicts = scores_per_endpoint(logits, CFG["task_names"], CFG["task_types"])
    score_iter = iter(score_dicts)

    results: list[PerMoleculeResult] = []
    for s, ok in zip(req.smiles, valid):
        if not ok:
            results.append(PerMoleculeResult(smiles=s, valid=False))
            continue
        scores = next(score_iter)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: req.top_k]
        top = [{"endpoint": n, "score": float(v),
                "kind": TASK_TYPE_MAP.get(n, "classification")} for n, v in ranked]
        explanation = None
        if req.explain:
            try:
                explanation = explain_predictions(s, scores, task_types=TASK_TYPE_MAP)
            except Exception as e:
                explanation = f"[explainer error] {type(e).__name__}: {e}"
        results.append(PerMoleculeResult(
            smiles=s, valid=True, scores=scores, top=top, explanation=explanation,
        ))

    return PredictResponse(n_tasks=CFG["n_tasks"], results=results)


@app.post("/explain", response_model=ExplainResponse,
          dependencies=[Depends(require_api_key)])
def explain(req: ExplainRequest) -> ExplainResponse:
    text = explain_predictions(req.smiles, req.predictions, task_types=TASK_TYPE_MAP)
    return ExplainResponse(explanation=text)
