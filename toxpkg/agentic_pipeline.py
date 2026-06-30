"""
Agentic toxicity pipeline — LangGraph implementation.

Two graphs work together:

OUTER GRAPH  (PipelineState — no messages)
  Handles the multi-iteration self-correction loop as explicit nodes + edges.
  This is the same loop that was a hand-written while-loop before.

  START → check_toxric ──(match)──────────────────────────────→ explain → END
               │
           (no match)
               ▼
          predict_kpgt
               ▼
        run_llm_validation   ← delegates to inner graph
               │
        ┌──────┴──────────┐
     (satisfied /      (not satisfied)
      degrading)             ▼
          │             pick_best ──(max_iter / no candidates)──→ explain → END
          ▼                  │
       explain           (continue)
        → END                └──────────────────────────────────→ check_toxric

INNER GRAPH  (MessagesState — mirrors tools-agent-langgraph.py exactly)
  The LLM decides which tools to call:
    • validate_toxicity_prediction  (always called first)
    • suggest_safer_alternatives    (called when confidence < threshold)

  START → llm ──(tool_use?)──→ tools → llm
                (end_turn)
                    END

The LLM calls wired as LangChain/LangGraph @tool functions.
Same interface as the previous hand-written version — scripts/predict_agentic.py
and app.py need no changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from .explainer import build_azure_openai_client, explain_predictions
from .predict import predict_smiles, scores_per_endpoint
from .similarity import lookup_name_pubchem
from .toxric_matcher import filter_by_toxric
from .validator import (
    compute_combined_similarity,
    suggest_similar_drugs as _suggest_similar_drugs,
    validate_prediction as _validate_prediction,
)


# ── Outer graph state ─────────────────────────────────────────────────────────
# Plain TypedDict — no messages field. The inner graph owns its own MessagesState.

class PipelineState(TypedDict):
    original_smiles: str
    current_smiles: str
    current_name: str
    scores: dict
    iteration: int
    prev_confidence: float | None
    confidence: float
    reasoning: str
    suggestions: list[dict]
    status: str          # running | toxric_match | satisfactory | degrading | max_iterations
    source: str          # toxric | kpgt
    iterations: list[dict]
    explanation: str


# ── Inner graph: LLM tool-calling (mirrors tools-agent-langgraph.py) ──────────

_AGENT_SYSTEM = """\
You are analysing a drug molecule's KPGT-predicted toxicity scores.

Follow this exact sequence:
1. ALWAYS call validate_toxicity_prediction first with the provided SMILES, name, and scores.
2. If the confidence returned is below {threshold}, ALSO call suggest_safer_alternatives.
3. Do not call any tool more than once. Stop after completing the sequence.\
"""


def _make_tools(llm_client, model_deployment: str):
    """Build @tool wrappers that capture the AzureOpenAI client via closure.

    Mirrors the @tool pattern from tools-agent-langgraph.py — the decorator
    inspects the function signature and docstring to generate the schema for
    the orchestrating LLM (AzureChatOpenAI) automatically.
    """

    @tool
    def validate_toxicity_prediction(smiles: str, name: str, scores_json: str) -> str:
        """Assess whether KPGT toxicity predictions are plausible for this drug molecule.

        Args:
            smiles:      Canonical SMILES string of the molecule.
            name:        Common drug name (may be empty).
            scores_json: JSON-encoded dict of {endpoint: probability} scores.

        Returns JSON: {"confidence": <int 0-10>, "reasoning": "<str>"}
        """
        scores = json.loads(scores_json)
        result = _validate_prediction(smiles, name, scores,
                                      llm_client=llm_client, model=model_deployment)
        return json.dumps(result)

    @tool
    def suggest_safer_alternatives(smiles: str, name: str, scores_json: str) -> str:
        """Suggest 5 drugs with a similar known toxicity profile when predictions are unreliable.

        Args:
            smiles:      Canonical SMILES string of the molecule.
            name:        Common drug name (may be empty).
            scores_json: JSON-encoded dict of {endpoint: probability} scores.

        Returns JSON: list of {name: str, smiles: str} with up to 5 entries.
        """
        scores = json.loads(scores_json)
        result = _suggest_similar_drugs(smiles, name, scores,
                                        llm_client=llm_client, model=model_deployment)
        return json.dumps(result)

    return [validate_toxicity_prediction, suggest_safer_alternatives]


def _build_inner_graph(llm_with_tools, tool_node: ToolNode):
    """Build the inner LLM tool-calling graph.

    Identical structure to tools-agent-langgraph.py:
      START → llm ──(tool_use?)──→ tools → llm → … → END
    The LLM decides to call validate first, then optionally suggest.
    """
    def llm_node(state: MessagesState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")
    return graph.compile()


def _run_inner_graph(inner_graph, smiles: str, name: str,
                     scores: dict, threshold: float) -> tuple[float, str, list[dict]]:
    """Invoke the inner graph and extract (confidence, reasoning, suggestions)."""
    score_lines = "\n".join(
        f"- {ep}: {v:.3f}" for ep, v in sorted(scores.items(), key=lambda kv: -kv[1])
    )
    name_line = f"Name: {name}\n" if name else ""
    result = inner_graph.invoke({
        "messages": [
            SystemMessage(content=_AGENT_SYSTEM.format(threshold=int(threshold))),
            HumanMessage(content=f"SMILES: {smiles}\n{name_line}\nPredicted scores:\n{score_lines}"),
        ]
    })

    confidence, reasoning, suggestions = 5.0, "", []
    for msg in result["messages"]:
        if hasattr(msg, "name") and hasattr(msg, "content"):
            try:
                content = json.loads(msg.content)
                if msg.name == "validate_toxicity_prediction":
                    confidence = float(content.get("confidence", 5))
                    reasoning = str(content.get("reasoning", ""))
                elif msg.name == "suggest_safer_alternatives":
                    suggestions = content if isinstance(content, list) else []
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
    return confidence, reasoning, suggestions


# ── Outer graph builder ───────────────────────────────────────────────────────

def _build_pipeline_graph(
    model, backbone, cfg: dict,
    inner_graph,
    merged_csv_path: str,
    satisfactory_threshold: float,
    max_iter: int,
    kpgt_dir: str,
    device: str,
    llm_client,
    explain: bool,
) -> any:
    """Build the outer LangGraph state machine.

    Each node is a closure capturing the heavy objects (model, backbone, etc.)
    so they are not serialised into the graph state.
    """
    task_names: list[str] = cfg["task_names"]
    task_types: list[str] = cfg["task_types"]
    task_type_map = dict(zip(task_names, task_types))

    # ── Node 1: TOXRIC check ─────────────────────────────────────────────────
    def check_toxric_node(state: PipelineState) -> PipelineState:
        smiles = state["current_smiles"]
        print(f"\n[graph] iteration {state['iteration'] + 1}/{max_iter}  "
              f"smiles={smiles[:50]}")
        print("  [node] check_toxric")

        matched, is_fallback = filter_by_toxric(
            [{"smiles": smiles, "similarity": 1.0, "chembl_id": ""}],
            merged_csv_path=merged_csv_path,
        )
        if not is_fallback:
            raw = matched[0]["toxric_labels"]
            scores = {
                k: float(v) for k, v in raw.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            }
            print(f"  [TOXRIC match] {len(scores)} endpoints.")
            return {**state, "scores": scores, "status": "toxric_match", "source": "toxric"}

        return {**state, "status": "running"}

    # ── Node 2: KPGT prediction ──────────────────────────────────────────────
    def predict_kpgt_node(state: PipelineState) -> PipelineState:
        print("  [node] predict_kpgt")
        smiles = state["current_smiles"]
        logits, valid = predict_smiles(model, [smiles], kpgt_dir=kpgt_dir, device=device)
        if logits.shape[0] == 0 or not valid[0]:
            print("  [warn] SMILES failed featurization.")
            return {**state, "status": "max_iterations"}
        scores = scores_per_endpoint(logits, task_names, task_types)[0]
        return {**state, "scores": scores}

    # ── Node 3: LLM validation (delegates to inner graph) ────────────────────
    def run_llm_validation_node(state: PipelineState) -> PipelineState:
        print("  [node] run_llm_validation → inner graph (validate + suggest tools)")
        smiles = state["current_smiles"]
        name = state["current_name"]
        scores = state["scores"]
        prev_confidence = state["prev_confidence"]
        iteration = state["iteration"]

        confidence, reasoning, suggestions = _run_inner_graph(
            inner_graph, smiles, name, scores, satisfactory_threshold
        )
        print(f"    confidence={confidence:.0f}/10  {reasoning[:70]}")
        if suggestions:
            print(f"    suggestions: {[s.get('name','?') for s in suggestions]}")

        iter_record = {
            "iteration": iteration + 1,
            "smiles": smiles,
            "name": name,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggestions": [s.get("name", "") for s in suggestions],
            "path": "KPGT + LLM",
        }

        # Determine status for routing
        if confidence >= satisfactory_threshold:
            new_status = "satisfactory"
        elif iteration > 0 and prev_confidence is not None and confidence < prev_confidence:
            new_status = "degrading"
        else:
            new_status = "running"

        return {
            **state,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggestions": suggestions,
            "status": new_status,
            "iterations": state["iterations"] + [iter_record],
        }

    # ── Node 4: Pick best candidate by combined similarity ───────────────────
    def pick_best_node(state: PipelineState) -> PipelineState:
        print("  [node] pick_best")
        suggestions = state["suggestions"]
        current_smiles = state["current_smiles"]
        iteration = state["iteration"]

        if not suggestions or iteration + 1 >= max_iter:
            return {**state, "status": "max_iterations"}

        best_smiles, best_name, best_sim = None, "", 0.0
        for sug in suggestions:
            sim = compute_combined_similarity(
                current_smiles, sug["smiles"], backbone,
                kpgt_dir=kpgt_dir, device=device,
            )
            print(f"    {sug.get('name','?')[:30]:<30} sim={sim:.4f}")
            if sim > best_sim:
                best_sim, best_smiles, best_name = sim, sug["smiles"], sug.get("name", "")

        if best_smiles is None:
            return {**state, "status": "max_iterations"}

        print(f"  [best] {best_name}  sim={best_sim:.4f}")
        return {
            **state,
            "current_smiles": best_smiles,
            "current_name": best_name,
            "prev_confidence": state["confidence"],
            "iteration": iteration + 1,
            "status": "running",
        }

    # ── Node 5: Explain ───────────────────────────────────────────────────────
    def explain_node(state: PipelineState) -> PipelineState:
        print(f"  [node] explain  status={state['status']}")
        if not explain or not state["scores"]:
            return {**state, "explanation": ""}
        try:
            expl = explain_predictions(
                state["current_smiles"], state["scores"],
                task_types=task_type_map, llm_client=llm_client,
            )
        except Exception as e:
            expl = f"[explainer error] {e}"
        return {**state, "explanation": expl}

    # ── Conditional routing ───────────────────────────────────────────────────
    def route_after_toxric(state: PipelineState) -> Literal["explain", "predict_kpgt"]:
        return "explain" if state["status"] == "toxric_match" else "predict_kpgt"

    def route_after_llm(
        state: PipelineState,
    ) -> Literal["explain", "pick_best"]:
        return "explain" if state["status"] in ("satisfactory", "degrading") else "pick_best"

    def route_after_pick_best(
        state: PipelineState,
    ) -> Literal["explain", "check_toxric"]:
        return "explain" if state["status"] == "max_iterations" else "check_toxric"

    # ── Assemble graph ────────────────────────────────────────────────────────
    graph = StateGraph(PipelineState)

    graph.add_node("check_toxric",       check_toxric_node)
    graph.add_node("predict_kpgt",       predict_kpgt_node)
    graph.add_node("run_llm_validation", run_llm_validation_node)
    graph.add_node("pick_best",          pick_best_node)
    graph.add_node("explain",            explain_node)

    graph.add_edge(START,                  "check_toxric")
    graph.add_conditional_edges("check_toxric",       route_after_toxric)
    graph.add_edge("predict_kpgt",         "run_llm_validation")
    graph.add_conditional_edges("run_llm_validation", route_after_llm)
    graph.add_conditional_edges("pick_best",          route_after_pick_best)
    graph.add_edge("explain",              END)

    return graph.compile()


# ── Public interface (unchanged) ──────────────────────────────────────────────

def run_agentic_pipeline(
    smiles: str,
    model,
    backbone,
    cfg: dict,
    merged_csv_path: str = "data/toxric_merged.csv",
    pretrained_path: str = "external/KPGT/models/pretrained/base/base.pth",
    llm_client=None,
    claude_client=None,     # backward-compat alias
    satisfactory_threshold: float = 6.0,
    max_iter: int = 3,
    kpgt_dir: str = "external/KPGT",
    device: str = "cpu",
    explain: bool = True,
) -> dict:
    """Run the agentic toxicity prediction pipeline (LangGraph implementation).

    Same signature as the previous hand-written version — callers need no changes.

    Returns dict with keys: status, source, original_smiles, final_smiles,
    final_name, scores, iterations, explanation.
    """
    load_dotenv()

    # ── Build GPT orchestrator (AzureChatOpenAI) ──────────────────────────────
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    model_deployment = os.getenv("MODEL_DEPLOYMENT", "gpt-4o")

    if not azure_endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT not set. Add it to .env (same value used in tools-app)."
        )

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    orchestrator_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=model_deployment,
        azure_ad_token_provider=token_provider,
        api_version="2024-05-01-preview",
    )

    # ── Build AzureOpenAI client for reasoning tools + explainer ─────────────
    # Same endpoint/credentials; AzureOpenAI is the lower-level SDK for direct calls.
    from openai import AzureOpenAI as _AzureOpenAI
    active_llm_client = llm_client or claude_client or _AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-05-01-preview",
    )

    # ── Build inner graph (LangGraph tool-calling, mirrors tools-agent-langgraph.py) ─
    tools = _make_tools(active_llm_client, model_deployment)
    llm_with_tools = orchestrator_llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    inner_graph = _build_inner_graph(llm_with_tools, tool_node)

    # ── Build outer graph ─────────────────────────────────────────────────────
    pipeline = _build_pipeline_graph(
        model=model, backbone=backbone, cfg=cfg,
        inner_graph=inner_graph,
        merged_csv_path=merged_csv_path,
        satisfactory_threshold=satisfactory_threshold,
        max_iter=max_iter,
        kpgt_dir=kpgt_dir,
        device=device,
        llm_client=active_llm_client,
        explain=explain,
    )

    # ── Initial state ─────────────────────────────────────────────────────────
    initial_name = lookup_name_pubchem(smiles)
    initial_state: PipelineState = {
        "original_smiles": smiles,
        "current_smiles": smiles,
        "current_name": initial_name,
        "scores": {},
        "iteration": 0,
        "prev_confidence": None,
        "confidence": 0.0,
        "reasoning": "",
        "suggestions": [],
        "status": "running",
        "source": "kpgt",
        "iterations": [],
        "explanation": "",
    }

    print(f"\n[LangGraph pipeline] input={smiles[:50]}  threshold={satisfactory_threshold}  "
          f"max_iter={max_iter}")
    final_state = pipeline.invoke(initial_state)

    return {
        "status":          final_state["status"],
        "source":          final_state["source"],
        "original_smiles": smiles,
        "final_smiles":    final_state["current_smiles"],
        "final_name":      final_state["current_name"],
        "scores":          final_state["scores"],
        "iterations":      final_state["iterations"],
        "explanation":     final_state["explanation"],
    }
