"""
Plain-English explanation of toxicity predictions via Azure OpenAI (GPT).

Uses AzureOpenAI with get_bearer_token_provider (DefaultAzureCredential) —
same auth pattern as the rest of the pipeline.
"""

from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI


SYSTEM_PROMPT = """You are a toxicologist translating computational predictions into plain English for a non-expert reader (a chemist, biologist, or curious clinician).

You will receive a compound's SMILES string and a table of predicted toxicity scores across multiple endpoints. Each endpoint represents a specific biological hazard (e.g. liver damage, cardiac arrhythmia risk from hERG channel blockade, mutagenicity, etc.).

Your output must follow this structure (Markdown):

### Compound
One sentence: if you recognize the SMILES as a known drug or chemical, name it; otherwise describe its structural class briefly.

### High-risk endpoints (probability ≥ 0.5)
For each, write 1–2 sentences explaining:
- What the endpoint biologically measures
- Which organ / system would be affected
- The observable health effect or symptom in animals or humans

### Low-risk endpoints (probability < 0.5)
Group these into a single short paragraph. Mention the most notable absences (e.g. "no predicted hepatotoxicity or cardiotoxicity").

### Overall assessment
One or two sentences summarizing the risk profile.

### Caveat
A single sentence noting these are computational predictions from a fine-tuned KPGT model on TOXRIC data — not clinical findings — and any decision-making should rely on actual toxicology studies.

Constraints:
- Stay under 400 words total.
- Avoid jargon when possible; when unavoidable, define it briefly in parentheses.
- Do not invent endpoints that weren't in the input. Only discuss what's listed."""


def build_azure_openai_client() -> AzureOpenAI:
    """Build an AzureOpenAI client using DefaultAzureCredential bearer token."""
    load_dotenv()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT not set. Add it to .env."
        )
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-05-01-preview",
    )


# Kept for any callers that still reference the old name.
def build_foundry_claude_client():
    return build_azure_openai_client()


def explain_predictions(
    smiles: str,
    predictions: dict[str, float],
    task_types: dict[str, str] | None = None,
    llm_client: AzureOpenAI | None = None,
    claude_client=None,          # backward-compat alias — ignored if llm_client is set
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Generate a plain-English health summary for one compound's predictions.

    Args:
        smiles:       The compound's SMILES string.
        predictions:  {endpoint_name: score}. Scores are in [0, 1] for
                      classification and raw values for regression.
        task_types:   {endpoint_name: 'classification'|'regression'}.
                      Defaults to all-classification.
        llm_client:   Pre-built AzureOpenAI client. Built from .env if None.
        model:        Azure deployment name. Defaults to $MODEL_DEPLOYMENT or 'gpt-4o'.

    Returns:
        Markdown-formatted explanation as a string.
    """
    client = llm_client or claude_client or build_azure_openai_client()
    if model is None:
        model = os.getenv("MODEL_DEPLOYMENT", "gpt-4o")

    types = task_types or {k: "classification" for k in predictions}
    lines = [f"SMILES: {smiles}", "", "Predicted toxicity scores (sorted high to low):"]
    for endpoint, value in sorted(predictions.items(), key=lambda kv: -kv[1]):
        kind = types.get(endpoint, "classification")
        if kind == "classification":
            lines.append(f"- {endpoint}: probability = {value:.3f}")
        else:
            lines.append(f"- {endpoint}: value = {value:.3f}")
    user_msg = "\n".join(lines)

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content or ""
