# ToxNav — Drug Toxicity Prediction & Safer Alternatives

Given a drug molecule SMILES, ToxNav predicts its toxicity across 30 biological endpoints and finds structurally similar molecules with lower predicted toxicity.

## Architecture

```
Input SMILES
     │
     ▼
ChEMBL similarity search  ──►  up to 20 structural analogues
     │
     ▼
TOXRIC matcher  ──(found)──►  use ground-truth binary labels
     │ (not found)
     ▼
KPGT fine-tuned GNN prediction  (30-endpoint sigmoid probabilities)
     │
     ▼
LangGraph agentic loop ──────────────────────────────────────────┐
  │  AzureChatOpenAI (GPT) orchestrates two tools:               │
  │    • validate_toxicity_prediction  → confidence 0–10         │
  │    • suggest_safer_alternatives    → 5 proxy drugs           │
  │  Loops up to 3× if confidence < threshold                    │
  └─────────────────────────────────────────────────────────────►│
                                                                  ▼
                                                    GPT explanation (Markdown)
                                                    Ranked comparison table
                                                    Molecule grid PNG
                                                    MCS highlight PNG
                                                    Streamlit UI
```

**Key components:**

| Component | Role |
|---|---|
| [TOXRIC](https://figshare.com/articles/dataset/TOXRIC/27195339) | 30-dataset toxicity ground-truth labels |
| [KPGT](https://github.com/lihan97/KPGT) | Pretrained molecular graph transformer backbone |
| [ChEMBL REST API](https://www.ebi.ac.uk/chembl/) | Structural similarity search |
| AzureChatOpenAI (GPT) | LangGraph orchestrator + reasoning (validate, suggest, explain) |
| Streamlit | Interactive web UI |

---

## Quick Start

```bash
git clone https://github.com/abose17/toxicity-kpgt.git
cd toxicity-kpgt
bash setup.sh
```

The setup script walks you through all steps interactively. It will:

1. Create a Python virtual environment and install all dependencies
2. Download the TOXRIC dataset and merge it into a single CSV
3. Clone the KPGT source code
4. Guide you to download the KPGT pretrained weights manually
5. Check your Azure CLI login (`az login`)
6. Prompt for your Azure OpenAI endpoint and write your `.env`
7. Run a smoke test and offer to launch the Streamlit app

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ (3.12 recommended) | `python3 --version` |
| Git | any | for cloning KPGT |
| Azure CLI | any | `az login` for `DefaultAzureCredential` auth |
| Azure OpenAI resource | GPT-4o deployment | for agentic validation + explanation |

---

## Step-by-step Setup (manual alternative to `setup.sh`)

### 1 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# DGL must be installed from its own wheel server (not on PyPI)
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/repo.html
```

### 2 — TOXRIC data + KPGT source code

```bash
python scripts/setup.py            # downloads TOXRIC (~8 MB) + clones KPGT

python scripts/merge_toxric.py \
    --source-dir data/toxric/toxric_30_datasets/toxric_30_datasets \
    --output data/toxric_merged.csv
```

### 3 — KPGT pretrained weights (manual)

The KPGT authors host the pretrained backbone behind an anonymous Figshare share link.

1. Open **https://figshare.com/s/d488f30c23946cf6898f**
2. Download the zip (~270 MB)
3. Extract and copy `base.pth` to:
   ```
   external/KPGT/models/pretrained/base/base.pth
   ```

> The app runs in **Demo Mode** without this file — no weights needed for the Streamlit UI demo.

### 4 — Azure authentication

```bash
az login
```

`DefaultAzureCredential` picks up your session automatically. No API key is stored anywhere.

### 5 — Environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
MODEL_DEPLOYMENT="gpt-4o"
```

Find these in **Azure portal → your OpenAI resource → Keys and Endpoint**.

---

## Running the App

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens at **http://localhost:8501**.

**Demo Mode** (default — no checkpoint needed): shows a pre-computed Aspirin example with realistic dummy scores. Fully interactive — comparison table, molecule visualizations, explanation, and iteration trace all work.

**Real Mode**: uncheck *Demo Mode* in the sidebar. Requires:
- A trained KPGT checkpoint at `checkpoints/best.pt` (see Training below)
- `AZURE_OPENAI_ENDPOINT` set in `.env`
- Active `az login` session

---

## Training the KPGT model

Fine-tune KPGT on the merged TOXRIC dataset:

```bash
python scripts/train.py \
    --merged-csv data/toxric_merged.csv \
    --pretrained external/KPGT/models/pretrained/base/base.pth \
    --output checkpoints/best.pt
```

Or submit to **Azure ML** (GPU recommended):

```bash
# Fill in Azure ML fields in .env first, then:
python azure/submit_job.py
```

---

## CLI Scripts

```bash
# Agentic prediction for one SMILES (requires trained checkpoint)
python scripts/predict_agentic.py \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --checkpoint checkpoints/best.pt

# Find safer structural alternatives
python scripts/find_alternatives.py \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --checkpoint checkpoints/best.pt \
    --output results/

# Basic batch prediction (no agentic loop)
python scripts/predict.py \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --checkpoint checkpoints/best.pt
```

---

## Project Structure

```
toxicity-kpgt/
├── app.py                    # Streamlit UI
├── setup.sh                  # Interactive one-shot setup
├── requirements.txt
├── .env.example              # Environment variable template
│
├── toxpkg/                   # Core Python package
│   ├── agentic_pipeline.py   # LangGraph outer + inner graphs
│   ├── comparator.py         # Rank molecules by toxicity score
│   ├── explainer.py          # GPT plain-English explanation
│   ├── featurizer.py         # KPGT graph/fp/descriptor features
│   ├── model.py              # LiGhTPredictor wrapper
│   ├── predict.py            # Batch KPGT inference
│   ├── similarity.py         # ChEMBL search + PubChem name lookup
│   ├── toxric_matcher.py     # Match SMILES against TOXRIC ground truth
│   ├── trainer.py            # Fine-tuning loop
│   ├── validator.py          # LLM validate + suggest + combined similarity
│   └── visualizer.py         # RDKit molecule grid + MCS highlight PNGs
│
├── scripts/
│   ├── setup.py              # Phase A: download TOXRIC + clone KPGT
│   ├── merge_toxric.py       # Merge 30 endpoint CSVs → one wide CSV
│   ├── train.py              # Fine-tune KPGT locally
│   ├── predict.py            # Batch prediction CLI
│   ├── predict_agentic.py    # Agentic pipeline CLI
│   └── find_alternatives.py  # Safer-alternatives finder CLI
│
├── azure/
│   ├── submit_job.py         # Submit Azure ML training job
│   └── env.yml               # Azure ML conda environment
│
└── serving/
    ├── app.py                # FastAPI serving endpoint (for Azure VM)
    └── setup_vm.sh           # VM provisioning script
```

---

## How It Works

1. **ChEMBL search** — queries the ChEMBL REST API for molecules with ≥70% Tanimoto similarity to the input
2. **TOXRIC match** — checks if each candidate's canonical SMILES exists in the merged TOXRIC CSV; uses ground-truth labels if found
3. **KPGT prediction** — unmatched molecules are scored by the fine-tuned graph neural network across 30 toxicity endpoints
4. **Agentic validation** — GPT reads the predicted scores and assesses whether they align with its knowledge of the drug; if confidence < threshold it suggests 5 proxy drugs and the loop continues
5. **Ranking** — molecules sorted by total toxicity score (sum of sigmoid probabilities across all endpoints)
6. **Explanation** — GPT generates a structured plain-English health risk summary

---

## Security

- No API keys are stored in code or `.env`
- All Azure auth uses `DefaultAzureCredential` (`az login` / managed identity)
- `.env` is gitignored
- Model weights and data files are gitignored (download separately)
