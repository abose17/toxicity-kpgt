#!/usr/bin/env bash
# ============================================================
#  ToxNav — interactive one-shot setup
#  Run from the repo root:  bash setup.sh
# ============================================================
set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

# ── Colour helpers ────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; CYAN="\033[0;36m"; NC="\033[0m"
ok()   { echo -e "${GREEN}  ✔  $*${NC}"; }
info() { echo -e "${CYAN}  →  $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${NC}"; }
err()  { echo -e "${RED}  ✖  $*${NC}"; }
hr()   { echo -e "${CYAN}$(printf '─%.0s' {1..60})${NC}"; }
step() { echo; hr; echo -e "${CYAN}  STEP $1 — $2${NC}"; hr; }
pause_for_user() {
    echo
    warn "$1"
    echo -e "  Press ${YELLOW}Enter${NC} when done, or ${RED}Ctrl+C${NC} to abort."
    read -r
}


echo
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         ToxNav — Drug Toxicity Prediction Setup              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
echo "  This script will guide you through:"
echo "    0  Git LFS checkpoint download (if available)"
echo "    1  Python environment + dependencies"
echo "    2  TOXRIC dataset + KPGT source code download"
echo "    3  KPGT pretrained model weights (manual download)"
echo "    4  Azure credential check (az login)"
echo "    5  .env configuration"
echo "    6  Launch the Streamlit app (demo mode)"
echo
echo -e "  Press ${YELLOW}Enter${NC} to begin, or ${RED}Ctrl+C${NC} to exit."
read -r


# ────────────────────────────────────────────────────────────
step 0 "Git LFS — fine-tuned checkpoint"
# ────────────────────────────────────────────────────────────

if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    info "Pulling LFS objects (fine-tuned checkpoint if uploaded) ..."
    git lfs pull
    if [ -f "checkpoints/best.pt" ]; then
        SIZE=$(du -sh checkpoints/best.pt | cut -f1)
        ok "checkpoints/best.pt downloaded ($SIZE)"
    else
        warn "No checkpoint in LFS yet — train one with scripts/train.py or Azure ML."
        warn "The Streamlit app will run in Demo Mode without it."
    fi
else
    warn "Git LFS not installed — skipping checkpoint download."
    warn "Install it later with: brew install git-lfs && git lfs install && git lfs pull"
fi


# ────────────────────────────────────────────────────────────
step 1 "Python environment"
# ────────────────────────────────────────────────────────────

# Check Python version
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

info "Detected $PYTHON → $PY_VER"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    err "Python 3.10+ required (3.12 recommended). Install it and re-run."
    exit 1
fi
ok "Python version OK"

# Create venv
if [ -d "$VENV_DIR" ]; then
    ok "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated $VENV_DIR"

# Upgrade pip silently
pip install --upgrade pip --quiet

# Install main requirements
info "Installing requirements.txt (this may take a few minutes) ..."
pip install -r requirements.txt --quiet
ok "Core packages installed"

# DGL — must come from its own wheel server (not PyPI)
info "Installing DGL 2.2.1 from dgl.ai wheel server ..."
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/repo.html --quiet
ok "DGL installed"


# ────────────────────────────────────────────────────────────
step 2 "TOXRIC data + KPGT source code"
# ────────────────────────────────────────────────────────────

info "Running scripts/setup.py  (downloads TOXRIC ~8 MB, clones KPGT) ..."
python scripts/setup.py

info "Merging 30 TOXRIC endpoint CSVs into data/toxric_merged.csv ..."
python scripts/merge_toxric.py \
    --source-dir data/toxric/toxric_30_datasets/toxric_30_datasets \
    --output data/toxric_merged.csv
ok "TOXRIC merged CSV ready"


# ────────────────────────────────────────────────────────────
step 3 "KPGT pretrained model weights  (manual download)"
# ────────────────────────────────────────────────────────────

BASEPTH="external/KPGT/models/pretrained/base/base.pth"

if [ -f "$BASEPTH" ]; then
    ok "base.pth already present at $BASEPTH"
else
    echo
    warn "The KPGT pretrained weights must be downloaded manually (~270 MB)."
    echo
    echo "  1. Open this URL in your browser:"
    echo -e "     ${YELLOW}https://figshare.com/s/d488f30c23946cf6898f${NC}"
    echo "  2. Download the zip file."
    echo "  3. Extract it and copy base.pth to:"
    echo -e "     ${YELLOW}$(pwd)/external/KPGT/models/pretrained/base/base.pth${NC}"
    echo
    pause_for_user "Complete the download above, then press Enter to continue."

    if [ -f "$BASEPTH" ]; then
        ok "base.pth found"
    else
        warn "base.pth not found yet — you can add it later."
        warn "The app will run in Demo Mode without it."
    fi
fi


# ────────────────────────────────────────────────────────────
step 4 "Azure credentials  (az login)"
# ────────────────────────────────────────────────────────────

if command -v az &>/dev/null; then
    AZ_ACCOUNT=$(az account show --query "user.name" -o tsv 2>/dev/null || true)
    if [ -n "$AZ_ACCOUNT" ]; then
        ok "Already logged in as: $AZ_ACCOUNT"
    else
        warn "Not logged in to Azure CLI."
        pause_for_user "Run 'az login' in another terminal, then press Enter."
        AZ_ACCOUNT=$(az account show --query "user.name" -o tsv 2>/dev/null || true)
        if [ -n "$AZ_ACCOUNT" ]; then
            ok "Logged in as: $AZ_ACCOUNT"
        else
            warn "Still not logged in — agentic pipeline will fail without Azure auth."
            warn "You can still use Demo Mode in the Streamlit app."
        fi
    fi
else
    warn "Azure CLI not found. Install from https://learn.microsoft.com/cli/azure/install-azure-cli"
    warn "You can still use Demo Mode in the Streamlit app."
fi


# ────────────────────────────────────────────────────────────
step 5 ".env configuration"
# ────────────────────────────────────────────────────────────

if [ -f ".env" ]; then
    ok ".env already exists — skipping"
else
    info "Creating .env from template ..."
    cp .env.example .env

    echo
    echo "  The pipeline needs two values from your Azure OpenAI resource:"
    echo
    echo -e "  ${YELLOW}AZURE_OPENAI_ENDPOINT${NC}"
    echo "  → Azure portal → your OpenAI resource → Keys and Endpoint → Endpoint"
    echo -e "  ${YELLOW}MODEL_DEPLOYMENT${NC}"
    echo "  → Azure OpenAI Studio → Deployments → your GPT-4o deployment name"
    echo
    read -rp "  Enter AZURE_OPENAI_ENDPOINT (or press Enter to skip): " AZ_ENDPOINT
    read -rp "  Enter MODEL_DEPLOYMENT      (or press Enter to use 'gpt-4o'): " AZ_DEPLOY
    AZ_DEPLOY=${AZ_DEPLOY:-gpt-4o}

    if [ -n "$AZ_ENDPOINT" ]; then
        # Replace empty placeholder values in .env
        sed -i.bak "s|AZURE_OPENAI_ENDPOINT=\"\"|AZURE_OPENAI_ENDPOINT=\"$AZ_ENDPOINT\"|" .env
        rm -f .env.bak
    fi
    sed -i.bak "s|MODEL_DEPLOYMENT=\"gpt-4o\"|MODEL_DEPLOYMENT=\"$AZ_DEPLOY\"|" .env
    rm -f .env.bak

    ok ".env written"
fi


# ────────────────────────────────────────────────────────────
step 6 "Smoke test + launch Streamlit"
# ────────────────────────────────────────────────────────────

info "Running import smoke test ..."
python -c "
from toxpkg.agentic_pipeline import run_agentic_pipeline, PipelineState
from toxpkg.explainer import explain_predictions
from toxpkg.visualizer import draw_molecule_grid
print('  imports OK')
"
ok "All core modules import cleanly"

echo
hr
echo -e "${GREEN}"
echo "  ✔  Setup complete!"
echo -e "${NC}"
echo "  To launch the Streamlit app:"
echo -e "    ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo -e "    ${YELLOW}streamlit run app.py${NC}"
echo
echo "  The app opens in Demo Mode (Aspirin example) — no checkpoint needed."
echo "  Uncheck 'Demo Mode' and provide a trained checkpoint to run real predictions."
echo
echo "  Useful scripts:"
echo "    python scripts/train.py          # fine-tune KPGT on TOXRIC"
echo "    python scripts/predict_agentic.py --smiles \"<SMILES>\" --checkpoint checkpoints/best.pt"
echo "    python scripts/find_alternatives.py --smiles \"<SMILES>\" --checkpoint checkpoints/best.pt"
echo
hr
echo
read -rp "  Launch Streamlit now? [Y/n]: " LAUNCH
LAUNCH=${LAUNCH:-Y}
if [[ "$LAUNCH" =~ ^[Yy]$ ]]; then
    streamlit run app.py
fi
