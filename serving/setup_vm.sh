#!/usr/bin/env bash
# One-shot VM bootstrap. Run this ONCE on the Azure VM after first ssh.
#
# Prereqs done elsewhere BEFORE running this:
#   - VM provisioned (Standard_B2s recommended for personal use)
#   - SSH key set up, NSG allows 22 (your IP only) + 8000 (your IP only)
#   - Repo cloned to ~/toxicity-kpgt  (or uploaded via scp/rsync)
#   - Trained checkpoint at ~/toxicity-kpgt/checkpoints/best.pt
#   - KPGT source at ~/toxicity-kpgt/external/KPGT/  (the wrapper imports from it)
#   - .env file at ~/toxicity-kpgt/.env with:
#         FOUNDRY_BASE_URL=...
#         CLAUDE_MODEL=claude-opus-4-7
#         API_KEY=<your chosen key>
#         MODEL_PATH=checkpoints/best.pt
#         KPGT_DIR=external/KPGT
#         SERVE_DEVICE=cpu
#
# After this script: API runs at http://<vm-ip>:8000/predict, auto-starts on reboot.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo "[1/5] system packages"
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-venv python3-pip git build-essential

echo "[2/5] python venv + CPU deps (~5 min, ~1.5 GB)"
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r serving/requirements-cpu.txt

echo "[3/5] sanity import check"
.venv/bin/python - <<'PY'
mods = ['torch', 'dgl', 'dgllife', 'rdkit', 'fastapi', 'uvicorn',
        'anthropic', 'azure.identity', 'transformers']
for m in mods:
    __import__(m); print(f'  OK {m}')
print('[ok] all CPU inference deps importable')
PY

echo "[4/5] systemd service install"
sudo cp serving/kpgt-toxric.service /etc/systemd/system/kpgt-toxric.service
sudo systemctl daemon-reload
sudo systemctl enable kpgt-toxric.service

echo "[5/5] start service"
sudo systemctl start kpgt-toxric.service
sleep 3
sudo systemctl --no-pager status kpgt-toxric.service | head -20

echo
echo "=================================================================="
echo "Done. Try:"
echo "  curl http://localhost:8000/health"
echo "  curl -X POST http://localhost:8000/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -H 'X-API-Key: \$API_KEY' \\"
echo "       -d '{\"smiles\": [\"CCO\"], \"top_k\": 5}'"
echo
echo "Logs:"
echo "  journalctl -u kpgt-toxric -f"
echo "=================================================================="
