FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libxrender1 \
    libxext6 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# torch 2.1.2 + torchdata 0.6.1 — matched pair used during training.
# torch 2.3+ removed DILL_AVAILABLE from datapipes, breaking torchdata.
RUN pip install --no-cache-dir \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cpu

# DGL CPU — 2.1.0 is the latest available from the DGL wheel server
RUN pip install --no-cache-dir \
    dgl==2.1.0 -f https://data.dgl.ai/wheels/repo.html

# All remaining app dependencies (exact versions from working local venv)
COPY requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application source
COPY toxpkg/          ./toxpkg/
COPY scripts/         ./scripts/
COPY app.py           ./app.py

# KPGT source (needed for graph featurization)
COPY external/KPGT/src/  ./external/KPGT/src/

# Data and model weights
COPY data/toxric_merged.csv  ./data/toxric_merged.csv
COPY external/KPGT/models/pretrained/base/base.pth \
     ./external/KPGT/models/pretrained/base/base.pth
COPY checkpoints/best.pt  ./checkpoints/best.pt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
