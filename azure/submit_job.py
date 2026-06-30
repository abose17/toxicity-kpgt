"""
Submit a KPGT fine-tuning job to Azure Machine Learning.

What this does:
    1. Authenticates to your AML workspace via DefaultAzureCredential
    2. Defines (or reuses) the Environment built from azure/env.yml
    3. Uploads code (src/ + scripts/ + external/KPGT/) and data as Inputs
    4. Submits a `command` job that runs `python scripts/train.py ...`
       on the GPU compute cluster you specify
    5. Optionally registers the trained model in the workspace after the job completes

Required Azure config (in .env or environment variables):
    AZURE_SUBSCRIPTION_ID
    AZURE_RESOURCE_GROUP
    AZURE_WORKSPACE_NAME
    AZURE_COMPUTE_CLUSTER   (e.g. 'gpu-cluster' — must already exist with at least one V100/T4 node)

Usage from labfiles/toxicity-kpgt/:
    # First time (uploads the preprocessed cache and pretrained weights as job inputs)
    python azure/submit_job.py \\
        --dataset toxric_multitask \\
        --epochs 30 \\
        --batch-size 32

    # Subsequent runs (reuse already-uploaded blobs by referencing registered data assets)
    python azure/submit_job.py \\
        --dataset toxric_multitask \\
        --epochs 50 \\
        --toxric-cache-uri azureml://datastores/workspaceblobstore/paths/kpgt-data/toxric_multitask/ \\
        --pretrained-uri  azureml://datastores/workspaceblobstore/paths/kpgt-data/pretrained/

Prerequisites BEFORE you submit:
    - Run Phase A + Phase B locally first (`scripts/setup.py`, then `scripts/merge_toxric.py`
      and `scripts/preprocess.py`) so `data/kpgt-cache/<dataset>/` and `external/KPGT/models/
      pretrained/base/base.pth` exist on your machine.
    - Have an AML workspace + GPU compute cluster already provisioned.
    - `pip install azure-ai-ml azure-identity python-dotenv`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from azure.ai.ml import Input, MLClient, Output, command
    from azure.ai.ml.entities import Environment, Model
    from azure.identity import DefaultAzureCredential
except ImportError:
    print("[error] azure-ai-ml not installed. Run: pip install azure-ai-ml azure-identity")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent   # labfiles/toxicity-kpgt/
ENV_NAME = "kpgt-toxric-env"
EXPERIMENT_NAME = "kpgt-toxric"
DEFAULT_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="toxric_multitask",
                   help="Name of the preprocessed dataset folder under data/kpgt-cache/.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--compute", default=None,
                   help="AML compute cluster name. Falls back to $AZURE_COMPUTE_CLUSTER.")
    p.add_argument("--toxric-cache-uri", default=None,
                   help="Pre-uploaded data asset URI. If unset, uploads the local dir.")
    p.add_argument("--pretrained-uri", default=None,
                   help="Pre-uploaded pretrained model URI. If unset, uploads the local dir.")
    p.add_argument("--display-name", default=None)
    p.add_argument("--register-model", action="store_true",
                   help="Wait for job to complete, then register the output as an AML model.")
    return p.parse_args()


def get_ml_client() -> MLClient:
    load_dotenv()
    required = ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing required env vars: {missing}. Set them in .env.")

    return MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )


def ensure_environment(ml_client: MLClient) -> Environment:
    """Create or update the conda-based environment from azure/env.yml."""
    env = Environment(
        name=ENV_NAME,
        description="KPGT fine-tuning: torch+dgl+rdkit+dgllife+transformers on CUDA 11.8.",
        conda_file=str(REPO_ROOT / "azure" / "env.yml"),
        image=DEFAULT_IMAGE,
    )
    registered = ml_client.environments.create_or_update(env)
    print(f"[env] {registered.name}:{registered.version}")
    return registered


def build_inputs(args) -> dict:
    """Decide whether to upload local dirs or reference pre-registered data assets."""
    local_cache = REPO_ROOT / "data" / "kpgt-cache" / args.dataset
    local_pretrained = REPO_ROOT / "external" / "KPGT" / "models" / "pretrained" / "base"

    if args.toxric_cache_uri:
        cache_input = Input(type="uri_folder", path=args.toxric_cache_uri, mode="ro_mount")
    else:
        if not local_cache.exists():
            raise SystemExit(
                f"Local cache not found at {local_cache}. Run scripts/preprocess.py first, "
                f"or pass --toxric-cache-uri to reference a registered asset."
            )
        cache_input = Input(type="uri_folder", path=str(local_cache), mode="ro_mount")
        print(f"[input] uploading toxric cache from {local_cache}")

    if args.pretrained_uri:
        pretrained_input = Input(type="uri_folder", path=args.pretrained_uri, mode="ro_mount")
    else:
        if not (local_pretrained / "base.pth").exists():
            raise SystemExit(
                f"base.pth not found at {local_pretrained}. Download it from the KPGT Figshare "
                f"share and place it there, or pass --pretrained-uri."
            )
        pretrained_input = Input(type="uri_folder", path=str(local_pretrained), mode="ro_mount")
        print(f"[input] uploading pretrained weights from {local_pretrained}")

    return {"toxric_cache": cache_input, "pretrained": pretrained_input}


def main() -> int:
    args = parse_args()
    ml_client = get_ml_client()

    compute = args.compute or os.environ.get("AZURE_COMPUTE_CLUSTER")
    if not compute:
        raise SystemExit("--compute (or AZURE_COMPUTE_CLUSTER) is required.")

    env = ensure_environment(ml_client)
    inputs = build_inputs(args)

    # AML mounts `inputs.toxric_cache` as a folder containing the same .pkl/.npz files we'd
    # have in data/kpgt-cache/<dataset>/. train.py expects --data-root to point to the PARENT
    # of that folder, so we craft a small shell prologue to set up a symlink at the expected path.
    train_cmd = (
        "set -ex && "
        "mkdir -p data/kpgt-cache && "
        "ln -sfn ${{inputs.toxric_cache}} data/kpgt-cache/" + args.dataset + " && "
        f"python scripts/train.py "
        f"--dataset {args.dataset} "
        f"--data-root data/kpgt-cache "
        f"--pretrained ${{{{inputs.pretrained}}}}/base.pth "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        f"--backbone-lr {args.backbone_lr} "
        f"--head-lr {args.head_lr} "
        f"--device cuda "
        f"--checkpoint-dir ${{{{outputs.model_dir}}}}"
    )

    job = command(
        code=str(REPO_ROOT),
        command=train_cmd,
        inputs=inputs,
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=f"{env.name}@latest",
        compute=compute,
        experiment_name=EXPERIMENT_NAME,
        display_name=args.display_name or f"kpgt-{args.dataset}-e{args.epochs}",
    )

    print(f"\n[submit] compute={compute}  experiment={EXPERIMENT_NAME}")
    submitted = ml_client.jobs.create_or_update(job)
    print(f"[ok] job: {submitted.name}")
    print(f"[ok] studio: {submitted.studio_url}")

    if args.register_model:
        print("\n[wait] waiting for job to complete before registering model...")
        ml_client.jobs.stream(submitted.name)
        done = ml_client.jobs.get(submitted.name)
        if done.status != "Completed":
            print(f"[skip] job status is {done.status} — not registering.")
            return 1
        model = Model(
            path=f"azureml://jobs/{submitted.name}/outputs/model_dir",
            name=f"kpgt-{args.dataset}",
            description=f"KPGT multi-task fine-tune from job {submitted.name}",
            type="custom_model",
        )
        registered = ml_client.models.create_or_update(model)
        print(f"[model] {registered.name}:{registered.version}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
