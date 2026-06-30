"""
Phase A utilities: download TOXRIC, inspect its schema, clone KPGT, and
verify the pretrained checkpoint is in place.

Every URL and import in this module has been verified against the real
TOXRIC Figshare record (DOI 27195339) and the lihan97/KPGT GitHub repo.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# --- Verified URLs (Figshare API responses captured 2026-06-09) ---------

TOXRIC_FIGSHARE_ARTICLE_ID = 27195339
TOXRIC_FILES = {
    "toxric_30_datasets.zip": {
        "file_id": 49694949,
        "size_bytes": 7_767_218,
        "url": "https://ndownloader.figshare.com/files/49694949",
    },
    "multiple_endpoint_acute_toxicity_dataset.zip": {
        "file_id": 49697235,
        "size_bytes": 4_802_087,
        "url": "https://ndownloader.figshare.com/files/49697235",
    },
    "all_descriptors.txt": {
        "file_id": 49954575,
        "size_bytes": 564_351_082,
        "url": "https://ndownloader.figshare.com/files/49954575",
    },
    "115-endpoint_acute_toxiciy_dataset.zip": {
        "file_id": 53142902,
        "size_bytes": 14_423_422,
        "url": "https://ndownloader.figshare.com/files/53142902",
    },
}

KPGT_GIT_URL = "https://github.com/lihan97/KPGT.git"
KPGT_PRETRAINED_INFO = {
    "share_url": "https://figshare.com/s/d488f30c23946cf6898f",
    "expected_path": "models/pretrained/base/base.pth",
    "approx_size_mb": 270,
}


# --- Download helpers ----------------------------------------------------

def _stream_download(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", expected_size or 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download_toxric(
    target_dir: str | os.PathLike = "data/toxric",
    which: str = "toxric_30_datasets.zip",
    extract: bool = True,
) -> Path:
    """Download a TOXRIC archive from Figshare into target_dir.

    `which` must be a key of TOXRIC_FILES. The 30-datasets zip (~7.7 MB)
    is the right one for multi-task fine-tuning experiments.
    """
    if which not in TOXRIC_FILES:
        raise ValueError(
            f"Unknown TOXRIC file '{which}'. Pick one of: {list(TOXRIC_FILES)}"
        )

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / which

    spec = TOXRIC_FILES[which]
    if archive_path.exists() and archive_path.stat().st_size >= spec["size_bytes"] * 0.95:
        print(f"[skip] {which} already present at {archive_path}")
    else:
        print(f"[download] {which} ({spec['size_bytes'] / 1e6:.1f} MB)")
        _stream_download(spec["url"], archive_path, expected_size=spec["size_bytes"])

    if extract and which.endswith(".zip"):
        extract_dir = target_dir / which.replace(".zip", "")
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[skip] {extract_dir} already extracted")
        else:
            print(f"[extract] {archive_path} -> {extract_dir}")
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
        return extract_dir

    return archive_path


# --- Schema inspection ---------------------------------------------------

def list_toxric_subdatasets(toxric_extract_dir: str | os.PathLike) -> list[Path]:
    """Walk the extracted TOXRIC archive and return every CSV found.

    The 30-datasets archive contains a per-endpoint CSV per subdirectory.
    We don't assume an exact layout — we just enumerate CSVs so the user
    can pick one. Each CSV typically has columns like:
        TAID, Canonical SMILES, <label columns...>
    """
    root = Path(toxric_extract_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist. Run download_toxric first.")
    return sorted(
        p for p in root.rglob("*.csv")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def inspect_csv(csv_path: str | os.PathLike, n_rows: int = 3) -> dict:
    """Return a dict describing a CSV's schema and a small sample.

    Output keys:
        path, n_rows_total, columns, dtypes, sample_head, smiles_column_guess
    The 'smiles_column_guess' is set if any column name looks like SMILES.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    smiles_guess = None
    for col in df.columns:
        if col.strip().lower() in {"smiles", "canonical_smiles", "canonical smiles"}:
            smiles_guess = col
            break

    return {
        "path": str(csv_path),
        "n_rows_total": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "sample_head": df.head(n_rows).to_dict(orient="records"),
        "smiles_column_guess": smiles_guess,
        "non_smiles_columns": [c for c in df.columns if c != smiles_guess],
    }


def print_inspection(info: dict) -> None:
    print(f"\n=== {info['path']} ===")
    print(f"rows: {info['n_rows_total']}")
    print(f"columns ({len(info['columns'])}): {info['columns']}")
    print(f"SMILES column guess: {info['smiles_column_guess']!r}")
    print(f"first {len(info['sample_head'])} rows:")
    for row in info["sample_head"]:
        print(f"  {row}")


# --- KPGT setup ----------------------------------------------------------

def clone_kpgt(target_dir: str | os.PathLike = "external/KPGT") -> Path:
    """Clone the lihan97/KPGT repo. Idempotent — skips if already cloned."""
    target = Path(target_dir)
    if target.exists() and (target / ".git").exists():
        print(f"[skip] KPGT already cloned at {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {KPGT_GIT_URL} -> {target}")
    subprocess.run(
        ["git", "clone", "--depth", "1", KPGT_GIT_URL, str(target)],
        check=True,
    )
    return target


def check_kpgt_pretrained(kpgt_dir: str | os.PathLike = "external/KPGT") -> bool:
    """Check whether base.pth is present at the expected location.

    The KPGT authors host the pretrained weights behind an anonymous
    Figshare share link (no public API), so this can't be automated
    without an interactive browser session. The user must download
    manually one time; this function tells them whether they've done it.
    """
    expected = Path(kpgt_dir) / KPGT_PRETRAINED_INFO["expected_path"]
    if expected.exists():
        size_mb = expected.stat().st_size / 1e6
        print(f"[ok] base.pth found at {expected} ({size_mb:.1f} MB)")
        return True

    print(
        "[missing] KPGT pretrained weights not found.\n"
        f"  1. Open: {KPGT_PRETRAINED_INFO['share_url']}\n"
        f"  2. Download the zip (~{KPGT_PRETRAINED_INFO['approx_size_mb']} MB).\n"
        f"  3. Unzip and place base.pth at: {expected}"
    )
    return False
