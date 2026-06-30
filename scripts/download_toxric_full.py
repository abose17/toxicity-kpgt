"""Download all 1,474 TOXRIC endpoints from the web API and merge into one CSV.

Discovered via browser DevTools on 2026-06-28:
  List API:  GET /jk/DownloadController/getCategoryDetailedInfo?pageNo=N&pageSize=100&pid=PID&search=&type=1
  Download:  GET /jk/DownloadController/DownloadToxicityInfo?toxicityId=ID  (returns TSV)

TSV columns: TAID, Pubchem CID, IUPAC Name, SMILES, Canonical SMILES, InChIKey, <endpoint>

Usage from labfiles/toxicity-kpgt/:
    python scripts/download_toxric_full.py
    python scripts/download_toxric_full.py --raw-dir data/toxric_full/raw --output data/toxric_full_merged.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://toxric.bioinforai.tech/jk/DownloadController"
LIST_API = BASE_URL + "/getCategoryDetailedInfo"
DOWNLOAD_API = BASE_URL + "/DownloadToxicityInfo"

SMILES_COL = "Canonical SMILES"
DELAY = 0.3  # seconds between requests — be polite to the server


# ---------------------------------------------------------------------------
# Phase 1: discover all active category PIDs
# ---------------------------------------------------------------------------

def discover_pids(pid_min: int = 1, pid_max: int = 400) -> list[int]:
    """Scan PID range and return those with at least one endpoint."""
    print(f"\n[1/3] Scanning PIDs {pid_min}–{pid_max} for active categories...")
    active = []
    for pid in tqdm(range(pid_min, pid_max + 1), unit="pid"):
        try:
            r = requests.get(
                LIST_API,
                params={"pageNo": 1, "pageSize": 1, "pid": pid, "search": "", "type": 1},
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json().get("data", {})
                if data.get("total", 0) > 0:
                    category = data["list"][0].get("toxicityCategory", "?")
                    active.append(pid)
                    tqdm.write(f"  pid={pid:4d}  total={data['total']:5d}  {category}")
        except Exception as e:
            tqdm.write(f"  pid={pid} error: {e}")
        time.sleep(DELAY)
    print(f"  Found {len(active)} active PIDs: {active}")
    return active


# ---------------------------------------------------------------------------
# Phase 2: collect all endpoint records across all active PIDs
# ---------------------------------------------------------------------------

def collect_endpoints(pids: list[int]) -> list[dict]:
    """Page through getCategoryDetailedInfo for each PID and return all endpoint records."""
    print(f"\n[2/3] Collecting endpoint records for {len(pids)} categories...")
    all_endpoints: list[dict] = []
    seen_ids: set[int] = set()

    for pid in pids:
        page = 1
        while True:
            try:
                r = requests.get(
                    LIST_API,
                    params={"pageNo": page, "pageSize": 100, "pid": pid, "search": "", "type": 1},
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()["data"]
            except Exception as e:
                print(f"  pid={pid} page={page} error: {e}")
                break

            for item in data["list"]:
                eid = item["id"]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    all_endpoints.append({
                        "id": eid,
                        "pid": item["pid"],
                        "category": item["category"],
                        "toxicityCategory": item.get("toxicityCategory", ""),
                        "benchmarkTask": item.get("benchmarkTask", ""),
                        "num": item.get("num", 0),
                    })

            if data["isLastPage"]:
                break
            page += 1
            time.sleep(DELAY)

    print(f"  Collected {len(all_endpoints)} unique endpoints")
    return all_endpoints


# ---------------------------------------------------------------------------
# Phase 3: download each endpoint TSV (resumable)
# ---------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def download_endpoints(endpoints: list[dict], raw_dir: Path) -> list[Path]:
    """Download each endpoint TSV. Skips files already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] Downloading {len(endpoints)} endpoint files to {raw_dir} ...")

    downloaded: list[Path] = []
    skipped = 0
    failed: list[int] = []

    for ep in tqdm(endpoints, unit="file"):
        fname = f"{ep['id']:05d}_{safe_filename(ep['toxicityCategory'])}_{safe_filename(ep['category'])}.tsv"
        dest = raw_dir / fname

        if dest.exists() and dest.stat().st_size > 100:
            skipped += 1
            downloaded.append(dest)
            continue

        try:
            r = requests.get(
                DOWNLOAD_API,
                params={"toxicityId": ep["id"]},
                timeout=60,
                stream=True,
            )
            r.raise_for_status()
            content = r.content
            if len(content) < 50:
                tqdm.write(f"  [skip-empty] id={ep['id']} {ep['category']}")
                failed.append(ep["id"])
                continue
            dest.write_bytes(content)
            downloaded.append(dest)
        except Exception as e:
            tqdm.write(f"  [error] id={ep['id']} {ep['category']}: {e}")
            failed.append(ep["id"])

        time.sleep(DELAY)

    print(f"  Downloaded: {len(downloaded) - skipped}  Skipped (cached): {skipped}  Failed: {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed}")
    return downloaded


# ---------------------------------------------------------------------------
# Phase 4: merge all TSVs into one wide CSV
# ---------------------------------------------------------------------------

def _endpoint_name_from_filename(path: Path) -> str:
    """'00012_Carcinogenicity_Carcinogenicity.tsv' -> 'Carcinogenicity_Carcinogenicity'"""
    # strip leading ID prefix (digits + underscore)
    stem = path.stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else stem


def merge_tsv_files(tsv_files: list[Path], output: Path) -> pd.DataFrame:
    """Outer-join all endpoint files on Canonical SMILES.

    Files are comma-separated despite the .tsv extension.
    The label column is always 'Toxicity Value' — renamed to the endpoint name.
    """
    print(f"\n[4/4] Merging {len(tsv_files)} files on '{SMILES_COL}' ...")
    merged: pd.DataFrame | None = None
    failed_parse = []
    skipped_no_smiles = 0

    for tsv in tqdm(tsv_files, unit="file"):
        try:
            df = pd.read_csv(tsv, sep=",", low_memory=False)
        except Exception as e:
            tqdm.write(f"  [parse-error] {tsv.name}: {e}")
            failed_parse.append(tsv.name)
            continue

        if SMILES_COL not in df.columns:
            skipped_no_smiles += 1
            continue

        # Rename the generic 'Toxicity Value' column to the endpoint name
        endpoint = _endpoint_name_from_filename(tsv)
        label_col = "Toxicity Value"
        if label_col not in df.columns:
            # fallback: any column that isn't metadata
            extras = [c for c in df.columns if c not in {
                "TAID", "Name", "IUPAC Name", "PubChem CID", SMILES_COL, "InChIKey"
            }]
            if not extras:
                skipped_no_smiles += 1
                continue
            label_col = extras[0]

        slim = (
            df[[SMILES_COL, label_col]]
            .dropna(subset=[SMILES_COL])
            .rename(columns={label_col: endpoint})
            .groupby(SMILES_COL, as_index=False)[endpoint]
            .first()
        )

        merged = slim if merged is None else merged.merge(slim, on=SMILES_COL, how="outer")

    if skipped_no_smiles:
        print(f"  Skipped (no SMILES col): {skipped_no_smiles}")

    if merged is None:
        raise RuntimeError("No valid TSV files could be parsed.")

    merged = merged.rename(columns={SMILES_COL: "smiles"})
    cols = ["smiles"] + [c for c in merged.columns if c != "smiles"]
    merged = merged[cols]

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    n = len(merged)
    n_ep = len(cols) - 1
    print(f"\n  Wrote {output}")
    print(f"  Unique compounds : {n:,}")
    print(f"  Endpoints        : {n_ep}")
    if failed_parse:
        print(f"  Parse failures   : {len(failed_parse)}")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/toxric_full/raw",
                   help="Directory to store per-endpoint TSV files.")
    p.add_argument("--output", default="data/toxric_full_merged.csv",
                   help="Path for the final merged CSV.")
    p.add_argument("--pid-min", type=int, default=1, help="PID scan start (default 1).")
    p.add_argument("--pid-max", type=int, default=400, help="PID scan end (default 400).")
    p.add_argument("--skip-discover", nargs="+", type=int, metavar="PID",
                   help="Skip PID discovery and use these PIDs directly.")
    p.add_argument("--download-only", action="store_true",
                   help="Skip merge step — only download TSVs.")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip download — merge existing TSVs in --raw-dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output = Path(args.output)

    if args.merge_only:
        tsv_files = sorted(raw_dir.glob("*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"No TSV files found in {raw_dir}")
        merge_tsv_files(tsv_files, output)
        return

    if args.skip_discover:
        pids = args.skip_discover
        print(f"Using provided PIDs: {pids}")
    else:
        pids = discover_pids(args.pid_min, args.pid_max)

    endpoints = collect_endpoints(pids)

    tsv_files = download_endpoints(endpoints, raw_dir)

    if not args.download_only:
        merge_tsv_files(tsv_files, output)


if __name__ == "__main__":
    main()
