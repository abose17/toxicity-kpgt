"""One-shot Phase A setup. Run this once before opening the notebook.

Usage from labfiles/toxicity-kpgt/:
    python scripts/setup.py
"""

import sys
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toxpkg.data_utils import (
    check_kpgt_pretrained,
    clone_kpgt,
    download_toxric,
    list_toxric_subdatasets,
)


def main() -> int:
    print("=" * 60)
    print("Phase A setup: TOXRIC + KPGT")
    print("=" * 60)

    print("\n[1/3] Downloading TOXRIC 30-datasets archive...")
    extract_dir = download_toxric(which="toxric_30_datasets.zip")
    csvs = list_toxric_subdatasets(extract_dir)
    print(f"     Found {len(csvs)} CSV(s) under {extract_dir}")

    print("\n[2/3] Cloning KPGT repo...")
    clone_kpgt()

    print("\n[3/3] Checking pretrained KPGT checkpoint...")
    ok = check_kpgt_pretrained()

    print("\n" + "=" * 60)
    if ok:
        print("Setup complete. Open notebook.ipynb to inspect the schema.")
    else:
        print("Setup partially complete — download base.pth manually (see message above).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
