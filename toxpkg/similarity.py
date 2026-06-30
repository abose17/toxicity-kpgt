"""
ChEMBL REST API similarity search: SMILES → structurally similar molecules.

Verified API shape (2026-06-20):
  GET https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{threshold}.json?limit=N
  Response: {"molecules": [...], "page_meta": {...}}
  Per molecule: molecule_structures.canonical_smiles, molecule_chembl_id,
                pref_name, similarity (str float 0-100)

PubChem name lookup verified (2026-06-20):
  GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName,Title/JSON
  Response: PropertyTable.Properties[0].Title  (common name)
            PropertyTable.Properties[0].IUPACName
"""

from __future__ import annotations

import urllib.parse

import requests

_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{threshold}.json"
_PUBCHEM_PROPS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName,Title/JSON"


def fetch_similar_chembl(
    smiles: str,
    threshold: float = 0.7,
    n: int = 20,
    timeout: int = 30,
) -> list[dict]:
    """Query ChEMBL similarity search and return up to n results.

    Args:
        smiles:    Query SMILES string.
        threshold: Tanimoto similarity cutoff in [0, 1].
        n:         Maximum number of results to return.
        timeout:   HTTP timeout in seconds.

    Returns:
        List of dicts [{smiles, similarity, chembl_id, name}] sorted by similarity desc.
        similarity is normalised to [0, 1]. name is pref_name from ChEMBL (may be empty).
    """
    threshold_int = max(1, min(100, int(threshold * 100)))
    encoded_smiles = urllib.parse.quote(smiles, safe="")
    url = _CHEMBL_BASE.format(smiles=encoded_smiles, threshold=threshold_int)

    resp = requests.get(url, params={"limit": n, "offset": 0}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    results: list[dict] = []
    for mol in data.get("molecules", []):
        canonical = (mol.get("molecule_structures") or {}).get("canonical_smiles")
        if not canonical:
            continue
        results.append({
            "smiles": canonical,
            "similarity": float(mol["similarity"]) / 100.0,
            "chembl_id": mol.get("molecule_chembl_id", ""),
            "name": mol.get("pref_name") or "",
        })

    return sorted(results, key=lambda x: -x["similarity"])


def lookup_name_pubchem(smiles: str, timeout: int = 15) -> str:
    """Look up the common name for a SMILES string via PubChem.

    Returns the Title (common/trade name) if found, falling back to IUPACName,
    or empty string on any error.
    """
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = _PUBCHEM_PROPS.format(smiles=encoded)
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return ""
        props = resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
        return props.get("Title") or props.get("IUPACName") or ""
    except Exception:
        return ""
