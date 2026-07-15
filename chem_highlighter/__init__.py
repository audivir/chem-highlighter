"""Highlight molecules using different backends."""

from __future__ import annotations

from chem_highlighter import align, decomposer
from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.diff import colorize_smiles_diff, get_smiles_diff
from chem_highlighter.hml import HML, HighlightBackendDocument, HighlightBackendDocumentT_co, HMol

__all__ = [
    "HML",
    "HMol",
    "HighlightBackendDocument",
    "HighlightBackendDocumentT_co",
    "RDKitDocument",
    "align",
    "colorize_smiles_diff",
    "decomposer",
    "get_smiles_diff",
]
