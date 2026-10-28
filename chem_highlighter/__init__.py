"""Provide functionality to highlight molecules using various backends."""

from __future__ import annotations

from chem_highlighter import align, decomposer
from chem_highlighter.backend.rdkit import (
    AsyncRDKitDocument,
    AsyncRDKitMolecule,
    RDKitDocument,
    RDKitMolecule,
)
from chem_highlighter.diff import colorize_smiles_diff, get_smiles_diff
from chem_highlighter.hml import (
    HML,
    AsyncDocument,
    AsyncHighlightBackendMolecule,
    AsyncHighlightBackendMoleculeT_co,
    Document,
    HighlightBackendMolecule,
    HighlightBackendMoleculeT_co,
)

__all__ = [
    "HML",
    "AsyncDocument",
    "AsyncHighlightBackendMolecule",
    "AsyncHighlightBackendMoleculeT_co",
    "AsyncRDKitDocument",
    "AsyncRDKitMolecule",
    "Document",
    "HighlightBackendMolecule",
    "HighlightBackendMoleculeT_co",
    "RDKitDocument",
    "RDKitMolecule",
    "align",
    "colorize_smiles_diff",
    "decomposer",
    "get_smiles_diff",
]

__version__ = "0.3.4"
