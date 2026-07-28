"""Configuration for chem-highlighter tests."""

from __future__ import annotations

import msgspec
import pytest

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.hml import HML, HighlightBackendDocument
from chem_highlighter.utils import mol_from_smiles


@pytest.fixture
def doc() -> HighlightBackendDocument:
    return RDKitDocument.from_mol(mol_from_smiles("c1ccccc1"))


@pytest.fixture
def hml_json() -> str:
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    return msgspec.json.encode(hml).decode()
