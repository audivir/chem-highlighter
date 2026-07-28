"""Tests for chem_highlighter.hml."""

from __future__ import annotations

import msgspec
import pytest

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.hml import HML, HighlightBackendDocument
from chem_highlighter.utils import mol_from_smiles


@pytest.mark.parametrize(
    ("atoms", "bonds", "rings", "rings_ixs", "n_atom_groups", "n_bond_groups", "n_ring_groups"),
    [
        ({0: [(1.0, 0.0, 0.0, 1.0)]}, {}, {}, [], 1, 0, 0),
        ({}, {0: [(0.0, 1.0, 0.0, 1.0)]}, {}, [], 0, 1, 0),
        ({}, {}, {0: [(0.0, 0.0, 1.0, 1.0)]}, [[0, 1, 2]], 0, 0, 1),
        (
            {0: [(1.0, 0.0, 0.0, 1.0)], 1: [(1.0, 0.0, 0.0, 1.0)]},
            {0: [(1.0, 0.0, 0.0, 1.0)]},
            {},
            [],
            2,
            1,
            0,
        ),
    ],
)
def test_from_multicolor(
    atoms: dict[int, list[tuple[float, float, float, float]]],
    bonds: dict[int, list[tuple[float, float, float, float]]],
    rings: dict[int, list[tuple[float, float, float, float]]],
    rings_ixs: list[list[int]],
    n_atom_groups: int,
    n_bond_groups: int,
    n_ring_groups: int,
) -> None:
    hml = HML.from_multicolor(atoms, bonds, rings, rings_ixs)
    assert len(hml.highlighted_atoms) == n_atom_groups
    assert len(hml.highlighted_bonds) == n_bond_groups
    assert len(hml.highlighted_rings) == n_ring_groups
    assert len(hml.rings) == len(rings_ixs)


def test_get_rgba() -> None:
    hml = HML(palette=["#ff0000", "#00ff00"])
    r, g, b, a = hml.get_rgba(0)
    assert r == 1.0
    assert g == 0.0
    assert b == 0.0
    assert a == 1.0
    r2, g2, b2, a2 = hml.get_rgba(1)
    assert r2 == 0.0
    assert g2 == 1.0
    assert b2 == 0.0
    assert a2 == 1.0


def test_to_hmol_json_without_hml() -> None:
    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    assert doc.get_hml_json() is None
    json_str = doc.to_hmol_json()
    decoded = msgspec.json.decode(json_str)
    assert b'"mol"' in msgspec.json.encode(decoded)


def test_to_hmol_json_with_hml(doc: HighlightBackendDocument, hml_json: str) -> None:
    doc.highlight_from_json(hml_json, False)
    found_hml_json = doc.get_hml_json()
    assert found_hml_json is not None
    # Palette must be serialised into the JSON
    assert "ff0000" in found_hml_json


@pytest.mark.parametrize("hide_hydrogens", [False, True])
def test_highlight_from_json(
    hide_hydrogens: bool, doc: HighlightBackendDocument, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, hide_hydrogens)
    assert doc.get_edit_state() == (True, False, hide_hydrogens)
    found_hml_json = doc.get_hml_json()
    assert found_hml_json is not None
    found_hml = msgspec.json.Decoder(HML).decode(hml_json)
    assert found_hml.palette == ["#ff0000"]


def test_cleanup_succeeds() -> None:
    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    doc.cleanup()
    assert doc.get_edit_state() == (True, False, False)
