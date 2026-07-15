"""Tests for chem_highlighter.hml."""

from __future__ import annotations

import msgspec
import pytest

from chem_highlighter.hml import HML


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
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    assert doc.hml is None
    json_str = doc.to_hmol_json()
    decoded = msgspec.json.decode(json_str)
    assert b'"mol"' in msgspec.json.encode(decoded)


def test_to_hmol_json_with_hml() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode())
    assert doc.hml is not None
    json_str = doc.to_hmol_json()
    # Palette must be serialised into the JSON
    assert "ff0000" in json_str


def test_highlight_from_json() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode())
    assert doc.hml is not None
    assert doc.hml.palette == ["#ff0000"]
