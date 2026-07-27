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
    assert doc.get_hml() is None
    json_str = doc.to_hmol_json()
    decoded = msgspec.json.decode(json_str)
    assert b'"mol"' in msgspec.json.encode(decoded)


def test_to_hmol_json_with_hml() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    assert doc.get_hml() is not None
    json_str = doc.to_hmol_json()
    # Palette must be serialised into the JSON
    assert "ff0000" in json_str


def test_highlight_from_json() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    doc_hml = doc.get_hml()
    assert doc_hml is not None
    assert doc_hml.palette == ["#ff0000"]


def test_highlight_from_json_with_show_hydrogens_marks_hydrogen_display_set() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), True)
    assert doc.get_edit_state().hydrogen_display_set is True
    with pytest.raises(
        ValueError, match="Setting hydrogen display after highlighting not supported"
    ):
        doc.set_hydrogen_display(False)


def test_cleanup_succeeds() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    doc.cleanup()
    edit_state = doc.get_edit_state()
    assert edit_state.kekulized is None
    assert edit_state.aligned is False


def test_cleanup_raises_when_already_kekulized() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("c1ccccc1"))
    doc.kekulize(True)
    with pytest.raises(ValueError, match="Cleanup after kekulization or alignment not supported"):
        doc.cleanup()


def test_align_to_reference_raises_when_already_aligned() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    doc.align_to_reference(doc.to_molblock())
    with pytest.raises(ValueError, match="Already aligned"):
        doc.align_to_reference(doc.to_molblock())


def test_kekulize_raises_when_already_kekulized() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("c1ccccc1"))
    doc.kekulize(True)
    with pytest.raises(ValueError, match="Already kekulized"):
        doc.kekulize(False)


def test_set_hydrogen_display_raises_when_already_set() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CC"))
    doc.set_hydrogen_display(True)
    with pytest.raises(ValueError, match="Hydrogen display already set"):
        doc.set_hydrogen_display(False)


def test_set_hydrogen_display_raises_after_highlighting() -> None:
    """Adding/removing hydrogens shifts atom and bond indices.

    This would silently invalidate any highlights already set, so it must be rejected instead.
    """
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    with pytest.raises(
        ValueError, match="Setting hydrogen display after highlighting not supported"
    ):
        doc.set_hydrogen_display(True)


def test_highlight_from_json_raises_when_already_highlighted() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    with pytest.raises(ValueError, match="Already highlighted"):
        doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)


def test_highlight_from_json_raises_after_hydrogen_display_set() -> None:
    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import mol_from_smiles

    doc = RDKitDocument.from_mol(mol_from_smiles("CCO"))
    doc.set_hydrogen_display(True)
    hml = HML(highlighted_atoms={0: 0}, palette=["#ff0000"])
    with pytest.raises(
        ValueError, match="Highlighting after setting hydrogen display not supported"
    ):
        doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
