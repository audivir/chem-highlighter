"""Tests for chem_highlighter.backend.rdkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import numpy as np
import pytest
from conftest import assert_mols_equal, extract_bond_codes, from_fixture_molblock
from rdkit import Chem

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.hml import HML
from chem_highlighter.utils import is_same_conformer, mol_from_smiles, move_molecule

if TYPE_CHECKING:
    from collections.abc import Sequence


def _doc(smiles: str) -> RDKitDocument:
    return RDKitDocument.from_mol(mol_from_smiles(smiles))


def _mol_from_explicit_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string without collapsing its explicit hydrogen atoms to implicit ones."""
    params = Chem.SmilesParserParams()
    params.removeHs = False  # type: ignore[assignment]
    return Chem.MolFromSmiles(smiles, params)


def test_from_mol() -> None:
    doc = _doc("CCO")
    assert doc.mol.GetNumAtoms() == 3
    assert doc.get_hml_json() is None
    assert doc.get_edit_state() == (True, False, False)


def test_from_molblock() -> None:
    molblock = Chem.MolToMolBlock(_doc("CCO").mol)
    doc = RDKitDocument.from_molblock(molblock)
    assert doc.mol.GetNumAtoms() == 3


def test_convert_molblock() -> None:
    molblock = Chem.MolToMolBlock(_doc("CCO").mol)
    mol = RDKitDocument.convert_molblock(molblock)
    assert mol.GetNumAtoms() == 3


def test_to_molblock() -> None:
    molblock = _doc("CCO").to_molblock()
    assert "V3000" in molblock


def test_to_svg() -> None:
    svg = _doc("CCO").to_svg()
    assert "<svg" in svg


def test_to_svg_with_atom_and_bond_highlights() -> None:
    doc = _doc("CCO")
    before = doc.to_svg()

    hml = HML(
        highlighted_atoms={0: 0, 1: 0},
        highlighted_bonds={0: 0},
        palette=["#ff0000"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    svg = doc.to_svg()
    assert "<svg" in svg
    assert svg != before, "highlighting should change the rendered SVG"
    assert "ff0000" in svg.lower(), "the highlight color should actually appear in the SVG"


def test_to_svg_with_ring_highlights() -> None:
    doc = _doc("c1ccccc1")
    before = doc.to_svg()

    hml = HML(
        highlighted_atoms={0: 0},
        highlighted_rings={0: 0},
        rings=[[0, 1, 2, 3, 4, 5]],
        palette=["#ff0000"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    svg = doc.to_svg()
    assert "<svg" in svg
    assert svg != before, "highlighting should change the rendered SVG"
    assert "ff0000" in svg.lower(), "the highlight color should actually appear in the SVG"


def test_to_png() -> None:
    png = _doc("CCO").to_png()
    assert png[:4] == b"\x89PNG"


def test_to_console() -> None:
    doc = _doc("C=COCc1ccc(C)cc1")
    hml = HML(
        highlighted_atoms={8: 0},
        highlighted_bonds={0: 1},
        palette=["#ff0000", "#00ff00"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    assert doc.to_console() == "C\033[38;2;0;255;0m=\033[0mCOCc1ccc\033[38;2;255;0;0m(C)\033[0mcc1"


def assert_benzene_kekulized(doc: RDKitDocument, kekulized: bool) -> None:
    assert doc.get_edit_state() == (kekulized, False, False)
    ring_bonds = [doc.mol.GetBondBetweenAtoms(i, (i + 1) % 6) for i in range(6)]
    ring_bond_types = [b.GetBondType() for b in ring_bonds]
    bond_type_codes = extract_bond_codes(doc.to_molblock())
    kekule_bonds = [
        [Chem.BondType.SINGLE if i % 2 == modulo else Chem.BondType.DOUBLE for i in range(6)]
        for modulo in (0, 1)
    ]

    assert all(bond.GetIsAromatic() for bond in ring_bonds)

    if kekulized:
        assert set(ring_bond_types) == {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}
        assert ring_bond_types in kekule_bonds
        assert 4 not in bond_type_codes, "exported Mol block still has aromatic bond codes"
        assert set(bond_type_codes) == {1, 2}
    else:
        assert set(ring_bond_types) == {Chem.BondType.AROMATIC}
        assert set(bond_type_codes) == {4}, "exported Mol block should use the aromatic bond code"


def test_kekulize() -> None:
    doc = _doc("c1ccccc1")
    assert_benzene_kekulized(doc, True)
    doc.kekulize(False)
    assert_benzene_kekulized(doc, False)
    doc.kekulize(True)
    assert_benzene_kekulized(doc, True)


@pytest.mark.parametrize(
    ("base", "hide"),
    [
        ("CC", "CC"),
        ("C([2H])C[H]", "[2H]CC"),
        ("c1ccn([H])c1", "c1cc[nH]c1"),
        ("[H]c1n([H])c([H])c([H])c1[H]", "c1cc[nH]c1"),
        ("c1cc([2H])n([H])c1", "[2H]c1ccc[nH]1"),
    ],
)
def test_hide_hydrogens_smiles_table(base: str, hide: str) -> None:
    doc = RDKitDocument.from_mol(mol_from_smiles(base))
    doc.hide_hydrogens_callback()
    assert_mols_equal(doc.mol, hide)


def test_hide_hydrogens_mol() -> None:
    doc = RDKitDocument.from_mol(from_fixture_molblock("hs_mol_base.mol"))
    doc.hide_hydrogens()
    assert_mols_equal(doc.mol, from_fixture_molblock("hs_mol_hide.mol"))


def test_hide_hydrogens_keeps_a_highlighted_hydrogen(hml_json: str) -> None:
    doc = RDKitDocument.from_mol(_mol_from_explicit_smiles("[H]C([2H])C[H]"))
    doc.hide_hydrogens_callback()
    assert doc.mol.GetNumAtoms() == 3  # only 2 Cs and [2H]

    doc = RDKitDocument.from_mol(_mol_from_explicit_smiles("[H]C([2H])C[H]"))
    doc.set_hml_json(hml_json)
    doc.hide_hydrogens_callback()
    assert doc.mol.GetNumAtoms() == 4  # 2 Cs, [2H], and highlighted [H]


def test_cleanup() -> None:
    atol = 1e-5

    doc = _doc("c1cnccc1")
    move_molecule(doc.mol, np.array([1.0, 1.0, 1.0]))

    expected = _doc("c1cnccc1")
    assert not is_same_conformer(doc.to_molblock(), expected.to_molblock(), atol=atol)

    doc.cleanup()
    assert is_same_conformer(doc.to_molblock(), expected.to_molblock(), atol=atol)


def test_cleanup_after_raises() -> None:
    doc = _doc("c1ccccc1")
    doc.kekulize(True)

    with pytest.raises(ValueError, match="Cleanup after kekulization or alignment not supported"):
        doc.cleanup()

    doc = _doc("c1ccccc1")
    doc.align_to_reference(doc.to_molblock())
    with pytest.raises(ValueError, match="Cleanup after kekulization or alignment not supported"):
        doc.cleanup()


@pytest.mark.parametrize(
    ("r_file", "q_file_suffixes"),
    [
        ("3-methylbutanone.mol", ["_44cw", "_b12", "_bf", "_hf", "_vf"]),
        ("acetylic_acid.mol", ["_44cw", "_b00", "_bf", "_hf", "_vf"]),
        (
            "mol.mol",
            [
                "_44cw",
                "_b66",
                "_b76",
                "_b86",
                "_b98",
                "_bf",
                "_hf",
                "_vf",
                "_multi_b66_b98_hf_22cw",
            ],
        ),
        ("ethanol.mol", ["_44cw", "_bf", "_hf", "_vf"]),
    ],
)
def test_align_to_reference(r_file: str, q_file_suffixes: Sequence[str]) -> None:
    r = from_fixture_molblock(r_file)
    for q_file_suffix in q_file_suffixes:
        q = from_fixture_molblock(r_file.removesuffix(".mol") + f"{q_file_suffix}.mol")
        for doc_mol, ref_mol in ((q, r), (r, q)):
            doc = RDKitDocument.from_mol(doc_mol)
            ref = RDKitDocument.from_mol(ref_mol)
            doc.align_to_reference(ref.to_molblock())
            assert is_same_conformer(doc.to_molblock(), ref.to_molblock(), atol=1e-5)
