"""Tests for chem_highlighter.backend.rdkit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import msgspec
import pytest
from conftest import from_fixture_molblock
from rdkit import Chem

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.hml import HML
from chem_highlighter.utils import is_same_conformer, mol_from_smiles

if TYPE_CHECKING:
    from collections.abc import Sequence


def _doc(smiles: str) -> RDKitDocument:
    return RDKitDocument.from_mol(mol_from_smiles(smiles))


def test_from_mol() -> None:
    doc = _doc("CCO")
    assert doc.mol.GetNumAtoms() == 3
    assert doc.hml is None
    assert not doc._aligned  # noqa: SLF001
    assert not doc._kekulized  # noqa: SLF001


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
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
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
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
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
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    assert doc.to_console() == "C\033[38;2;0;255;0m=\033[0mCOCc1ccc\033[38;2;255;0;0m(C)\033[0mcc1"


def _v3000_bond_type_codes(molblock: str) -> list[int]:
    """Bond type codes (1=single, 2=double, 4=aromatic, ...) from a V3000 Mol block's BOND
    block, in file order -- the actual on-disk representation, not RDKit's in-memory bond
    objects (which is what a downstream consumer of the exported file, e.g. ChemDraw, sees).
    """
    codes = []
    in_bond_block = False
    for line in molblock.splitlines():
        stripped = line.strip()
        if stripped == "M  V30 BEGIN BOND":
            in_bond_block = True
        elif stripped == "M  V30 END BOND":
            in_bond_block = False
        elif in_bond_block:
            codes.append(int(line.split()[3]))
    return codes


@pytest.mark.parametrize("kekulize", [True, False])
def test_kekulize(kekulize: bool) -> None:
    doc = _doc("c1ccccc1")
    doc.kekulize(kekulize)
    assert doc._kekulized == kekulize  # noqa: SLF001

    ring_bonds = [doc.mol.GetBondBetweenAtoms(i, (i + 1) % 6) for i in range(6)]
    bond_type_codes = _v3000_bond_type_codes(doc.to_molblock())
    if kekulize:
        # `Chem.Kekulize` sets explicit SINGLE/DOUBLE bond types but, by default, does not clear
        # the separate `IsAromatic` bookkeeping flag on the bond -- that's RDKit's own intended
        # default behavior and doesn't leak into the exported file, so it's not asserted here.
        # What actually matters -- whether the *file* really shows a kekulized structure, the
        # same question this test failed to answer before -- is checked via the Mol block's own
        # bond type codes, which must NOT be 4 (aromatic) once kekulized.
        assert {bond.GetBondType() for bond in ring_bonds} == {
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
        }
        assert [bond.GetBondType() for bond in ring_bonds] == [
            Chem.BondType.SINGLE if i % 2 == 0 else Chem.BondType.DOUBLE for i in range(6)
        ]
        assert 4 not in bond_type_codes, "exported Mol block still has aromatic bond codes"
        assert set(bond_type_codes) == {1, 2}
    else:
        assert all(bond.GetIsAromatic() for bond in ring_bonds)
        assert all(bond.GetBondType() == Chem.BondType.AROMATIC for bond in ring_bonds)
        assert set(bond_type_codes) == {4}, "exported Mol block should use the aromatic bond code"


@pytest.mark.parametrize("show", [True, False])
def test_set_hydrogen_display(show: bool) -> None:
    doc = _doc("CC")
    heavy_count = doc.mol.GetNumAtoms()
    doc.set_hydrogen_display(show)
    if show:
        assert doc.mol.GetNumAtoms() > heavy_count
    else:
        assert doc.mol.GetNumAtoms() == heavy_count


def test_set_hydrogen_display_round_trip_returns_to_heavy_atom_count() -> None:
    doc = _doc("CC")
    heavy_count = doc.mol.GetNumAtoms()
    doc.set_hydrogen_display(True)
    assert doc.mol.GetNumAtoms() > heavy_count
    doc.set_hydrogen_display(False)
    assert doc.mol.GetNumAtoms() == heavy_count


def test_set_hydrogen_display_hides_a_hydrogen_that_was_already_explicit() -> None:
    """A hydrogen already explicit in the input (not just newly shown) must also be hidden.

    `set_hydrogen_display_callback(False)` -> `Chem.RemoveHs` -- confirm it doesn't only remove
    the hydrogens *it* added via a prior `show`, but any plain explicit hydrogen already present
    on the molecule (e.g. from a Mol block written with an explicit H atom).
    """
    from rdkit import Chem

    mol = mol_from_smiles("CC")
    heavy_count = mol.GetNumAtoms()
    # Materialize atom 0's implicit hydrogens as real, explicit atoms -- simulating input that
    # already had an explicit hydrogen before `set_hydrogen_display` is ever called.
    mol_with_explicit_h = Chem.AddHs(mol, onlyOnAtoms=[0])
    assert mol_with_explicit_h.GetNumAtoms() > heavy_count

    doc = RDKitDocument.from_mol(mol_with_explicit_h)
    doc.set_hydrogen_display(False)
    assert doc.mol.GetNumAtoms() == heavy_count

    doc.set_hydrogen_display(True)
    doc.set_hydrogen_display(False)
    assert doc.mol.GetNumAtoms() == heavy_count


def test_cleanup() -> None:
    doc = _doc("c1ccccc1")
    doc.cleanup()
    assert doc.mol.GetNumAtoms() == 6


def test_cleanup_after_kekulize_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    doc = _doc("c1ccccc1")
    doc.kekulize(True)
    with caplog.at_level(logging.WARNING, logger="chem_highlighter.backend.rdkit"):
        doc.cleanup()
    assert "Kekulization" in caplog.text


def test_cleanup_after_align_logs_warning(caplog: pytest.LogCaptureFixture) -> None:

    doc = _doc("CCO")
    ref = _doc("CCO")
    doc.align_to_reference(ref.to_molblock())
    with caplog.at_level(logging.WARNING, logger="chem_highlighter.backend.rdkit"):
        doc.cleanup()
    assert "Alignment" in caplog.text


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
