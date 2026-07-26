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
    hml = HML(
        highlighted_atoms={0: 0, 1: 0},
        highlighted_bonds={0: 0},
        palette=["#ff0000"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    svg = doc.to_svg()
    assert "<svg" in svg


def test_to_svg_with_ring_highlights() -> None:
    doc = _doc("c1ccccc1")
    hml = HML(
        highlighted_atoms={0: 0},
        highlighted_rings={0: 0},
        rings=[[0, 1, 2, 3, 4, 5]],
        palette=["#ff0000"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), None)
    svg = doc.to_svg()
    assert "<svg" in svg


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


@pytest.mark.parametrize("kekulize", [True, False])
def test_kekulize(kekulize: bool) -> None:
    doc = _doc("c1ccccc1")
    doc.kekulize(kekulize)
    assert doc._kekulized == kekulize  # noqa: SLF001


@pytest.mark.parametrize("show", [True, False])
def test_set_hydrogen_display(show: bool) -> None:
    doc = _doc("CC")
    heavy_count = doc.mol.GetNumAtoms()
    doc.set_hydrogen_display(show)
    if show:
        assert doc.mol.GetNumAtoms() > heavy_count
    else:
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
        ("ethanol.mol", ["_hf"]),
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
