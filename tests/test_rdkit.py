"""Tests for chem_highlighter.backend.rdkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest
from backend_test import (
    ALIGNMENTS,
    HIDE_HYDROGENS,
    assert_align_to_reference,
    assert_cleanup,
    assert_doc_init,
    assert_export,
    assert_export_images,
    assert_export_mdl,
    assert_export_png_respects_configured_size,
    assert_export_unsupported_format,
    assert_from_bytes,
    assert_from_bytes_cdxml,
    assert_from_bytes_unsupported_format,
    assert_hide_hydrogens_special,
    assert_hide_hydrogens_table,
    assert_highlight_from_json,
    assert_kekulize,
)

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.hml import HML, InputFormat, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


def test_doc() -> None:
    assert_doc_init(RDKitDocument)


@pytest.mark.parametrize("fmt", ["SDF", "Mol"])
@pytest.mark.parametrize("use_v2000", [False, True])
def test_export_mdl(fmt: Literal["SDF", "Mol", "RXN"], use_v2000: bool) -> None:
    assert_export_mdl(fmt, use_v2000, RDKitDocument)


@pytest.mark.parametrize("fmt", ["SVG", "PNG"])
def test_export_images(fmt: Literal["SVG", "PNG", "EPS"]) -> None:
    assert_export_images(fmt, RDKitDocument)


def test_export_png_respects_configured_size() -> None:
    assert_export_png_respects_configured_size(RDKitDocument)


@pytest.mark.parametrize("fmt", ["SMILES", "InChI", "InChIKey"])
def test_export(fmt: Literal["SMILES", "CDX", "CDXML", "InChI", "InChIKey"]) -> None:
    assert_export(fmt, RDKitDocument)


@pytest.mark.parametrize("fmt", ["RXN", "CDX", "CDXML", "EPS"])
def test_export_unsupported_format(fmt: OutputFormat) -> None:
    assert_export_unsupported_format(fmt, RDKitDocument)


@pytest.mark.parametrize("fmt", ["SDF", "Mol", "SMILES", "InChI"])
def test_from_bytes(fmt: InputFormat) -> None:
    assert_from_bytes(fmt, RDKitDocument)


def test_from_bytes_cdxml() -> None:
    assert_from_bytes_cdxml(RDKitDocument)


@pytest.mark.parametrize("fmt", ["RXN", "CDX"])
def test_from_bytes_unsupported_format(fmt: Literal["RXN", "CDX"]) -> None:
    assert_from_bytes_unsupported_format(fmt, RDKitDocument)


@pytest.mark.parametrize(("r_file", "q_file_suffixes"), ALIGNMENTS)
def test_align_to_reference(r_file: str, q_file_suffixes: Sequence[str]) -> None:
    assert_align_to_reference(r_file, q_file_suffixes, RDKitDocument)


def test_cleanup() -> None:
    doc = RDKitDocument.from_string("c1cnccc1", "SMILES")
    assert_cleanup(doc.to_molblock(), RDKitDocument)


def test_kekulize() -> None:
    assert_kekulize(RDKitDocument)


@pytest.mark.parametrize(
    ("base", "hide"),
    HIDE_HYDROGENS,
)
def test_hide_hydrogens_table(base: str, hide: str) -> None:
    assert_hide_hydrogens_table(base, hide, RDKitDocument)


def test_hide_hydrogens_special(hml_json: str) -> None:
    assert_hide_hydrogens_special(hml_json, RDKitDocument)


def test_highlight_from_json(hml_json: str) -> None:
    assert_highlight_from_json(hml_json, RDKitDocument)


def test_to_console() -> None:
    doc = RDKitDocument.from_string("C=COCc1ccc(C)cc1", "SMILES")
    doc.kekulize(False)
    hml = HML(
        highlighted_atoms={8: 0},
        highlighted_bonds={0: 1},
        palette=["#ff0000", "#00ff00"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    assert (
        doc.to_console() == "C\033[38;2;0;255;0m=\033[0mCOCc1ccc\033[38;2;255;0;0m(C)\033[0mcc1"
    )


def test_high_precision_v3000_export_survives_reload() -> None:
    """get_high_precision_v3000's extra decimal places aren't just cosmetic padding.

    Mirrors the equivalent precision check on the Rust side: the plain V3000 export rounds
    coordinates to 6 decimal places; the high-precision one carries the conformer's own unrounded
    floats. Checked by actually reloading both back into RDKit and comparing conformer positions,
    not just by inspecting decimal digit counts in the text.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    from chem_highlighter.utils import get_high_precision_v3000

    mol = Chem.MolFromSmiles("c1ccccc1O")
    rdDepictor.Compute2DCoords(mol)
    Chem.Kekulize(mol)

    default = Chem.MolToMolBlock(mol, forceV3000=True)
    high_precision = get_high_precision_v3000(mol)
    assert default != high_precision

    mol_default = Chem.MolFromMolBlock(default)
    mol_hp = Chem.MolFromMolBlock(high_precision)
    conf_default = mol_default.GetConformer()
    conf_hp = mol_hp.GetConformer()

    original_conf = mol.GetConformer()
    saw_a_real_difference = False
    for atom_ix in range(mol.GetNumAtoms()):
        original_pos = original_conf.GetAtomPosition(atom_ix)
        default_pos = conf_default.GetAtomPosition(atom_ix)
        hp_pos = conf_hp.GetAtomPosition(atom_ix)

        # The high-precision reload should always be at least as close to the true original as
        # the plain rounded reload -- and strictly closer for at least one atom, confirming the
        # extra digits actually reached the reloaded conformer.
        assert abs(hp_pos.x - original_pos.x) <= abs(default_pos.x - original_pos.x)
        assert abs(hp_pos.y - original_pos.y) <= abs(default_pos.y - original_pos.y)
        if hp_pos.x != default_pos.x or hp_pos.y != default_pos.y:
            saw_a_real_difference = True
    assert saw_a_real_difference


def test_molfromvolblock_import_is_bit_exact() -> None:
    """RDKit's V3000 parser has no precision floor of its own.

    Unlike some backends' V3000 importers, which impose their own precision ceiling
    independent of the file format's own limits, whatever decimal value is in the text comes
    back out of the conformer bit-for-bit.
    """
    from rdkit import Chem

    x, y = 0.123456789012345, -0.987654321098765
    molblock = (
        "\n  Test\n\n  0  0  0     0  0              0 V3000\n"
        "M  V30 BEGIN CTAB\nM  V30 COUNTS 1 0 0 0 0\nM  V30 BEGIN ATOM\n"
        f"M  V30 1 C {x!r} {y!r} 0.0 0\nM  V30 END ATOM\nM  V30 END CTAB\nM  END\n"
    )
    mol = Chem.MolFromMolBlock(molblock)
    pos = mol.GetConformer().GetAtomPosition(0)
    assert pos.x == x
    assert pos.y == y
