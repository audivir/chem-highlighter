"""Backend-independent test functions."""

from __future__ import annotations  # noqa: I001,RUF100

from typing import TYPE_CHECKING, Literal

import msgspec
import pytest
from utils import (
    assert_benzene_kekulized,
    assert_mols_equal,
    from_fixture_molblock,
    mol_from_explicit_smiles,
    png_dimensions,
    png_has_transparency,
    read_fixture,
    set_env,
)

from chem_highlighter.backend.rdkit import RDKitMolecule
from chem_highlighter.hml import (
    HML,
    HighlightBackendMoleculeT_co,
    InputFormat,
    InputFormatNotSupported,
    OutputFormat,
    OutputFormatNotSupported,
)
from chem_highlighter.utils import is_same_conformer, mol_to_smiles

if TYPE_CHECKING:
    from collections.abc import Sequence

ALIGNMENTS = [
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
]

HIDE_HYDROGENS = [
    ("CC", "CC"),
    ("C([2H])C[H]", "[2H]CC"),
    ("c1ccn([H])c1", "c1cc[nH]c1"),
    ("[H]c1n([H])c([H])c([H])c1[H]", "c1cc[nH]c1"),
    ("c1cc([2H])n([H])c1", "[2H]c1ccc[nH]1"),
]


def create_doc(
    smiles: str, backend: type[HighlightBackendMoleculeT_co]
) -> HighlightBackendMoleculeT_co:
    return backend.from_string(smiles, "SMILES")


def assert_doc_init(backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = create_doc("c1ccccc1", backend)
    assert doc.export_string("SMILES") == "C1=CC=CC=C1"  # kekulize on default
    assert doc.get_hml_json() is None  # non-highlighted
    assert doc.get_edit_state() == (True, False, False)  # set correct state


def assert_export_mdl(
    fmt: Literal["SDF", "Mol", "RXN"], use_v2000: bool, backend: type[HighlightBackendMoleculeT_co]
) -> None:
    if fmt == "RXN":
        raise NotImplementedError
    doc = create_doc("c1ccccc1", backend)
    expected = "C1=CC=CC=C1"
    output = doc.export_string(fmt, use_v2000)
    if use_v2000:
        assert "V2000" in output
    else:
        assert "V3000" in output
    endswith_dollars = output.strip().endswith("$$$$")
    assert endswith_dollars == (fmt == "SDF")
    assert backend.from_string(output, fmt).export_string("SMILES") == expected


def assert_hp_mol_v3000(atol: float, backend: type[HighlightBackendMoleculeT_co]) -> None:
    molblock = """\

     RDKit          2D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 2 1 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C -0.12727200 -0.39237500 0.00000000 0
M  V30 2 C 0.12727200 0.39237500 0.00000000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 END BOND
M  V30 END CTAB
M  END
"""
    doc = backend.from_molblock(molblock)

    assert is_same_conformer(doc.to_molblock(), molblock, atol)

    assert doc.from_molblock(doc.to_molblock()).to_molblock() == doc.to_molblock()


def assert_export_images(
    fmt: Literal["SVG", "PNG", "EPS"],
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    output = create_doc("c1ccccc1", backend).export(fmt)
    if fmt == "SVG":
        assert b"<svg" in output
    elif fmt == "PNG":
        assert b"\x89PNG" in output
        width, height = png_dimensions(output)
        assert width > 0
        assert height > 0
    elif fmt == "EPS":
        assert b"EPSF-1.2" in output


def assert_export_png_respects_environment(
    transparent: bool,
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    """Confirm CHEM_HIGHLIGHTER_PNG_WIDTH/HEIGHT actually reach the PNG renderer."""
    with set_env(
        CHEM_HIGHLIGHTER_PNG_WIDTH="137",
        CHEM_HIGHLIGHTER_PNG_HEIGHT="141",
        CHEM_HIGHLIGHTER_PNG_TRANSPARENT="1" if transparent else None,
    ):
        output = create_doc("c1ccccc1", backend).export("PNG")
    width, height = png_dimensions(output)
    assert width <= 137, f"width {width} exceeds the configured bound of 137"
    assert height <= 141, f"height {height} exceeds the configured bound of 141"
    assert max(width, height) >= 100, (
        f"expected a size close to the configured bound, got {width}x{height}"
    )
    assert png_has_transparency(output) == transparent


def assert_export(
    fmt: Literal["SMILES", "CDX", "CDXML", "InChI", "InChIKey"],
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    output = create_doc("c1ccccc1", backend).export(fmt)
    if fmt == "SMILES":
        assert output == b"C1=CC=CC=C1"
    elif fmt == "CDX":
        assert output.startswith(b"VjCD")
    elif fmt == "CDXML":
        assert b"</CDXML>" in output
    elif fmt == "InChI":
        assert output == b"InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    elif fmt == "InChIKey":
        assert output == b"UHOVQNZJYSORNB-UHFFFAOYSA-N"
    else:  # pragma: no cover
        raise ValueError("Unexpected format")


def assert_export_unsupported_format(
    fmt: OutputFormat,
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    with pytest.raises(
        OutputFormatNotSupported, match=f"{backend.__name__} does not support exporting to {fmt}"
    ):
        create_doc("c1ccccc1", backend).export(fmt)


def assert_from_bytes(
    fmt: InputFormat,
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    if fmt == "RXN":
        raise NotImplementedError
    exported = backend.from_string("c1ccccc1", "SMILES").export(fmt)
    doc = backend.from_bytes(exported, fmt)

    assert doc.export_string("SMILES") == "C1=CC=CC=C1"

    # empty input
    with pytest.raises((ValueError, RuntimeError), match=f"Invalid {fmt} input"):
        backend.from_bytes(b"", fmt)

    with pytest.raises((ValueError, RuntimeError), match=f"Invalid {fmt} input"):
        backend.from_bytes(b"invalid input", fmt)


def assert_from_bytes_cdxml(backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = backend.from_string(read_fixture("one_mol.cdxml"), "CDXML")
    assert doc.kekulize(False).export_string("SMILES") == "c1ccccc1"

    with pytest.raises((ValueError, RuntimeError), match="Invalid CDXML input"):
        backend.from_string(read_fixture("no_mol.cdxml"), "CDXML")

    with pytest.raises((ValueError, RuntimeError), match="Invalid CDXML input: Multiple groups"):
        backend.from_string(read_fixture("two_mols.cdxml"), "CDXML")


def assert_from_bytes_unsupported_format(
    fmt: InputFormat,
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    with pytest.raises(
        InputFormatNotSupported, match=f"{backend.__name__} does not support importing from {fmt}"
    ):
        backend.from_bytes(b"", fmt)


def assert_align_to_reference(
    r_file: str,
    q_file_suffixes: Sequence[str],
    atol: float,
    backend: type[HighlightBackendMoleculeT_co],
) -> None:
    r = from_fixture_molblock(r_file)
    for q_file_suffix in q_file_suffixes:
        q = from_fixture_molblock(r_file.removesuffix(".mol") + f"{q_file_suffix}.mol")
        for doc_mol, ref_mol in ((q, r), (r, q)):
            doc = backend.from_mol(doc_mol)
            ref = backend.from_mol(ref_mol)
            doc.align_to_reference(ref.to_molblock())
            assert is_same_conformer(doc.to_molblock(), ref.to_molblock(), atol=atol)


def assert_cleanup(expected: str, atol: float, backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = backend.from_molblock(read_fixture("pyrimidine_dirty.mol"))
    assert not is_same_conformer(doc.to_molblock(), expected, atol=atol)

    doc.cleanup()

    assert is_same_conformer(doc.to_molblock(), expected, atol=atol)


def assert_kekulize(backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = create_doc("c1ccccc1", backend)
    assert_benzene_kekulized(doc, True)
    doc.kekulize(False)
    assert_benzene_kekulized(doc, False)
    doc.kekulize(True)
    assert_benzene_kekulized(doc, True)


def assert_hide_hydrogens_table(
    base: str, hide: str, backend: type[HighlightBackendMoleculeT_co]
) -> None:
    doc = create_doc(base, backend)
    doc.hide_hydrogens()
    output = doc.kekulize(False).export_string("SMILES")
    canonical_output = mol_to_smiles(mol_from_explicit_smiles(output))
    assert canonical_output == hide


def assert_hide_hydrogens_special(
    hml_json: str, backend: type[HighlightBackendMoleculeT_co]
) -> None:
    doc = backend.from_molblock(read_fixture("hs_mol_base.mol"))
    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    with pytest.raises(AssertionError):
        assert_mols_equal(mol, from_fixture_molblock("hs_mol_hide.mol"))
    doc.hide_hydrogens()
    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    assert_mols_equal(mol, from_fixture_molblock("hs_mol_hide.mol"))

    doc = backend.from_mol(mol_from_explicit_smiles("[H]C([2H])C[H]"))
    doc.hide_hydrogens_callback()
    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    assert mol.GetNumAtoms() == 3  # only 2 Cs and [2H]

    rdkit_doc = RDKitMolecule.from_mol(mol_from_explicit_smiles("[H]C([2H])C[H]"))
    doc = backend.from_molblock(rdkit_doc.to_molblock())
    doc.highlight_from_json(hml_json, True)
    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    assert mol.GetNumAtoms() == 4  # 2 Cs, [2H], and highlighted [H]


def assert_highlight_from_json(hml_json: str, backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = create_doc("[H]CCO[2H]", backend)
    before = doc.to_svg()

    doc.highlight_from_json(hml_json, False)
    svg = doc.to_svg()

    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    assert mol.GetNumAtoms() == 5  # 2 Cs, [2H], O, and highlighted [H]
    assert "<svg" in svg
    assert svg != before, "highlighting should change the rendered SVG"
    assert "ff0000" in svg.lower(), "the highlight color should actually appear in the SVG"

    doc = create_doc("[H]CCO[2H]", backend)
    doc.highlight_from_json(hml_json, True)
    svg_with_hidden = doc.to_svg()

    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    assert mol.GetNumAtoms() == 5  # 2 Cs, [2H], O, and highlighted [H]
    assert svg == svg_with_hidden

    doc = create_doc("[H]CCO[2H]", backend)
    doc.hide_hydrogens()

    mol = RDKitMolecule.from_molblock(doc.to_molblock()).mol
    assert mol.GetNumAtoms() == 4  # 2 Cs, [2H], O

    doc = create_doc("c1ccccc1", backend)
    hml = HML(
        highlighted_atoms={0: 0},
        highlighted_rings={0: 1},
        rings=[[0, 1, 2, 3, 4, 5]],
        palette=["#ff0000", "#00ff00"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    ring_svg = doc.to_svg()
    assert "<svg" in ring_svg
    assert ring_svg != before, "highlighting should change the rendered SVG"
    assert "ff0000" in ring_svg.lower(), "the highlight color should actually appear in the SVG"
    assert "00ff00" in ring_svg.lower(), (
        "the ring highlight color should actually appear in the SVG"
    )


def assert_add_label(backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = create_doc("c1ccccc1O", backend)
    before = doc.to_svg()
    assert doc.get_label() is None

    doc.add_label("Compound 1")

    assert doc.get_label() == "Compound 1"
    svg = doc.to_svg()
    assert "<svg" in svg
    assert svg != before, "adding a label should change the rendered SVG"

    with pytest.raises(ValueError, match="Already labeled"):
        doc.add_label("Compound 2")


def assert_to_console(backend: type[HighlightBackendMoleculeT_co]) -> None:
    doc = backend.from_string("C=COCc1ccc(C)cc1", "SMILES")
    doc.kekulize(False)
    hml = HML(
        highlighted_atoms={8: 0},
        highlighted_bonds={0: 1},
        palette=["#ff0000", "#00ff00"],
    )
    doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    assert doc.to_console() == "C\033[38;2;0;255;0m=\033[0mCOCc1ccc\033[38;2;255;0;0m(C)\033[0mcc1"
