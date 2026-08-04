"""Tests for chem_highlighter.hml."""

from __future__ import annotations

import msgspec
import pytest
from utils import assert_benzene_kekulized

from chem_highlighter.align import get_alignment_ops_from_molblock
from chem_highlighter.backend.rdkit import RDKitMolecule
from chem_highlighter.hml import HML, HighlightBackendMolecule
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
    doc = RDKitMolecule.from_mol(mol_from_smiles("CCO"))
    assert doc.get_hml_json() is None
    json_str = doc.to_hmol_json()
    decoded = msgspec.json.decode(json_str)
    assert b'"mol"' in msgspec.json.encode(decoded)


def test_to_hmol_json_with_hml(doc: HighlightBackendMolecule, hml_json: str) -> None:
    doc.highlight_from_json(hml_json, False)
    found_hml_json = doc.get_hml_json()
    assert found_hml_json is not None
    # Palette must be serialised into the JSON
    assert "ff0000" in found_hml_json
    json_str = doc.to_hmol_json()
    assert "ff0000" in json_str
    assert b'"mol"' in msgspec.json.encode(msgspec.json.decode(json_str))


@pytest.mark.parametrize("hide_hydrogens", [False, True])
def test_highlight_from_json(
    hide_hydrogens: bool, doc: HighlightBackendMolecule, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, hide_hydrogens)
    assert doc.get_edit_state() == (True, False, hide_hydrogens)
    found_hml_json = doc.get_hml_json()
    assert found_hml_json is not None
    found_hml = msgspec.json.Decoder(HML).decode(hml_json)
    assert found_hml.palette == ["#ff0000"]


def test_cleanup_succeeds() -> None:
    doc = RDKitMolecule.from_mol(mol_from_smiles("CCO"))
    doc.cleanup()
    assert doc.get_edit_state() == (True, False, False)


def test_to_svg(doc: HighlightBackendMolecule) -> None:
    assert "<svg" in doc.to_svg()


def test_to_png(doc: HighlightBackendMolecule) -> None:
    assert doc.to_png()[:4] == b"\x89PNG"


@pytest.mark.parametrize("canonical", [True, False])
def test_to_console_default_implementation(canonical: bool, doc: HighlightBackendMolecule) -> None:
    # RDKitMolecule overrides `to_console`; call the base implementation directly.
    assert HighlightBackendMolecule.to_console(doc, canonical=canonical)


def test_align_to_reference_raises_when_already_aligned(doc: HighlightBackendMolecule) -> None:
    doc.align_to_reference(doc.to_molblock())
    with pytest.raises(ValueError, match="Already aligned"):
        doc.align_to_reference(doc.to_molblock())


def test_cleanup_raises_when_already_aligned(doc: HighlightBackendMolecule) -> None:
    doc.align_to_reference(doc.to_molblock())
    with pytest.raises(ValueError, match="Cleanup after alignment not supported"):
        doc.cleanup()


def test_cleanup_after_kekulize_reapplies_kekulize_state(doc: HighlightBackendMolecule) -> None:
    assert_benzene_kekulized(doc, True)
    doc.kekulize(False)
    assert_benzene_kekulized(doc, False)
    doc.cleanup()
    assert_benzene_kekulized(doc, False)


def test_hide_hydrogens_raises_when_already_set(doc: HighlightBackendMolecule) -> None:
    doc.hide_hydrogens()
    with pytest.raises(ValueError, match="Hydrogen display already set"):
        doc.hide_hydrogens()


def test_hide_hydrogens_raises_after_highlighting(
    doc: HighlightBackendMolecule, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, False)
    with pytest.raises(
        ValueError, match="Setting hydrogen display after highlighting not supported"
    ):
        doc.hide_hydrogens()


def test_highlight_from_json_raises_when_already_highlighted(
    doc: HighlightBackendMolecule, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, False)
    with pytest.raises(ValueError, match="Already highlighted"):
        doc.highlight_from_json(hml_json, False)


def test_highlight_from_json_raises_after_hydrogens_hidden(
    doc: HighlightBackendMolecule, hml_json: str
) -> None:
    doc.hide_hydrogens()

    with pytest.raises(
        ValueError, match="Highlighting after setting hydrogen display not supported"
    ):
        doc.highlight_from_json(hml_json, False)


def test_hide_hydrogens_raises_after_highlight_from_json_with_hide_hydrogens(
    doc: HighlightBackendMolecule, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, False)
    with pytest.raises(
        ValueError, match="Setting hydrogen display after highlighting not supported"
    ):
        doc.hide_hydrogens()


def test_callback_methods_bypass_the_guard(
    doc: HighlightBackendMolecule, hml_json: str, atol: float
) -> None:
    doc.cleanup_callback()
    doc.cleanup_callback()

    flips, global_flip, angle = get_alignment_ops_from_molblock(
        doc.to_molblock(), doc.to_molblock(), atol=atol
    )
    doc.align_to_reference_callback(flips, global_flip, angle)
    doc.align_to_reference_callback(flips, global_flip, angle)

    doc.hide_hydrogens_callback()
    doc.hide_hydrogens_callback()

    doc.highlight_from_json_callback(hml_json, False)
    doc.highlight_from_json_callback(hml_json, False)

    doc.add_label_callback("Compound 1")
    doc.add_label_callback("Compound 1")


def test_add_label(doc: HighlightBackendMolecule) -> None:
    assert doc.get_label() is None
    before = doc.to_svg()
    doc.add_label("Compound 1")
    assert doc.get_label() == "Compound 1"
    # RDKit draws the legend as vector paths, not literal `<text>`, so the string itself
    # doesn't appear in the SVG -- check it actually changed the rendering instead.
    assert doc.to_svg() != before
    assert 'class="legend"' in doc.to_svg()


def test_add_label_raises_when_already_labeled(doc: HighlightBackendMolecule) -> None:
    doc.add_label("Compound 1")
    with pytest.raises(ValueError, match="Already labeled"):
        doc.add_label("Compound 2")
