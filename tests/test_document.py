"""Tests for chem_highlighter.hml.Document, the multi-molecule counterpart to Molecule."""

from __future__ import annotations

import pytest
from utils import read_fixture

from chem_highlighter.backend.rdkit import RDKitMolecule
from chem_highlighter.hml import Document, InputFormatNotSupported, OutputFormatNotSupported


def _make_sdf(smiles_list: list[str]) -> str:
    """Concatenate several single-record SDF exports into one multi-record SDF."""
    return "".join(
        RDKitMolecule.from_string(smi, "SMILES").export_string("SDF") for smi in smiles_list
    )


def _make_rxn(reactant_smiles: str, product_smiles: str) -> str:
    """Build a minimal $RXN block with one reactant and one product."""
    from rdkit.Chem import rdChemReactions

    rxn = rdChemReactions.ReactionFromSmarts(
        f"{reactant_smiles}>>{product_smiles}", useSmiles=True
    )
    return rdChemReactions.ReactionToRxnBlock(rxn)


# --- from_bytes: SDF -----------------------------------------------------------------------


def test_from_bytes_sdf_reads_every_record() -> None:
    sdf = _make_sdf(["CCO", "c1ccccc1", "CC(=O)O"])
    doc = Document.from_bytes(sdf.encode(), "SDF", RDKitMolecule)
    assert len(doc) == 3
    smiles = [m.kekulize(False).export_string("SMILES") for m in doc.molecules]
    assert smiles == ["CCO", "c1ccccc1", "CC(=O)O"]
    assert doc.source_format == "SDF"
    assert doc.offsets == [(0.0, 0.0)] * 3


def test_from_bytes_sdf_empty_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SDF input"):
        Document.from_bytes(b"", "SDF", RDKitMolecule)


def test_from_bytes_sdf_garbage_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SDF input"):
        Document.from_bytes(b"invalid input", "SDF", RDKitMolecule)


# --- from_bytes: RXN -------------------------------------------------------------------------


def test_from_bytes_rxn_reads_reactants_then_agents_then_products() -> None:
    block = _make_rxn("CCO", "CC=O")
    doc = Document.from_bytes(block.encode(), "RXN", RDKitMolecule)
    # MDL RXN files have no distinct agent section -- only reactants then products.
    assert len(doc) == 2
    smiles = [m.export_string("SMILES") for m in doc.molecules]
    assert smiles == ["CCO", "CC=O"]
    assert doc.source_format == "RXN"


def test_from_bytes_rxn_empty_raises() -> None:
    with pytest.raises((ValueError, RuntimeError), match="Invalid RXN input"):
        Document.from_bytes(b"", "RXN", RDKitMolecule)


def test_from_bytes_rxn_garbage_raises() -> None:
    with pytest.raises((ValueError, RuntimeError), match="Invalid RXN input"):
        Document.from_bytes(b"invalid input", "RXN", RDKitMolecule)


# --- from_bytes: CDXML -------------------------------------------------------------------------


def test_from_bytes_cdxml_allows_multiple_molecules() -> None:
    doc = Document.from_bytes(read_fixture("two_mols.cdxml").encode(), "CDXML", RDKitMolecule)
    assert len(doc) == 2
    assert doc.molecule(0) is not doc.molecule(1)
    # Each molecule round-trips independently through its own export_string.
    assert doc.molecule(0).kekulize(False).export_string("SMILES") == "c1ccccc1"
    assert doc.molecule(1).kekulize(False).export_string("SMILES") == "c1ccccc1"


def test_from_bytes_cdxml_single_molecule() -> None:
    doc = Document.from_bytes(read_fixture("one_mol.cdxml").encode(), "CDXML", RDKitMolecule)
    assert len(doc) == 1


def test_from_bytes_cdxml_no_molecules_raises() -> None:
    with pytest.raises((ValueError, RuntimeError), match="Invalid CDXML input"):
        Document.from_bytes(read_fixture("no_mol.cdxml").encode(), "CDXML", RDKitMolecule)


# --- from_bytes: SMILES ------------------------------------------------------------------------


def test_from_bytes_smiles_splits_fragments() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    assert len(doc) == 2
    smiles = [m.kekulize(False).export_string("SMILES") for m in doc.molecules]
    assert smiles == ["CCO", "c1ccccc1"]


def test_from_bytes_smiles_single_fragment() -> None:
    doc = Document.from_bytes(b"CCO", "SMILES", RDKitMolecule)
    assert len(doc) == 1


def test_from_bytes_smiles_empty_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SMILES input"):
        Document.from_bytes(b"", "SMILES", RDKitMolecule)


def test_from_bytes_smiles_garbage_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SMILES input"):
        Document.from_bytes(b"invalid input", "SMILES", RDKitMolecule)


# --- from_bytes: single-molecule delegation (Mol, InChI, CDX) ----------------------------------


def test_from_bytes_mol_is_single_molecule() -> None:
    molblock = RDKitMolecule.from_string("CCO", "SMILES").to_molblock()
    doc = Document.from_bytes(molblock.encode(), "Mol", RDKitMolecule)
    assert len(doc) == 1
    assert doc.source_format == "Mol"


def test_from_bytes_inchi_is_single_molecule() -> None:
    inchi = RDKitMolecule.from_string("CCO", "SMILES").export_string("InChI")
    doc = Document.from_string(inchi, "InChI", RDKitMolecule)
    assert len(doc) == 1


def test_from_bytes_cdx_still_unsupported() -> None:
    with pytest.raises(InputFormatNotSupported, match="does not support importing from CDX"):
        Document.from_bytes(b"", "CDX", RDKitMolecule)


# --- __init__ / __len__ / molecule ---------------------------------------------------------


def test_document_requires_at_least_one_molecule() -> None:
    with pytest.raises(ValueError, match="Document requires at least one molecule"):
        Document([], "Mol")


def test_document_offsets_length_must_match_molecules() -> None:
    mol = RDKitMolecule.from_string("CCO", "SMILES")
    with pytest.raises(ValueError, match="offsets must have the same length as molecules"):
        Document([mol], "Mol", offsets=[(0.0, 0.0), (1.0, 1.0)])


def test_document_default_offsets_are_zero() -> None:
    mol = RDKitMolecule.from_string("CCO", "SMILES")
    doc = Document([mol], "Mol")
    assert doc.offsets == [(0.0, 0.0)]


def test_document_molecule_out_of_range_raises_index_error() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    with pytest.raises(IndexError):
        doc.molecule(2)


# --- export / export_string ------------------------------------------------------------------


def test_export_with_molecule_ix_delegates_to_that_molecule() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    assert doc.export_string("SMILES", molecule_ix=0) == doc.molecule(0).export_string("SMILES")
    assert doc.export("Mol", molecule_ix=1) == doc.molecule(1).export("Mol")


def test_export_sdf_concatenates_every_molecule() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    output = doc.export_string("SDF")
    assert output.count("$$$$") == 2
    # roundtrips back into the same number of records
    reparsed = Document.from_bytes(output.encode(), "SDF", RDKitMolecule)
    assert len(reparsed) == 2


def test_export_smiles_dot_joins_every_molecule() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    expected = ".".join(m.export_string("SMILES") for m in doc.molecules)
    assert doc.export_string("SMILES") == expected


def test_export_svg_combines_molecules_with_default_offsets() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    svg = doc.export_string("SVG")
    assert "<svg" in svg


def test_export_svg_translates_by_explicit_offsets() -> None:
    mols = [RDKitMolecule.from_string("CCO", "SMILES"), RDKitMolecule.from_string("CCO", "SMILES")]
    doc = Document(mols, "SMILES", offsets=[(0.0, 0.0), (25.0, 0.0)])
    svg = doc.export_string("SVG")
    assert "<svg" in svg


def test_export_png_combines_molecules() -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    png = doc.export("PNG")
    assert png[:4] == b"\x89PNG"


@pytest.mark.parametrize("fmt", ["RXN", "CDX", "CDXML", "EPS", "Mol", "InChI", "InChIKey"])
def test_export_whole_document_unsupported_formats(fmt: str) -> None:
    doc = Document.from_bytes(b"CCO.c1ccccc1", "SMILES", RDKitMolecule)
    with pytest.raises(
        OutputFormatNotSupported,
        match=f"Document does not support exporting the whole document to {fmt}",
    ):
        doc.export(fmt)  # type: ignore[arg-type]
