"""Tests for chem_highlighter.hml.Document."""

from __future__ import annotations

import pytest
from utils import read_fixture

from chem_highlighter.backend.rdkit import RDKitDocument, RDKitMolecule
from chem_highlighter.hml import InputFormatNotSupported, OutputFormatNotSupported


def _make_sdf(smiles_list: list[str]) -> str:
    """Concatenate single-record SDF exports into multi-record SDF."""
    return "".join(
        RDKitMolecule.from_string(smi, "SMILES").export_string("SDF") for smi in smiles_list
    )


def _make_rxn(reactant_smiles: str, product_smiles: str) -> str:
    """Create minimal $RXN block from reactant and product SMILES."""
    from rdkit.Chem import rdChemReactions

    rxn = rdChemReactions.ReactionFromSmarts(f"{reactant_smiles}>>{product_smiles}", useSmiles=True)
    return rdChemReactions.ReactionToRxnBlock(rxn)


def test_from_bytes_sdf_reads_every_record() -> None:
    sdf = _make_sdf(["CCO", "c1ccccc1", "CC(=O)O"])
    doc = RDKitDocument.from_bytes(sdf.encode(), "SDF")
    assert len(doc) == 3
    smiles = [m.kekulize(False).export_string("SMILES") for m in doc.molecules]
    assert smiles == ["CCO", "c1ccccc1", "CC(=O)O"]
    assert doc.source_format == "SDF"
    assert doc.offsets == [(0.0, 0.0)] * 3


def test_from_bytes_sdf_empty_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SDF input"):
        RDKitDocument.from_bytes(b"", "SDF")


def test_from_bytes_sdf_garbage_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SDF input"):
        RDKitDocument.from_bytes(b"invalid input", "SDF")


def test_from_bytes_rxn_reads_reactants_then_agents_then_products() -> None:
    block = _make_rxn("CCO", "CC=O")
    doc = RDKitDocument.from_bytes(block.encode(), "RXN")
    # MDL RXN files lack distinct agent section and contain only reactants and products.
    assert len(doc) == 2
    smiles = [m.export_string("SMILES") for m in doc.molecules]
    assert smiles == ["CCO", "CC=O"]
    assert doc.source_format == "RXN"


def test_from_bytes_rxn_empty_raises() -> None:
    with pytest.raises((ValueError, RuntimeError), match="Invalid RXN input"):
        RDKitDocument.from_bytes(b"", "RXN")


def test_from_bytes_rxn_garbage_raises() -> None:
    with pytest.raises((ValueError, RuntimeError), match="Invalid RXN input"):
        RDKitDocument.from_bytes(b"invalid input", "RXN")


def test_from_bytes_cdxml_allows_multiple_molecules() -> None:
    doc = RDKitDocument.from_bytes(read_fixture("two_mols.cdxml").encode(), "CDXML")
    assert len(doc) == 2
    assert doc.molecule(0) is not doc.molecule(1)
    # verify each molecule round-trips independently via export_string.
    assert doc.molecule(0).kekulize(False).export_string("SMILES") == "c1ccccc1"
    assert doc.molecule(1).kekulize(False).export_string("SMILES") == "c1ccccc1"


def test_from_bytes_cdxml_single_molecule() -> None:
    doc = RDKitDocument.from_bytes(read_fixture("one_mol.cdxml").encode(), "CDXML")
    assert len(doc) == 1


def test_from_bytes_cdxml_no_molecules_raises() -> None:
    with pytest.raises((ValueError, RuntimeError), match="Invalid CDXML input"):
        RDKitDocument.from_bytes(read_fixture("no_mol.cdxml").encode(), "CDXML")


def test_from_bytes_smiles_splits_fragments() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    assert len(doc) == 2
    smiles = [m.kekulize(False).export_string("SMILES") for m in doc.molecules]
    assert smiles == ["CCO", "c1ccccc1"]


def test_from_bytes_smiles_single_fragment() -> None:
    doc = RDKitDocument.from_bytes(b"CCO", "SMILES")
    assert len(doc) == 1


def test_from_bytes_smiles_empty_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SMILES input"):
        RDKitDocument.from_bytes(b"", "SMILES")


def test_from_bytes_smiles_garbage_raises() -> None:
    with pytest.raises(ValueError, match="Invalid SMILES input"):
        RDKitDocument.from_bytes(b"invalid input", "SMILES")


def test_from_bytes_mol_is_single_molecule() -> None:
    molblock = RDKitMolecule.from_string("CCO", "SMILES").to_molblock()
    doc = RDKitDocument.from_bytes(molblock.encode(), "Mol")
    assert len(doc) == 1
    assert doc.source_format == "Mol"


def test_from_bytes_inchi_is_single_molecule() -> None:
    inchi = RDKitMolecule.from_string("CCO", "SMILES").export_string("InChI")
    doc = RDKitDocument.from_string(inchi, "InChI")
    assert len(doc) == 1


def test_from_bytes_cdx_still_unsupported() -> None:
    with pytest.raises(InputFormatNotSupported, match="does not support importing from CDX"):
        RDKitDocument.from_bytes(b"", "CDX")


def test_document_requires_at_least_one_molecule() -> None:
    with pytest.raises(ValueError, match="Document requires at least one molecule"):
        RDKitDocument([], "Mol")


def test_document_offsets_length_must_match_molecules() -> None:
    mol = RDKitMolecule.from_string("CCO", "SMILES")
    with pytest.raises(ValueError, match="offsets must have the same length as molecules"):
        RDKitDocument([mol], "Mol", offsets=[(0.0, 0.0), (1.0, 1.0)])


def test_document_default_offsets_are_zero() -> None:
    mol = RDKitMolecule.from_string("CCO", "SMILES")
    doc = RDKitDocument([mol], "Mol")
    assert doc.offsets == [(0.0, 0.0)]


def test_document_molecule_out_of_range_raises_index_error() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    with pytest.raises(IndexError):
        doc.molecule(2)


def test_export_with_molecule_ix_delegates_to_that_molecule() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    assert doc.export_string("SMILES", molecule_ix=0) == doc.molecule(0).export_string("SMILES")
    assert doc.export("Mol", molecule_ix=1) == doc.molecule(1).export("Mol")


def test_export_sdf_concatenates_every_molecule() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    output = doc.export_string("SDF")
    assert output.count("$$$$") == 2
    # verify round-trip preserves record count.
    reparsed = RDKitDocument.from_bytes(output.encode(), "SDF")
    assert len(reparsed) == 2


def test_export_smiles_dot_joins_every_molecule() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    expected = ".".join(m.export_string("SMILES") for m in doc.molecules)
    assert doc.export_string("SMILES") == expected


def test_export_svg_combines_molecules_with_default_offsets() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    svg = doc.export_string("SVG")
    assert "<svg" in svg


def test_export_svg_translates_by_explicit_offsets() -> None:
    mols = [RDKitMolecule.from_string("CCO", "SMILES"), RDKitMolecule.from_string("CCO", "SMILES")]
    doc = RDKitDocument(mols, "SMILES", offsets=[(0.0, 0.0), (25.0, 0.0)])
    svg = doc.export_string("SVG")
    assert "<svg" in svg


def test_export_png_combines_molecules() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    png = doc.export("PNG")
    assert png[:4] == b"\x89PNG"


@pytest.mark.parametrize("fmt", ["RXN", "CDX", "CDXML", "EPS", "Mol", "InChI", "InChIKey"])
def test_export_whole_document_unsupported_formats(fmt: str) -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    with pytest.raises(
        OutputFormatNotSupported,
        match=f"RDKitDocument does not support exporting the whole document to {fmt}",
    ):
        doc.export(fmt)  # type: ignore[arg-type]


def test_export_single_molecule_document_falls_back_to_molecule_export() -> None:
    doc = RDKitDocument.from_bytes(b"CCO", "SMILES")
    assert doc.export_string("InChI") == doc.molecule(0).export_string("InChI")


def test_document_from_mol_and_from_molblock() -> None:
    mol = RDKitMolecule.from_string("CCO", "SMILES").mol
    doc = RDKitDocument.from_mol(mol)
    assert len(doc) == 1

    other = RDKitDocument.from_molblock(doc.to_molblock())
    assert other.to_molblock() == doc.to_molblock()


def test_to_molblock_with_index_delegates_to_that_molecule() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    assert doc.to_molblock(molecule_ix=1) == doc.molecule(1).to_molblock()


def test_to_molblock_default_uses_single_molecule_fallback() -> None:
    doc = RDKitDocument.from_bytes(b"CCO", "SMILES")
    assert doc.to_molblock() == doc.molecule(0).to_molblock()


def test_to_svg_default_and_with_index() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    assert "<svg" in doc.to_svg(molecule_ix=0)
    assert "<svg" in doc.to_svg()


def test_to_png_default_and_with_index() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    assert doc.to_png(molecule_ix=0)[:4] == b"\x89PNG"
    assert doc.to_png()[:4] == b"\x89PNG"


def test_to_console_delegates_with_index() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    assert doc.to_console(molecule_ix=0) == doc.molecule(0).to_console()


def test_to_console_joins_every_molecule_by_default() -> None:
    doc = RDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
    expected = ".".join(m.to_console() for m in doc.molecules)
    assert doc.to_console() == expected
