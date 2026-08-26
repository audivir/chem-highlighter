"""Tests for chem_highlighter.hml.AsyncDocument, mirroring test_document.py."""

from __future__ import annotations

import asyncio

import pytest

from chem_highlighter.backend.rdkit import AsyncRDKitDocument, RDKitMolecule
from chem_highlighter.hml import OutputFormatNotSupported


def test_document_requires_at_least_one_molecule() -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="Document requires at least one molecule"):
            AsyncRDKitDocument([], "Mol")

    asyncio.run(run())


def test_document_offsets_length_must_match_molecules() -> None:
    async def run() -> None:
        single_doc = await AsyncRDKitDocument.from_string("CCO", "SMILES")
        with pytest.raises(ValueError, match="offsets must have the same length as molecules"):
            AsyncRDKitDocument(single_doc.molecules, "Mol", offsets=[(0.0, 0.0), (1.0, 1.0)])

    asyncio.run(run())


def test_document_default_offsets_and_len_and_molecule_access() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        assert len(doc) == 2
        assert doc.offsets == [(0.0, 0.0), (0.0, 0.0)]
        assert doc.molecule(0) is not doc.molecule(1)
        with pytest.raises(IndexError):
            doc.molecule(2)

    asyncio.run(run())


def test_document_from_mol_and_from_molblock() -> None:
    async def run() -> None:
        mol = RDKitMolecule.from_string("CCO", "SMILES").mol
        doc = await AsyncRDKitDocument.from_mol(mol)
        assert len(doc) == 1

        other = await AsyncRDKitDocument.from_molblock(await doc.to_molblock())
        assert await other.to_molblock() == await doc.to_molblock()

    asyncio.run(run())


def test_export_with_molecule_ix_delegates_to_that_molecule() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        assert await doc.export_string("SMILES", molecule_ix=0) == await doc.molecule(
            0
        ).export_string("SMILES")

    asyncio.run(run())


def test_export_sdf_concatenates_every_molecule() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        output = await doc.export_string("SDF")
        assert output.count("$$$$") == 2

    asyncio.run(run())


def test_export_smiles_dot_joins_every_molecule() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        expected = ".".join([await m.export_string("SMILES") for m in doc.molecules])
        assert await doc.export_string("SMILES") == expected

    asyncio.run(run())


def test_export_svg_and_png_combine_molecules() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        assert "<svg" in await doc.to_svg()
        assert (await doc.to_png())[:4] == b"\x89PNG"

    asyncio.run(run())


def test_export_single_molecule_document_falls_back_to_molecule_export() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO", "SMILES")
        assert await doc.export_string("InChI") == await doc.molecule(0).export_string("InChI")
        assert await doc.to_molblock() == await doc.molecule(0).to_molblock()

    asyncio.run(run())


def test_export_whole_document_unsupported_format_raises() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        with pytest.raises(
            OutputFormatNotSupported,
            match="AsyncRDKitDocument does not support exporting the whole document to InChI",
        ):
            await doc.export("InChI")

    asyncio.run(run())


def test_to_molblock_and_to_svg_and_to_png_with_index() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        assert await doc.to_molblock(molecule_ix=1) == await doc.molecule(1).to_molblock()
        assert "<svg" in await doc.to_svg(molecule_ix=0)
        assert (await doc.to_png(molecule_ix=0))[:4] == b"\x89PNG"

    asyncio.run(run())


def test_to_console_delegates_with_index() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        assert await doc.to_console(molecule_ix=0) == await doc.molecule(0).to_console()

    asyncio.run(run())


def test_to_console_joins_every_molecule_by_default() -> None:
    async def run() -> None:
        doc = await AsyncRDKitDocument.from_bytes(b"CCO.c1ccccc1", "SMILES")
        expected = ".".join([await m.to_console() for m in doc.molecules])
        assert await doc.to_console() == expected

    asyncio.run(run())
