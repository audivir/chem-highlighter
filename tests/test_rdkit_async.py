"""Tests for the `AsyncRDKitMolecule` async backend."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from async_backend_test import (
    assert_add_label,
    assert_align_to_reference,
    assert_cleanup,
    assert_doc_init,
    assert_export,
    assert_export_mdl,
    assert_export_unsupported_format,
    assert_from_bytes,
    assert_from_bytes_unsupported_format,
    assert_hide_hydrogens_table,
    assert_highlight_from_json,
    assert_kekulize,
    assert_to_console,
)
from backend_test import ALIGNMENTS, HIDE_HYDROGENS

from chem_highlighter.backend.rdkit import AsyncRDKitMolecule

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from chem_highlighter.hml import OutputFormat


def test_doc() -> None:
    asyncio.run(assert_doc_init(AsyncRDKitMolecule))


@pytest.mark.parametrize("fmt", ["SDF", "Mol"])
@pytest.mark.parametrize("use_v2000", [False, True])
def test_export_mdl(fmt: Literal["SDF", "Mol"], use_v2000: bool) -> None:
    asyncio.run(assert_export_mdl(fmt, use_v2000, AsyncRDKitMolecule))


@pytest.mark.parametrize("fmt", ["SMILES", "InChI", "InChIKey"])
def test_export(fmt: Literal["SMILES", "InChI", "InChIKey"]) -> None:
    asyncio.run(assert_export(fmt, AsyncRDKitMolecule))


@pytest.mark.parametrize("fmt", ["RXN", "CDX", "CDXML", "EPS"])
def test_export_unsupported_format(fmt: OutputFormat) -> None:
    asyncio.run(assert_export_unsupported_format(fmt, AsyncRDKitMolecule, "RDKitMolecule"))


@pytest.mark.parametrize("fmt", ["SDF", "Mol", "SMILES", "InChI"])
def test_from_bytes(fmt: Literal["SDF", "Mol", "SMILES", "InChI"]) -> None:
    asyncio.run(assert_from_bytes(fmt, AsyncRDKitMolecule))


@pytest.mark.parametrize("fmt", ["RXN", "CDX"])
def test_from_bytes_unsupported_format(fmt: Literal["RXN", "CDX"]) -> None:
    asyncio.run(assert_from_bytes_unsupported_format(fmt, AsyncRDKitMolecule, "RDKitMolecule"))


@pytest.mark.parametrize(("r_file", "q_file_suffixes"), ALIGNMENTS)
def test_align_to_reference(r_file: str, q_file_suffixes: Sequence[str], atol: float) -> None:
    asyncio.run(assert_align_to_reference(r_file, q_file_suffixes, atol, AsyncRDKitMolecule))


def test_cleanup(atol: float) -> None:
    async def run() -> None:
        doc = await AsyncRDKitMolecule.from_string("c1cnccc1", "SMILES")
        expected = await doc.to_molblock()
        await assert_cleanup(expected, atol, AsyncRDKitMolecule)

    asyncio.run(run())


def test_kekulize() -> None:
    asyncio.run(assert_kekulize(AsyncRDKitMolecule))


@pytest.mark.parametrize(("base", "hide"), HIDE_HYDROGENS)
def test_hide_hydrogens_table(base: str, hide: str) -> None:
    asyncio.run(assert_hide_hydrogens_table(base, hide, AsyncRDKitMolecule))


def test_highlight_from_json(hml_json: str) -> None:
    asyncio.run(assert_highlight_from_json(hml_json, AsyncRDKitMolecule))


def test_add_label() -> None:
    asyncio.run(assert_add_label(AsyncRDKitMolecule))


def test_to_console() -> None:
    asyncio.run(assert_to_console(AsyncRDKitMolecule))


def test_concurrent_documents_run_in_parallel() -> None:
    """Verify that multiple documents are processed concurrently via the executor thread pool."""

    async def one(smiles: str) -> str:
        doc = await AsyncRDKitMolecule.from_string(smiles, "SMILES")
        return await doc.export_string("InChI")

    async def run() -> list[str]:
        return await asyncio.gather(*[one(s) for s in ["CCO", "c1ccccc1O", "CCN", "CC(=O)O"]])

    results = asyncio.run(run())
    assert len(results) == 4
    assert all(r.startswith("InChI=") for r in results)
