"""Tests for `AsyncHighlightBackendMolecule` edit-state guards, mirroring `test_hml.py`."""

from __future__ import annotations

import asyncio

import pytest

from chem_highlighter.backend.rdkit import AsyncRDKitMolecule


async def make_doc() -> AsyncRDKitMolecule:
    return await AsyncRDKitMolecule.from_string("c1ccccc1", "SMILES")


def test_cleanup_raises_when_already_aligned() -> None:
    async def run() -> None:
        doc = await make_doc()
        await doc.align_to_reference(await doc.to_molblock())
        with pytest.raises(ValueError, match="Cleanup after alignment not supported"):
            await doc.cleanup()

    asyncio.run(run())


def test_hide_hydrogens_raises_when_already_set() -> None:
    async def run() -> None:
        doc = await make_doc()
        await doc.hide_hydrogens()
        with pytest.raises(ValueError, match="Hydrogen display already set"):
            await doc.hide_hydrogens()

    asyncio.run(run())


def test_hide_hydrogens_raises_after_highlighting(hml_json: str) -> None:
    async def run() -> None:
        doc = await make_doc()
        await doc.highlight_from_json(hml_json, False)
        with pytest.raises(
            ValueError, match="Setting hydrogen display after highlighting not supported"
        ):
            await doc.hide_hydrogens()

    asyncio.run(run())


def test_highlight_from_json_raises_when_already_highlighted(hml_json: str) -> None:
    async def run() -> None:
        doc = await make_doc()
        await doc.highlight_from_json(hml_json, False)
        with pytest.raises(ValueError, match="Already highlighted"):
            await doc.highlight_from_json(hml_json, False)

    asyncio.run(run())


def test_highlight_from_json_raises_after_hydrogens_hidden(hml_json: str) -> None:
    async def run() -> None:
        doc = await make_doc()
        await doc.hide_hydrogens()
        with pytest.raises(
            ValueError, match="Highlighting after setting hydrogen display not supported"
        ):
            await doc.highlight_from_json(hml_json, False)

    asyncio.run(run())
