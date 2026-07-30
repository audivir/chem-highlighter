"""Tests for chem_highlighter.backend.rdkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    assert_export_png_respects_environment,
    assert_export_unsupported_format,
    assert_from_bytes,
    assert_from_bytes_cdxml,
    assert_from_bytes_unsupported_format,
    assert_hide_hydrogens_special,
    assert_hide_hydrogens_table,
    assert_highlight_from_json,
    assert_hp_mol_v3000,
    assert_kekulize,
    assert_to_console,
)

from chem_highlighter.backend.rdkit import RDKitDocument

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from chem_highlighter.hml import InputFormat, OutputFormat


def test_doc() -> None:
    assert_doc_init(RDKitDocument)


@pytest.mark.parametrize("fmt", ["SDF", "Mol"])
@pytest.mark.parametrize("use_v2000", [False, True])
def test_export_mdl(fmt: Literal["SDF", "Mol", "RXN"], use_v2000: bool) -> None:
    assert_export_mdl(fmt, use_v2000, RDKitDocument)


def test_hp_mol_v3000() -> None:
    assert_hp_mol_v3000(1e-8, RDKitDocument)


@pytest.mark.parametrize("fmt", ["SVG", "PNG"])
def test_export_images(fmt: Literal["SVG", "PNG", "EPS"]) -> None:
    assert_export_images(fmt, RDKitDocument)


@pytest.mark.parametrize("transparent", [False, True])
def test_assert_export_png_respects_environment(transparent: bool) -> None:
    assert_export_png_respects_environment(transparent, RDKitDocument)


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
def test_align_to_reference(r_file: str, q_file_suffixes: Sequence[str], atol: float) -> None:
    assert_align_to_reference(r_file, q_file_suffixes, atol, RDKitDocument)


def test_cleanup(atol: float) -> None:
    doc = RDKitDocument.from_string("c1cnccc1", "SMILES")
    assert_cleanup(doc.to_molblock(), atol, RDKitDocument)


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
    assert_to_console(RDKitDocument)
