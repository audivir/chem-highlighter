"""Tests for table module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from chem_highlighter.table import (
    CELL_PADDING_PX,
    CHAR_WIDTH_PX,
    HEADER_ICON_PADDING_PX,
    MAX_COLUMN_WIDTH_PX,
    MIN_COLUMN_WIDTH_PX,
    content_width,
    create_visualization_table,
    polars_to_aggrid,
)
from chem_highlighter.utils import mol_from_smiles

if TYPE_CHECKING:
    import pytest


def test_content_width_empty_row_data_uses_minimum() -> None:
    assert content_width("a", []) == MIN_COLUMN_WIDTH_PX


def test_content_width_long_value_clamped_to_maximum() -> None:
    row_data = [{"col": "x" * 100}]
    assert content_width("col", row_data) == MAX_COLUMN_WIDTH_PX


def test_content_width_uses_header_when_longer_than_content() -> None:
    header = "a_much_longer_header_name"
    row_data = [{header: "a"}]
    expected = len(header) * CHAR_WIDTH_PX + CELL_PADDING_PX + HEADER_ICON_PADDING_PX
    assert content_width(header, row_data) == expected


def test_content_width_ignores_none_values() -> None:
    row_data: list[dict[str, Any]] = [{"col": None}, {"col": "abc"}]
    content_px = len("abc") * CHAR_WIDTH_PX + CELL_PADDING_PX
    header_width = len("col") * CHAR_WIDTH_PX + CELL_PADDING_PX + HEADER_ICON_PADDING_PX
    assert content_width("col", row_data) == max(content_px, header_width)


def test_polars_to_aggrid_basic() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    html = polars_to_aggrid(df)
    assert "<!DOCTYPE html>" in html
    assert '"field":"a"' in html
    assert "pinnedTopRowData" not in html


def test_polars_to_aggrid_with_svg_cols_wraps_cell_renderer() -> None:
    df = pl.DataFrame({"svg": ["<svg></svg>"]})
    html = polars_to_aggrid(df, svg_cols=["svg"])
    assert "params.value" in html


def test_polars_to_aggrid_with_pinned_rows() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    html = polars_to_aggrid(df, pinned_rows=1)
    assert "pinnedTopRowData" in html


def test_polars_to_aggrid_with_preset_filter() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    html = polars_to_aggrid(df, preset_filter={"a": {"filterType": "number", "filter": 1}})
    assert "gridApi.setFilterModel(" in html
    assert 'id="set-defaults"' in html


def test_create_visualization_table_without_reference() -> None:
    mol = mol_from_smiles("c1ccccc1")
    df = pl.DataFrame({"Molecule": [mol]})
    html = create_visualization_table(df)
    assert "<svg" in html


def test_create_visualization_table_with_reference_molecule() -> None:
    mol = mol_from_smiles("c1ccccc1")
    reference = mol_from_smiles("c1ccccc1O")
    df = pl.DataFrame({"Molecule": [mol]})
    html = create_visualization_table(df, reference=reference, clean_reference=False)
    assert "<svg" in html


def test_create_visualization_table_with_reference_index() -> None:
    mol = mol_from_smiles("c1ccccc1")
    df = pl.DataFrame({"Molecule": [mol, mol]})
    html = create_visualization_table(df, reference=0)
    assert "<svg" in html


def test_create_visualization_table_reference_index_not_a_molecule(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = pl.DataFrame({"Molecule": [None]})
    create_visualization_table(df, reference=0)
    assert "Reference molecule is not a molecule" in caplog.text


def test_create_visualization_table_non_molecule_value_reports_error() -> None:
    df = pl.DataFrame({"Molecule": pl.Series(["not a molecule"], dtype=pl.Object)})
    html = create_visualization_table(df)
    assert "Error during visualization" in html
