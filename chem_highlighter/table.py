"""Provide utilities to render Polars DataFrames as AG Grid HTML tables."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import msgspec
from typing_extensions import NotRequired

from chem_highlighter import HighlightBackendMolecule, RDKitMolecule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl
    from rdkit import Chem

logger = logging.getLogger()

# pixel width estimates.
CHAR_WIDTH_PX = 8
CELL_PADDING_PX = 32
HEADER_ICON_PADDING_PX = 20
MIN_COLUMN_WIDTH_PX = 60
MAX_COLUMN_WIDTH_PX = 400


def content_width(col: str, row_data: list[dict[str, Any]]) -> int:
    """Calculate column pixel width based on header and longest value."""
    content_len = max(
        (len(str(row[col])) for row in row_data if row.get(col) is not None),
        default=0,
    )
    content_px = content_len * CHAR_WIDTH_PX + CELL_PADDING_PX
    header_width = len(col) * CHAR_WIDTH_PX + CELL_PADDING_PX + HEADER_ICON_PADDING_PX
    width = max(content_px, header_width)
    return max(MIN_COLUMN_WIDTH_PX, min(MAX_COLUMN_WIDTH_PX, width))


class FilterParams(TypedDict):
    """Container for AG Grid column filter parameters."""

    buttons: list[str]


class ColumnDef(TypedDict):
    """Container for AG Grid column definitions."""

    field: str
    headerName: str
    sortable: bool
    filter: bool
    resizable: bool
    cellRenderer: NotRequired[str]
    width: NotRequired[int]
    filterParams: NotRequired[FilterParams]


class GridOptions(TypedDict):
    """Container for AG Grid definitions."""

    rowData: list[dict[str, Any]]
    columnDefs: list[ColumnDef]
    theme: NotRequired[str]
    rowHeight: NotRequired[int]
    pinnedTopRowData: NotRequired[list[dict[str, Any]]]


class TextFilterModel(TypedDict):
    """Container for AG Grid text column filter models."""

    filterType: NotRequired[Literal["text"]]
    type: NotRequired[
        Literal[
            "empty",
            "equals",
            "notEqual",
            "contains",
            "notContains",
            "startsWith",
            "endsWith",
            "blank",
            "notBlank",
        ]
    ]
    filter: NotRequired[str]
    filterTo: NotRequired[str]


class NumberFilterModel(TypedDict):
    """Container for AG Grid number column filter models."""

    filterType: NotRequired[Literal["number"]]
    type: NotRequired[
        Literal[
            "empty",
            "equals",
            "notEqual",
            "lessThan",
            "lessThanOrEqual",
            "greaterThan",
            "greaterThanOrEqual",
            "inRange",
            "blank",
            "notBlank",
        ]
    ]
    filter: NotRequired[float]
    filterTo: NotRequired[float]


class DateFilterModel(TypedDict):
    """Container for AG Grid date column filter models."""

    filterType: NotRequired[Literal["date"]]
    type: NotRequired[
        Literal[
            "empty",
            "equals",
            "notEqual",
            "lessThan",
            "greaterThan",
            "inRange",
            "blank",
            "notBlank",
        ]
    ]
    dateFrom: NotRequired[str]
    dateTo: NotRequired[str]


class SetFilterModel(TypedDict):
    """Container for AG Grid set column filter models."""

    filterType: NotRequired[Literal["set"]]
    values: list[str] | None


SimpleFilterModel = TextFilterModel | NumberFilterModel | DateFilterModel


class CombinedFilterModel(TypedDict):
    """Container for AG Grid combined column filter models."""

    filterType: NotRequired[str]
    operator: Literal["AND", "OR"]
    conditions: list[SimpleFilterModel]


FilterModel = SimpleFilterModel | SetFilterModel | CombinedFilterModel


def polars_to_aggrid(
    df: pl.DataFrame,
    svg_cols: Sequence[str] | None = None,
    pinned_rows: int = 0,
    preset_filter: Mapping[str, FilterModel] | None = None,
) -> str:
    """Convert Polars DataFrame to AG Grid webpage.

    Args:
        df: Input DataFrame.
        svg_cols: Columns to render as embedded SVG images.
        pinned_rows: Number of rows to pin at top.
        preset_filter: AG Grid filter model to apply on load.
    """
    has_svg = False
    placeholder = str(uuid.uuid4())
    grid_placeholder = str(uuid.uuid4())

    all_row_data = df.to_dicts()
    pinned_row_data, row_data = all_row_data[:pinned_rows], all_row_data[pinned_rows:]

    col_defs: list[ColumnDef] = []
    for col in df.columns:
        col_def: ColumnDef = {
            "field": col,
            "headerName": col,
            "sortable": True,
            "filter": True,
            "resizable": True,
            "filterParams": {"buttons": ["reset"]},  # reset button for each row.
        }
        if svg_cols and col in svg_cols:
            col_def["cellRenderer"] = placeholder
            has_svg = True
        else:
            col_def["width"] = content_width(col, all_row_data)
        col_defs.append(col_def)

    grid_opts: GridOptions = {
        "rowData": row_data,
        "columnDefs": col_defs,
        "theme": grid_placeholder,
        "rowHeight": 200,
    }
    if pinned_row_data:
        grid_opts["pinnedTopRowData"] = pinned_row_data

    grid_opts_str = msgspec.json.encode(grid_opts).decode()

    if has_svg:
        grid_opts_str = grid_opts_str.replace(
            f'"{placeholder}"',
            """\
(params) => `
    <div style="
        width:100%;
        height:100%;
        display:flex;
        align-items:center;
        justify-content:center;
    ">
        ${params.value}
    </div>
`""",
        )

    grid_opts_str = grid_opts_str.replace(f'"{grid_placeholder}"', "gridTheme")

    set_filter_model_call = ""
    set_defaults_button_html = ""
    set_defaults_handler = ""
    if preset_filter:
        filter_model_str = msgspec.json.encode(preset_filter).decode()
        set_filter_model_call = f"gridApi.setFilterModel({filter_model_str});"
        set_defaults_button_html = (
            '<button id="set-defaults" class="toolbar-btn">Set defaults</button>'
        )
        set_defaults_handler = (
            "document.querySelector('#set-defaults').addEventListener('click', () => "
            f"gridApi.setFilterModel({filter_model_str}));"
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
    <head>
        <script src="https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js"></script>

        <style>
        * {"{ padding: 0; margin: 0; }"}
        html, body {
        '''{
            min-height: 100% !important;
            height: 100%;
        }'''
    }
        .toolbar-btn {
        '''{
            margin: 8px 0 8px 8px;
            padding: 4px 12px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            font-size: 13px;
            color: #181d1f;
            background-color: #f8f8f8;
            border: 1px solid #babfc7;
            border-radius: 4px;
            cursor: pointer;
        }'''
    }
        .toolbar-btn:hover {
        '''{
            background-color: #eef7ff;
            border-color: #2196f3;
            color: #2196f3;
        }'''
    }
        .toolbar-btn:active {
        '''{
            background-color: #dcefff;
        }'''
    }
        </style>
    </head>
    <body>

        <button id="clear-filters" class="toolbar-btn">Clear filters</button>
        {set_defaults_button_html}
        <div id="grid" style="height: calc(100% - 40px)"></div>

        <script>
            console.log("HI");
            const gridTheme = agGrid.themeQuartz.withParams({
        '''{
                autoHeightMinBodyHeight: 200,
            }'''
    });
            const gridOptions = {grid_opts_str};
            const gridElement = document.querySelector('#grid');
            const gridApi = agGrid.createGrid(gridElement, gridOptions);
            {set_filter_model_call}
            document.querySelector('#clear-filters').addEventListener('click', () => {
        '''{
                gridApi.setFilterModel(null);
            }'''
    });
            {set_defaults_handler}
        </script>
    </body>
</html>
"""


def create_visualization_table(  # noqa: PLR0913,PLR0917
    df: pl.DataFrame,
    vis_mol_col: str = "Molecule",
    reference: int | Chem.Mol | None = None,
    clean_reference: bool = True,
    backend: type[HighlightBackendMolecule] = RDKitMolecule,
    pinned_rows: int = 0,
    preset_filter: Mapping[str, FilterModel] | None = None,
) -> str:
    """Render Polars DataFrame as AG Grid HTML table with molecule SVGs.

    Args:
        df: Input DataFrame.
        vis_mol_col: Column name containing RDKit molecule objects.
        reference: Reference molecule for alignment, provided as index or RDKit molecule.
        clean_reference: Whether to clean reference molecule before alignment.
        backend: Backend used to generate SVG depictions.
        pinned_rows: Number of rows to pin at top.
        preset_filter: AG Grid filter model to apply on load.

    Returns:
        Standalone HTML document containing interactive AG Grid table.
    """
    import polars as pl
    from rdkit import Chem

    reference_doc: HighlightBackendMolecule | None = None
    reference_molblock: str | None = None

    if isinstance(reference, int):
        placeholder = str(uuid.uuid4())
        reference = (
            df.with_row_index(placeholder)
            .filter(pl.col(placeholder) == reference)
            .select(vis_mol_col)
            .item()
        )
        if not isinstance(reference, Chem.Mol):
            logger.warning("Reference molecule is not a molecule, cannot align")
        else:
            reference_doc = backend.from_mol(reference)
    elif reference:
        reference_doc = backend.from_mol(reference)

    if reference_doc:
        if clean_reference:
            reference_doc.cleanup()
        reference_molblock = reference_doc.to_molblock()

    def to_svg(mol_like: Any) -> str:
        if not isinstance(mol_like, Chem.Mol):
            return "Error during visualization"
        doc = backend.from_mol(mol_like)
        if reference_molblock:
            doc.align_to_reference(reference_molblock)
        return doc.to_svg()

    df = df.with_columns(pl.col(vis_mol_col).map_elements(to_svg, pl.String))
    return polars_to_aggrid(
        df, svg_cols=[vis_mol_col], pinned_rows=pinned_rows, preset_filter=preset_filter
    )
