"""Utilities for rendering Polars DataFrames as AG Grid HTML tables."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
from typing_extensions import NotRequired

from chem_highlighter import HighlightBackendDocument, RDKitDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from rdkit import Chem

logger = logging.getLogger()


class ColumnDef(TypedDict):
    """AG Grid column definition."""

    field: str
    headerName: str
    sortable: bool
    filter: bool
    resizable: bool
    cellRenderer: NotRequired[str]


class GridOptions(TypedDict):
    """AG Grid grid definition."""

    rowData: list[dict[str, Any]]
    columnDefs: list[ColumnDef]
    theme: NotRequired[str]
    rowHeight: NotRequired[int]


def polars_to_aggrid(df: pl.DataFrame, svg_cols: Sequence[str] | None = None) -> str:
    """Convert a Polars DataFrame to an AG Grid webpage."""
    has_svg = False
    placeholder = str(uuid.uuid4())
    grid_placeholder = str(uuid.uuid4())

    row_data = df.to_dicts()
    col_defs: list[ColumnDef] = []
    for col in df.columns:
        col_def: ColumnDef = {
            "field": col,
            "headerName": col,
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        if svg_cols and col in svg_cols:
            col_def["cellRenderer"] = placeholder
            has_svg = True
        col_defs.append(col_def)

    grid_opts: GridOptions = {
        "rowData": row_data,
        "columnDefs": col_defs,
        "theme": grid_placeholder,
        "rowHeight": 200,
    }

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
        </style>
    </head>
    <body>

        <div id="grid" style="height: 100%"></div>

        <script>
            console.log("HI");
            const gridTheme = agGrid.themeQuartz.withParams({
        '''{
                autoHeightMinBodyHeight: 200,
            }'''
    });
            const gridOptions = {grid_opts_str};
            const gridElement = document.querySelector('#grid');
            agGrid.createGrid(gridElement, gridOptions);
        </script>
    </body>
</html>
"""


def create_visualization_table(
    df: pl.DataFrame,
    vis_mol_col: str = "Molecule",
    reference: int | Chem.Mol | None = None,
    clean_reference: bool = True,
    backend: type[HighlightBackendDocument] = RDKitDocument,
) -> str:
    """Render a Polars DataFrame as an AG Grid HTML table with molecule SVGs.

    Example:
        >>> import polars as pl
        >>> from rdkit import Chem
        >>> df = pl.DataFrame({"SMILES": ["CCCC", "CCO"]})
        >>> df = df.with_columns(
        ...     pl.col("SMILES")
        ...     .map_elements(Chem.MolFromSmiles, return_dtype=pl.Object)
        ...     .alias("Molecule")
        ... )
        >>> html = create_visualization_table(df, reference=0)

    Args:
        df: Input DataFrame.
        vis_mol_col: Name of the column containing RDKit molecule objects.
        reference: Reference molecule to align to, either as index or RDKit molecule.
        clean_reference: Clean up reference before aligning.
        backend: Backend used to generate SVG depictions.

    Returns:
        A standalone HTML document containing the interactive AG Grid table.
    """
    import polars as pl
    from rdkit import Chem

    reference_doc: HighlightBackendDocument | None = None
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
    return polars_to_aggrid(df, svg_cols=[vis_mol_col])
