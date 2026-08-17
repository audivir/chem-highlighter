"""Highlight R-groups in molecule."""

from __future__ import annotations

import re
from io import StringIO
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from chem_highlighter.backend.rdkit import RDKitMolecule
from chem_highlighter.hml import HML, HighlightBackendMolecule, HighlightBackendMoleculeT_co
from chem_highlighter.utils import get_atoms, get_bonds, mol_from_smiles, mol_to_smiles, setup_cmap

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import polars as pl
    from rdkit import Chem
    from rdkit.Chem import Mol

    from chem_highlighter.decomposer.core import DecomposedList, DecomposedMol
    from chem_highlighter.utils import RGBA

# rdBase.DisableLog("rdApp.warning") # noqa: ERA001


def remove_empty_branches(smiles: str) -> str:
    """Remove empty branches from SMILES string."""
    return smiles.replace("()", "")


def validate_data(  # pragma: no cover
    df: pl.DataFrame,
    base: str,
    legends_col: str | None = None,
    ignore_cols: Sequence[str] | None = None,
) -> None:
    """Validate input DataFrame column naming and SMILES strings.

    Args:
        df: DataFrame containing SMILES data.
        base: Base SMILES string to validate.
        legends_col: Column name to exclude from validation.
        ignore_cols: Sequence of column names to exclude from validation.

    Raises:
        ValueError: If columns are not named in 'R{n}' format.
    """
    ignore_cols = [] if ignore_cols is None else list(ignore_cols)

    if legends_col is not None:
        ignore_cols.append(legends_col)

    ignore_cols_set = set(ignore_cols)
    smi_cols = [x for x in df.columns if x not in ignore_cols_set]

    for name in smi_cols:
        if not (name.startswith("R") and name[1:].isdigit()):
            raise ValueError(f"Columns should be named R1, R2, R3, etc. Got {name}")

    # validate SMILES strings.
    mol_from_smiles(base)
    for elem in df[smi_cols].to_numpy().flat:
        mol_from_smiles(elem)


def complete_smiles(  # pragma: no cover
    df: pl.DataFrame, base: str, only_cores: bool = False
) -> list[str] | set[str]:
    """Generate complete SMILES strings by replacing R-group placeholders in base structure.

    Args:
        df: DataFrame containing R-group substitutions.
        base: Base SMILES string with map numbers.
        only_cores: Whether to exclude R-groups with map numbers below 10 and return unique cores.

    Returns:
        List or set of completed SMILES strings.
    """
    base = mol_to_smiles(mol_from_smiles(base))

    rgroups = [x for x in df.columns if x.startswith("R")]

    smiles: list[str] = []

    for row in df.iter_rows(named=True):
        smi = base

        for r_group in rgroups:
            if only_cores and int(r_group[1:]) < 10:  # noqa: PLR2004
                continue
            smi = smi.replace(f"[*:{r_group[1:]}]", row[r_group])

        smi = remove_empty_branches(smi)
        smi = mol_to_smiles(mol_from_smiles(smi))

        smiles.append(smi)

    if only_cores:
        return set(smiles)

    return smiles


def create_empty(core: Mol) -> Chem.Mol:
    """Generate empty core by replacing R-group dummies with carbons and removing empty branches."""
    core_smi = mol_to_smiles(core)
    empty = re.sub(r"\[\*\:\d+\]", "C", core_smi)
    empty = remove_empty_branches(empty)
    empty_mol = mol_from_smiles(empty)
    if empty_mol is None:  # pragma: no cover
        raise ValueError(f"{empty} is not valid SMILES")
    return empty_mol


def set_neighbor_labels_for_atom(
    lbl: str, mol: Chem.Mol, at: Chem.Atom, source_idx_prop: str = "SourceAtomIdx"
) -> None:
    """Assign isotope labels to neighbors of dummy atom based on map number."""
    if not (
        not at.GetAtomicNum()
        and at.GetAtomMapNum()
        and at.HasProp("dummyLabel")
        and at.GetProp("dummyLabel") == lbl
    ):
        return

    at_map_num = at.GetAtomMapNum()
    # identify atoms connected to attachment point within molecule.
    for nbr in at.GetNeighbors():
        if not nbr.HasProp(source_idx_prop):  # pragma: no cover
            continue
        nbr_ix = nbr.GetIntProp(source_idx_prop)
        m_at = mol.GetAtomWithIdx(nbr_ix)

        if m_at_iso := m_at.GetIsotope():  # pragma: no cover
            m_at.SetIntProp("_OrigIsotope", m_at_iso)
        m_at.SetIsotope(200 + at_map_num)


def set_orig_isotope(
    old_idx: int, at: Chem.Atom, source_idx_prop: str = "SourceAtomIdx"
) -> tuple[int, int] | None:
    """Restore original isotope value for atom based on source index property."""
    # restore original isotope values and adjust for atom index shifts from hydrogen removal.
    if at.HasProp(source_idx_prop):
        if at.HasProp("_OrigIsotope"):  # pragma: no cover
            at.SetIsotope(at.GetIntProp("_OrigIsotope"))
            at.ClearProp("_OrigIsotope")
        else:
            at.SetIsotope(0)

        new_idx = at.GetIntProp(source_idx_prop)
        return old_idx, new_idx

    return None  # pragma: no cover


def get_ring_fill(
    aring: tuple[int, ...],
    rquery: Chem.Mol,
    old_new_atom_map: dict[int, int],
    source_idx_prop: str = "SourceAtomIdx",
) -> list[int] | None:
    """Map ring atom indices to corresponding source indices."""
    tring: list[int] = []
    all_found = True
    for aid in aring:
        at = rquery.GetAtomWithIdx(aid)
        if not at.HasProp(source_idx_prop):  # pragma: no cover
            all_found = False
            break
        tring.append(old_new_atom_map[at.GetIntProp(source_idx_prop)])
    if all_found:
        return tring

    return None  # pragma: no cover


def get_bond_fill(
    qbnd: Chem.Bond,
    tmol: Chem.Mol,
    old_new_atom_map: dict[int, int],
    source_idx_prop: str = "SourceAtomIdx",
) -> int | None:
    """Retrieve original bond index using source atom properties and atom mapping."""
    batom = qbnd.GetBeginAtom()
    eatom = qbnd.GetEndAtom()

    if batom.HasProp(source_idx_prop) and eatom.HasProp(source_idx_prop):
        orig_bnd = tmol.GetBondBetweenAtoms(
            old_new_atom_map[batom.GetIntProp(source_idx_prop)],
            old_new_atom_map[eatom.GetIntProp(source_idx_prop)],
        )
        return orig_bnd.GetIdx()
    return None  # pragma: no cover


def get_old_new_atom_map(tmol: Chem.Mol, source_idx_prop: str = "SourceAtomIdx") -> dict[int, int]:
    """Map original atom indices to current indices using source atom properties."""
    old_new_atom_map: dict[int, int] = {}
    for ix, at in enumerate(get_atoms(tmol)):
        mapping = set_orig_isotope(ix, at, source_idx_prop)
        if mapping is None:  # pragma: no cover
            continue
        old_idx, new_idx = mapping
        old_new_atom_map[old_idx] = new_idx
    return old_new_atom_map


def convert_decomposed_to_hml(
    mol: Chem.Mol,
    group: Mapping[str, Chem.Mol],
    lbls: Sequence[str],
    source_idx_prop: str = "SourceAtomIdx",
    get_colors_func: Callable[[], list[RGBA]] | None = None,
) -> HML:
    """Convert decomposed molecule groups into highlight mapping for visualization."""
    from rdkit import Chem

    old_new_atom_map = get_old_new_atom_map(mol, source_idx_prop)
    old_new_atom_map = {v: k for k, v in old_new_atom_map.items()}

    get_colors_func = get_colors_func or setup_cmap
    colors = get_colors_func()

    atoms: dict[int, list[RGBA]] = {}
    bonds: dict[int, list[RGBA]] = {}
    rings: dict[int, list[RGBA]] = {}
    rings_ixs: dict[tuple[int, ...], int] = {}

    for ix, lbl in enumerate(lbls):
        color = colors[ix % len(colors)]
        rquery = group[lbl]
        Chem.GetSSSR(rquery)

        for at in get_atoms(rquery):
            if at.HasProp(source_idx_prop):
                orig_idx = old_new_atom_map[at.GetIntProp(source_idx_prop)]
                atoms.setdefault(orig_idx, []).append(color)

        rinfo = rquery.GetRingInfo()
        for aring in rinfo.AtomRings():
            tring = get_ring_fill(aring, rquery, old_new_atom_map, source_idx_prop)
            if tring is None:  # pragma: no cover
                continue
            ring_ix = rings_ixs.setdefault(tuple(sorted(tring)), len(rings_ixs))
            rings.setdefault(ring_ix, []).append(color)

        for qbnd in get_bonds(rquery):
            if (bnd_idx := get_bond_fill(qbnd, mol, old_new_atom_map, source_idx_prop)) is not None:
                bonds.setdefault(bnd_idx, []).append(color)

    return HML.from_multicolor(atoms, bonds, rings, list(rings_ixs))


def highlight_rgroups(  # noqa: PLR0913,PLR0917
    groups: dict[str, Chem.Mol],
    lbls: list[str],
    mol: Mol,
    core: Mol,
    source_idx_prop: str = "SourceAtomIdx",
    get_colors_func: Callable[[], list[RGBA]] | None = None,
    backend: type[HighlightBackendMoleculeT_co] = RDKitMolecule,  # type: ignore[assignment]
) -> HighlightBackendMoleculeT_co:
    """Highlight R-group substituents within molecule."""
    from rdkit import Chem
    from rdkit.Chem import rdqueries

    mol = Chem.Mol(mol)
    core = Chem.Mol(core)

    # include atom map numbers in substructure search to improve alignment with symmetric cores.
    for at in get_atoms(core):
        if at_map_num := at.GetAtomMapNum():
            at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200 + at_map_num))

    for lbl in lbls:
        group = groups[lbl]

        for at in get_atoms(group):
            set_neighbor_labels_for_atom(lbl, mol, at, source_idx_prop)

    doc = RDKitMolecule.from_mol(mol)

    # identify atoms, bonds, and rings for highlighting using `mol` to preserve `source_idx_prop`.
    hml = convert_decomposed_to_hml(
        mol,
        groups,
        lbls,
        source_idx_prop,
        get_colors_func,
    )

    new_doc = backend.from_molblock(doc.to_molblock())
    new_doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)

    return new_doc


def draw_single(
    decomposed: DecomposedMol,
    # sub_img_size: tuple[int, int] = (250, 200), # noqa: ERA001
    get_colors_func: Callable[[], list[RGBA]] | None = None,
    # opts: MolDrawOptions | None = None,  # noqa: ERA001
    backend: type[HighlightBackendMoleculeT_co] = RDKitMolecule,  # type: ignore[assignment]
) -> HighlightBackendMoleculeT_co:
    """Draw decomposed molecule using highlighting and alignment.

    Args:
        decomposed: Decomposed molecule data.
        get_colors_func: Optional function to provide color mapping.
        backend: Highlighting backend.

    Returns:
        Highlighted document.
    """
    doc = highlight_rgroups(
        decomposed.groups,
        decomposed.labels,
        decomposed.complete,
        decomposed.qcore,
        get_colors_func=get_colors_func,
        backend=backend,
    )
    ref_doc = backend.from_mol(decomposed.qcore)
    doc.align_to_reference(ref_doc.to_molblock())
    doc.kekulize(True)

    return doc


def draw_multiple(  # pragma: no cover
    filename: str | Path,
    groups: list[DecomposedMol] | DecomposedList,
    n_per_row: int = 4,
    # sub_img_size: tuple[int, int] = (250, 200), # noqa: ERA001
    get_colors_func: Callable[[], list[RGBA]] | None = None,
    backend: type[HighlightBackendMolecule] = RDKitMolecule,
) -> None:
    """Draw multiple molecule groups in grid layout."""
    import imgkit
    import numpy as np
    import pandas as pd

    n_rows = ceil(len(groups) / n_per_row)

    matrix = np.empty((n_rows, n_per_row), dtype=object)
    matrix[:] = ""

    for ix, group in enumerate(groups):
        col = ix % n_per_row
        row = ix // n_per_row

        doc = draw_single(group, get_colors_func, backend=backend)

        matrix[row, col] = doc.to_svg()

    buffer = StringIO()

    pd.DataFrame(matrix).to_html(  # type: ignore[unused-ignore,call-overload]
        buffer, border=0, escape=False, header=None, index=None
    )

    data = buffer.getvalue()
    # convert numpy matrix to HTML table.
    filename = Path(filename)
    if filename.suffix == ".html":
        Path(filename).write_text(data, encoding="utf-8")

    elif filename.suffix == ".png":
        imgkit.from_string(data, filename)

    else:
        raise ValueError("filename must end with .html or .png")
