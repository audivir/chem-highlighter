"""Module for decomposing molecules into their core and residues."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.decomposer.core import Core, DecomposedList, DecomposedMol
from chem_highlighter.decomposer.highlight import draw_single
from chem_highlighter.decomposer.rejoin import join_multiple, mask_hydrogens
from chem_highlighter.utils import get_atoms, mol_to_smiles

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from rdkit import Chem

    from chem_highlighter.hml import HighlightBackendDocumentT_co


def _change_map_atoms(mol: Chem.Mol, from_: int, to: int) -> Chem.Mol:
    """Return a copy of *mol* with all atom-map numbers equal to *from_* changed to *to*."""
    from rdkit import Chem

    mol = Chem.Mol(mol)
    for atom in get_atoms(mol):
        if atom.GetAtomMapNum() == from_:
            atom.SetAtomMapNum(to)
    return mol


def _switch_equiv(
    a_smi: tuple[Chem.Mol, int], b_smi: tuple[Chem.Mol, int]
) -> tuple[Chem.Mol, Chem.Mol]:
    """Swap two R-group substituents while exchanging their attachment-point map numbers."""
    (a_old, a), (b_old, b) = a_smi, b_smi
    a_new = _change_map_atoms(b_old, b, a)
    b_new = _change_map_atoms(a_old, a, b)
    return a_new, b_new


def _fix_rdata_equivs(
    rdata_: Mapping[str, Sequence[Chem.Mol]], equivalents: Sequence[tuple[int, int]]
) -> dict[str, list[Chem.Mol]]:
    """Reorder equivalent R-group columns so the most-common substituent is at each position."""
    rdata = {key: list(values) for key, values in rdata_.items()}

    for a, b in equivalents:
        a_values = rdata[f"R{a}"]
        b_values = rdata[f"R{b}"]
        joined_counts = Counter(
            [mol_to_smiles(x) for x in a_values] + [mol_to_smiles(x) for x in b_values]
        )

        for ix, (a_val, b_val) in enumerate(zip(a_values, b_values, strict=True)):
            if joined_counts[mol_to_smiles(b_val)] > joined_counts[mol_to_smiles(a_val)]:
                a_values[ix], b_values[ix] = _switch_equiv((a_val, a), (b_val, b))

        rdata[f"R{a}"] = a_values
        rdata[f"R{b}"] = b_values

    return rdata


def _fix_core_equivs(
    decomposed: DecomposedMol,
    most_tracker: dict[str, tuple[Chem.Mol, float]],
    equivalents: Sequence[tuple[int, int]],
    symbol: str = "I",
) -> dict[str, tuple[Chem.Mol, float]]:
    """Sync most-tracker assignments with those found by re-decomposing the joined core."""
    for a, b in equivalents:
        a_val = decomposed.groups[f"R{a}"]
        a_mol, a_ratio = most_tracker[f"R{a}"]

        if mol_to_smiles(a_val) != mol_to_smiles(a_mol).replace("[H]", symbol):
            # swap most with b
            b_mol, b_ratio = most_tracker[f"R{b}"]
            a_new, b_new = _switch_equiv((a_mol, a), (b_mol, b))
            most_tracker[f"R{a}"] = (a_new, b_ratio)
            most_tracker[f"R{b}"] = (b_new, a_ratio)

    return most_tracker


def replace_atom(svg: str, decomposed: DecomposedMol, symbol: str = "I") -> str:
    """Hide *symbol* atoms in *svg* and replace their bonds with dashed lines."""
    from rdkit import Chem
    from rdkit_svg.utils import save_parse

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    xml = save_parse(svg)

    compl_wo_hs = Chem.RemoveHs(decomposed.complete)
    atom_indices = [x.GetIdx() for x in get_atoms(compl_wo_hs) if x.GetSymbol() == symbol]

    for elem in xml.iter():
        if not ("path" in elem.tag and "class" in elem.attrib):
            continue
        if elem.attrib["class"] in [f"atom-{x}" for x in atom_indices]:
            elem.attrib["style"] = "display: none;"
        if "bond" in elem.attrib["class"] and any(
            f"atom-{x}" in elem.attrib["class"] for x in atom_indices
        ):
            elem.attrib["stroke-dasharray"] = "5,5"
    return ET.tostring(xml).decode()


def fix_decomposed(
    decomposed: DecomposedList, equivalents: Sequence[tuple[int, int]] | None = None
) -> tuple[DecomposedList, dict[str, list[Chem.Mol]]]:
    """Reorder *decomposed* R-groups by equivalence and return updated per-position lists."""
    rdata = decomposed.build_rdata()

    if equivalents is None:
        return decomposed, rdata

    rdata = _fix_rdata_equivs(rdata, equivalents)

    for ix, mol in enumerate(decomposed):
        for key in decomposed.labels:
            mol.groups[key] = rdata[key][ix]

    return decomposed, rdata


def plot_decomposed(  # noqa: PLR0913
    decomposed: DecomposedList,
    equivalents: Sequence[tuple[int, int]] | None = None,
    # sub_img_size: tuple[int, int] = (250, 200),# noqa: ERA001
    ylim: tuple[float, float] | None = None,
    symbol: str = "I",
    fig: Figure | SubFigure | None = None,
    # opts: MolDrawOptions | None = None, # noqa: ERA001
    title: str | None = None,
    backend: type[HighlightBackendDocumentT_co] = RDKitDocument,  # type: ignore[assignment]
) -> tuple[DecomposedMol, HighlightBackendDocumentT_co, str, Axes]:
    """Render the most-common R-group for each position with a frequency-scaled colorbar.

    Returns:
        A tuple of (final DecomposedMol, highlighted document, SVG string, molecule axes).
    """
    import matplotlib as mpl
    from matplotlib.figure import Figure

    if equivalents is None:
        pass

    decomposed, rdata = fix_decomposed(decomposed, equivalents)

    most_tracker: dict[str, tuple[Chem.Mol, float]] = {}

    for key, values in rdata.items():
        most_smi, most_count = Counter(mol_to_smiles(x) for x in values).most_common(1)[0]
        most_mol = next(x for x in values if mol_to_smiles(x) == most_smi)
        most_tracker[key] = (most_mol, most_count / len(values))

    complete = join_multiple(
        decomposed.core,
        {
            key: mask_hydrogens(mol, int(key.removeprefix("R")), symbol)
            for key, (mol, _) in most_tracker.items()
        },
    )

    final_comp, _ = Core(mol_to_smiles(decomposed.qcore)).decompose([mol_to_smiles(complete)])
    final = final_comp[0]

    if equivalents is not None:
        most_tracker = _fix_core_equivs(final, most_tracker, equivalents, symbol)

    cmap = mpl.colormaps.get_cmap("plasma")

    color_codes = [cmap(most_tracker[key][1]) for key in final.labels]

    # make a color bar for 0 - 100 %
    if fig is None:
        fig = Figure(figsize=(10, 10))

    axes: tuple[Axes, Axes] = fig.subplots(2, 1, height_ratios=[6, 1])
    ax, colorbar_ax = axes

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])

    doc = draw_single(
        final,
        # sub_img_size=sub_img_size # noqa: ERA001
        get_colors_func=lambda: color_codes,
        # opts=opts # noqa: ERA001
        backend=backend,
    )
    decomposed_svg = doc.to_svg()
    decomposed_svg = replace_atom(decomposed_svg, final, symbol)

    ax.axis("off")
    colorbar_ax.axis("off")

    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal")

    if title:
        cbar.ax.set_xlabel(title, fontsize=10)

    if ylim is not None:
        ax.set_ylim(*ylim)

    return final, doc, decomposed_svg, ax


def colorbar(fig: Figure | SubFigure, nticks: int = 2, cmap: str = "plasma") -> Axes:
    """Render a horizontal gradient colorbar spanning 0→1 inside *fig*."""
    import numpy as np
    from matplotlib.patches import Rectangle

    ax = fig.add_subplot()

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    # ...extent=[left, right, bottom, top]
    _ = ax.imshow(gradient, aspect="auto", cmap=cmap, extent=(0, 1, 0, 0.05))

    # Formatting the "Colorbar"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.05)

    ax.set_xticks(np.linspace(0, 1, nticks))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, nticks)])
    ax.set_yticks([])

    rect = Rectangle((0, 0), 1, 0.05, linewidth=1, edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    return ax
