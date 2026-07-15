"""Highlight molecules using RDKit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Self, TypeAlias, override

import matplotlib as mpl

from chem_highlighter.align import get_alignment_flips_and_transform
from chem_highlighter.backend.map_tokens import map_smiles_tokens
from chem_highlighter.hml import HML, HighlightBackendDocument
from chem_highlighter.utils import RESET_COLOR, get_ansi_color

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Geometry import Point2D
logger = logging.getLogger(__name__)

Drawer: TypeAlias = "Draw.MolDraw2DCairo | Draw.MolDraw2DSVG"


def highlight_without_rings(
    drawer: Drawer, hml: HML, mol: Chem.Mol, atomrad: float = 0.4, widthmult: float = 2
) -> None:
    """Draw the molecule with the provided highlight options. Does not fill rings."""
    atoms = {
        atom_ix: [hml.get_rgba(group_ix)] for atom_ix, group_ix in hml.highlighted_atoms.items()
    }
    atomrads = dict.fromkeys(atoms, atomrad)
    bonds = {
        bond_ix: [hml.get_rgba(group_ix)] for bond_ix, group_ix in hml.highlighted_bonds.items()
    }
    widthmults = dict.fromkeys(bonds, widthmult)

    drawer.DrawMoleculeWithHighlights(mol, "", atoms, bonds, atomrads, widthmults)


def draw_polygon(
    drawer: Drawer,
    conf: Chem.Conformer,
    atom_ixs: Sequence[int],
    color: str,
) -> None:
    """Draw a polygon inbetween the specified atoms."""
    from rdkit.Geometry import Point2D

    points: list[Point2D] = []
    for atom_ix in atom_ixs:
        atom_pos = Point2D(conf.GetAtomPosition(atom_ix))
        points.append(atom_pos)
    drawer.SetFillPolys(True)
    drawer.SetColour(mpl.colors.to_rgba(color))
    drawer.DrawPolygon(points)


def highlight_rings(
    drawer: Drawer,
    hml: HML,
    mol: Chem.Mol,
) -> None:
    """Fill the rings with polygons."""
    # a hack to set the molecule scale
    highlight_without_rings(drawer, hml, mol)
    drawer.ClearDrawing()
    conf = mol.GetConformer()
    for ring_ix, group_ix in hml.highlighted_rings.items():
        color = hml.palette[group_ix]
        ring_atom_ixs = hml.rings[ring_ix]
        draw_polygon(drawer, conf, ring_atom_ixs, color)


def draw_mol(
    hml: HML | None,
    mol: Chem.Mol,
    output: Literal["png", "svg"],
    fill_rings: bool = True,
    opts: Draw.MolDrawOptions | None = None,
) -> bytes:
    """Draw the molecule with the provided highlighting options in the given output format."""
    from rdkit.Chem import Draw
    from rdkit_svg.draw import fix_svg, get_rdkit_svg
    from rdkit_svg.utils import to_string

    if not hml:
        hml = HML()
    # width, height = sub_img_size # noqa: ERA001

    # if align_to and tmol.HasSubstructMatch(align_to):
    #     rdDepictor.GenerateDepictionMatching2DStructure(tmol, align_to) # noqa: ERA001

    drawer = Draw.MolDraw2DCairo(-1, -1) if output == "png" else Draw.MolDraw2DSVG(-1, -1)

    clear_background = True
    if not opts:
        mean_bond_length = Draw.MeanBondLength(mol) or 1.0
        opts = drawer.drawOptions()
        Draw.SetACS1996Mode(opts, mean_bond_length)
    else:  # pragma: no cover
        clear_background = opts.clearBackground
    opts.clearBackground = True  # type: ignore[assignment]
    opts.prepareMolsBeforeDrawing = False  # type: ignore[assignment]
    drawer.SetDrawOptions(opts)

    if fill_rings:  # pragma: no branch
        # if we are filling rings, go ahead and do that first so that we draw
        # the molecule on top of the filled rings
        highlight_rings(drawer, hml, mol)
        if not clear_background:  # pragma: no cover
            opts.clearBackground = False  # type: ignore[assignment]

    highlight_without_rings(drawer, hml, mol)

    if output == "svg":
        svg = get_rdkit_svg(drawer)
        tree = fix_svg(svg)
        # svg = add_legend(svg, legend, line_breaks=False) # noqa: ERA001
        return to_string(tree).encode()  # type: ignore[no-any-return]
    return drawer.GetDrawingText()  # type: ignore[return-value]


class RDKitDocument(HighlightBackendDocument):
    """A structure to store a molecule for RDKit backend."""

    def __init__(self, mol: Chem.Mol) -> None:
        """Initialize the RDKitDocument from a RDKit molecule.

        Args:
            mol: The RDKit molecule to wrap.
        """
        from rdkit.Chem import rdDepictor

        self.mol = mol
        self.hml: HML | None = None
        if not self.mol.GetNumConformers():
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(self.mol)
        self._aligned = False
        self._kekulized = False

    @classmethod
    def convert_molblock(cls, molblock: str) -> Chem.Mol:
        """Convert a molecule as Mol block to a RDKit molecule."""
        return cls.from_molblock(molblock).mol

    @override
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create a RDKitDocument from a provided molecule as RDKit molecule.

        Args:
            mol: The RDKit molecule to import.
        """
        return cls(mol)

    @override
    @classmethod
    def from_molblock(cls, molblock: str) -> Self:
        """Create a RDKitDocument from a provided molecule as Mol block.

        Args:
            molblock: The Mol block to import.
        """
        from rdkit import Chem

        return cls(Chem.MolFromMolBlock(molblock, removeHs=False))

    @override
    def to_molblock(self) -> str:
        """Return the underlying molecule as Mol block."""
        from rdkit import Chem

        return Chem.MolToMolBlock(self.mol, forceV3000=True)

    @override
    def to_svg(self) -> str:
        """Return a highlighted (if set) SVG visualization of the molecule."""
        return draw_mol(self.hml, self.mol, "svg").decode()

    @override
    def to_png(self) -> bytes:
        """Return a highlighted (if set) PNG visualization of the molecule."""
        return draw_mol(self.hml, self.mol, "png")

    @override
    def align_to_reference(self, reference: str) -> None:
        """Align the underlying molecule to a reference molecule.

        Args:
            reference: The reference molecule as Mol block.
        """
        from rdkit import Chem
        from rdkit.Chem.rdMolTransforms import TransformConformer

        self._aligned = True
        query = self.mol
        reference_mol = Chem.MolFromMolBlock(reference)
        _, transform = get_alignment_flips_and_transform(query, reference_mol)
        conf = query.GetConformer()
        TransformConformer(conf, transform)

    @override
    def cleanup(self) -> None:
        """Sanitize the molecule and recalculate its 2D coordinates."""
        from rdkit import Chem
        from rdkit.Chem.rdDepictor import Compute2DCoords
        from rdkit.Chem.rdmolops import SanitizeMol

        if self._kekulized:
            logger.warning("Running Cleanup after Kekulization is broken for RDKit backend")
        if self._aligned:
            logger.warning("Running Cleanup after Alignment might change orientation")
        SanitizeMol(self.mol, Chem.SANITIZE_ALL)
        Compute2DCoords(self.mol, clearConfs=True)

    @override
    def kekulize(self, kekulize: bool) -> None:
        """Kekulize or dekekulize the underlying molecule."""
        from rdkit import Chem
        from rdkit.Chem.rdmolops import SanitizeMol

        self._kekulized = kekulize
        if kekulize:
            Chem.Kekulize(self.mol)
        else:
            SanitizeMol(self.mol, Chem.SANITIZE_SETAROMATICITY)

    @override
    def set_hydrogen_display(self, show_hydrogens: bool) -> None:
        """Show or hide featureless and non-highlighted hydrogens."""
        from rdkit import Chem

        if show_hydrogens:
            self.mol = Chem.AddHs(self.mol)
        else:
            rhps = Chem.RemoveHsParameters()
            rhps.removeMapped = False  # type: ignore[assignment]
            self.mol = Chem.RemoveHs(self.mol, rhps)

    @override
    def highlight_from_json_callback(self, hml_json: str) -> None:
        """Do nothing as highlighting occurs during visualization only."""

    @override
    def to_console(self, canonical: bool = True) -> str:
        """Return a colored string visualization of the molecule.

        Atoms and bonds that are highlighted in the HML are rendered with
        ANSI color codes via termcolor.
        """
        import os

        os.environ["FORCE_COLOR"] = "1"

        from rdkit import Chem

        smiles = Chem.MolToSmiles(self.mol, canonical=canonical, allBondsExplicit=True)

        char_maps = map_smiles_tokens(smiles, self.mol)

        hl_atoms = self.hml.highlighted_atoms if self.hml else {}
        hl_bonds = self.hml.highlighted_bonds if self.hml else {}
        palette = self.hml.palette if self.hml else []

        current_color: int | None = None
        chars_out: list[str] = []
        for cm in char_maps:
            if cm.type == "impl_bond":
                continue
            group_ix = hl_atoms.get(cm.ix) if cm.belongs_to == "atom" else hl_bonds.get(cm.ix)
            if group_ix is None:
                if current_color is not None:
                    current_color = None
                    chars_out.append(RESET_COLOR)
            elif group_ix != current_color:
                new_color = get_ansi_color(palette, group_ix)
                current_color = group_ix
                chars_out.append(new_color)
            chars_out.append(cm.token)
        if current_color is not None:
            chars_out.append(RESET_COLOR)

        return "".join(chars_out)
