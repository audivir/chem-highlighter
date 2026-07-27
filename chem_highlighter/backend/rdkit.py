"""Highlight molecules using RDKit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, TypeAlias

import matplotlib as mpl
from typing_extensions import Self, override

from chem_highlighter.align import get_alignment_flips_and_transform
from chem_highlighter.backend.map_tokens import map_smiles_tokens
from chem_highlighter.hml import HML, EditState, HighlightBackendDocument
from chem_highlighter.modify import apply_transform, flip_bond, parse_transform
from chem_highlighter.utils import RESET_COLOR, get_ansi_color, get_atoms

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Geometry import Point2D
logger = logging.getLogger(__name__)

Drawer: TypeAlias = "Draw.MolDraw2DCairo | Draw.MolDraw2DSVG"

SENTINEL_ISOTOPE = 999


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
        from rdkit import Chem
        from rdkit.Chem import rdDepictor

        self.mol = Chem.Mol(mol)
        self._hml: HML | None = None
        if not self.mol.GetNumConformers():
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(self.mol)
        self._edit_state = EditState(
            None, False, False
        )  # RDKit automatically dekekulize on parsing

    @override
    def get_edit_state(self) -> EditState:
        """Get the edit state tuple."""
        return self._edit_state

    @override
    def set_edit_state(self, edit_state: EditState) -> None:
        """Set the edit state."""
        self._edit_state = edit_state

    @override
    def get_hml(self) -> HML | None:
        """Get the HML object."""
        return self._hml

    @override
    def set_hml(self, hml: HML) -> None:
        """Set the HML object."""
        self._hml = hml

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

        return Chem.MolToMolBlock(
            self.mol, kekulize=self.get_edit_state().kekulized or False, forceV3000=True
        )

    @override
    def to_svg(self) -> str:
        """Return a highlighted (if set) SVG visualization of the molecule."""
        return draw_mol(self.get_hml(), self.mol, "svg").decode()

    @override
    def to_png(self) -> bytes:
        """Return a highlighted (if set) PNG visualization of the molecule."""
        return draw_mol(self.get_hml(), self.mol, "png")

    @override
    def align_to_reference_callback(self, reference: str) -> None:
        """Align the underlying molecule to a reference molecule."""
        from rdkit import Chem

        atol = 1e-5
        query = self.mol
        reference_mol = Chem.MolFromMolBlock(reference)
        flips, transform = get_alignment_flips_and_transform(query, reference_mol, atol=atol)
        global_flip, found_angle = parse_transform(transform, atol=atol)
        for bond_ix, anchor_atom_ix in flips:
            query = flip_bond(query, bond_ix, anchor_atom_ix, atol=atol)
        query = apply_transform(query, found_angle, flip_horizontal=global_flip, atol=atol)
        self.mol = query

    @override
    def cleanup_callback(self) -> None:
        """Sanitize the molecule and recalculate its 2D coordinates."""
        from rdkit import Chem
        from rdkit.Chem.rdDepictor import Compute2DCoords
        from rdkit.Chem.rdmolops import SanitizeMol

        SanitizeMol(self.mol, Chem.SANITIZE_ALL)
        Compute2DCoords(self.mol, clearConfs=True)

    @override
    def kekulize_callback(self, kekulize: bool) -> None:
        """Kekulize or dekekulize the underlying molecule."""
        from rdkit import Chem
        from rdkit.Chem.rdmolops import SanitizeMol

        if kekulize:
            Chem.Kekulize(self.mol)
        else:
            SanitizeMol(self.mol, Chem.SANITIZE_SETAROMATICITY)

    @override
    def set_hydrogen_display_callback(self, show_hydrogens: bool) -> None:
        """Show or hide hydrogens on carbon atoms."""
        from rdkit import Chem

        if show_hydrogens:
            atoms = self.mol.GetAtoms()  # type: ignore[no-untyped-call]
            carbon_ixs = [atom.GetIdx() for atom in atoms if atom.GetSymbol() == "C"]
            self.mol = Chem.AddHs(self.mol, onlyOnAtoms=carbon_ixs, addCoords=True)
        else:
            mol = Chem.Mol(self.mol)
            hml = self.get_hml()
            if hml:
                for ix in hml.highlighted_atoms:
                    atom = mol.GetAtomWithIdx(ix)
                    if atom.GetSymbol() == "H" and atom.GetIsotope() == 0: # pragma: no branch
                        atom.SetIsotope(SENTINEL_ISOTOPE)

            mol = Chem.RemoveHs(mol)

            for atom in get_atoms(mol):
                if atom.GetSymbol() == "H" and atom.GetIsotope() == SENTINEL_ISOTOPE:
                    atom.SetIsotope(0)

            self.mol = mol

    @override
    def highlight_from_json_callback(self, hml_json: str, show_hydrogens: bool | None) -> None:
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

        hml = self.get_hml()
        hl_atoms = hml.highlighted_atoms if hml else {}
        hl_bonds = hml.highlighted_bonds if hml else {}
        palette = hml.palette if hml else []

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
        if current_color is not None:  # pragma: no cover
            chars_out.append(RESET_COLOR)

        return "".join(chars_out)
