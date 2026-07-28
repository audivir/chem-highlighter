"""Highlight molecules using RDKit."""

from __future__ import annotations

import logging
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Literal, TypeAlias

import matplotlib as mpl
import msgspec
from typing_extensions import Self, override

from chem_highlighter.backend.map_tokens import map_smiles_tokens
from chem_highlighter.hml import (
    HML,
    HighlightBackendDocument,
    InputFormat,
    InputFormatNotSupported,
    OutputFormat,
    OutputFormatNotSupported,
)
from chem_highlighter.modify import apply_transform, flip_bond
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
        self._hml_json: str | None = None
        if not self.mol.GetNumConformers():
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(self.mol)
        Chem.Kekulize(self.mol)
        self._edit_state = True, False, False

    @override
    def get_edit_state(self) -> tuple[bool, bool, bool]:
        """Get the edit state tuple."""
        return self._edit_state

    @override
    def set_edit_state(self, kekulized: bool, aligned: bool, hydrogens_hidden: bool) -> None:
        """Set the edit state."""
        self._edit_state = kekulized, aligned, hydrogens_hidden

    @override
    def get_hml_json(self) -> str | None:
        """Get the JSON-encoded HML object."""
        return self._hml_json

    @override
    def set_hml_json(self, hml_json: str) -> None:
        """Set the JSON-encoded HML object."""
        self._hml_json = hml_json

    @override
    @classmethod
    def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Create a document from byte data."""
        from rdkit import Chem

        mol: Chem.Mol | None = None
        if fmt == "SDF":
            buffer = BytesIO(data)
            with Chem.ForwardSDMolSupplier(buffer) as sds:
                sd_mol: Chem.Mol | None = next(iter(sds), None)
            mol = sd_mol
        elif fmt == "Mol":
            mol = Chem.MolFromMolBlock(data.decode(), removeHs=False)
        elif fmt == "CDXML":
            mols: tuple[Chem.Mol, ...] = Chem.MolsFromCDXML(data.decode(), removeHs=False)
            if not mols:
                mol = None
            elif len(mols) > 1:
                raise ValueError("Multiple molecules in CDXML")
            else:
                mol = mols[0]
        elif fmt == "SMILES":
            mol = Chem.MolFromSmiles(data.decode())
        elif fmt == "InChI":
            inchi_mol: Chem.Mol | None = Chem.MolFromInchi(data.decode(), removeHs=False)  # type: ignore[no-untyped-call]
            mol = inchi_mol
        else:
            # RXN, CDX
            raise InputFormatNotSupported(f"{cls.__name__} does not support importing from {fmt}")
        if not mol or mol.GetNumAtoms() < 1:
            raise ValueError(f"Invalid {fmt} input")
        return cls(mol)

    @override
    def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:  # noqa: PLR0911
        """Export the document to the specified format as bytes."""
        from rdkit import Chem

        kekulized, _, _ = self.get_edit_state()
        hml_json = self.get_hml_json()
        hml = msgspec.json.Decoder(HML).decode(hml_json) if hml_json else None
        if fmt == "SDF":
            buffer = StringIO()
            with Chem.SDWriter(buffer) as sdw:
                sdw.SetForceV3000(not use_v2000)
                sdw.SetKekulize(kekulized)
                sdw.write(self.mol)
            return buffer.getvalue().encode()
        if fmt == "Mol":
            return Chem.MolToMolBlock(
                self.mol, kekulize=kekulized, forceV3000=not use_v2000
            ).encode()
        if fmt == "SMILES":
            return Chem.MolToSmiles(self.mol, kekuleSmiles=kekulized, canonical=False).encode()
        if fmt == "InChI":
            return Chem.MolToInchi(self.mol).encode()  # type: ignore[no-any-return,no-untyped-call]
        if fmt == "InChIKey":
            return Chem.MolToInchiKey(self.mol).encode()  # type: ignore[no-any-return,no-untyped-call]
        if fmt == "SVG":
            return draw_mol(hml, self.mol, "svg")
        if fmt == "PNG":
            return draw_mol(hml, self.mol, "png")
        # RXN, CDX, CDXML, EPS
        raise OutputFormatNotSupported(f"{type(self).__name__} does not support exporting to {fmt}")

    @override
    def align_to_reference_callback(
        self, flips: list[tuple[int, int]], global_flip: bool, angle: float
    ) -> None:
        """Align the underlying molecule to a reference molecule."""
        atol = 1e-5
        query = self.mol
        for bond_ix, anchor_atom_ix in flips:
            query = flip_bond(query, bond_ix, anchor_atom_ix, atol=atol)
        query = apply_transform(query, angle, flip_horizontal=global_flip, atol=atol)
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
        _, aligned, hydrogens_hidden = self.get_edit_state()
        self.set_edit_state(kekulize, aligned, hydrogens_hidden)

    @override
    def hide_hydrogens_callback(self) -> None:
        """Hide featureless hydrogens on carbon atoms."""
        from rdkit import Chem

        mol = Chem.Mol(self.mol)
        hml_json = self.get_hml_json()
        if hml_json:
            hml = msgspec.json.Decoder(HML).decode(hml_json)
            for ix in hml.highlighted_atoms:
                atom = mol.GetAtomWithIdx(ix)
                if atom.GetSymbol() == "H" and atom.GetIsotope() == 0:  # pragma: no branch
                    atom.SetIsotope(SENTINEL_ISOTOPE)

        mol = Chem.RemoveHs(mol)

        for atom in get_atoms(mol):
            if atom.GetSymbol() == "H" and atom.GetIsotope() == SENTINEL_ISOTOPE:
                atom.SetIsotope(0)

        self.mol = mol

    @override
    def highlight_from_json_callback(self, hml_json: str, hide_hydrogens: bool | None) -> None:
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

        hml_json = self.get_hml_json()
        hml = msgspec.json.Decoder(HML).decode(hml_json) if hml_json else None
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
