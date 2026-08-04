"""Highlight molecules using RDKit."""

from __future__ import annotations

import logging
import re
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Literal, TypeAlias

import matplotlib as mpl
import msgspec
from typing_extensions import Self, override

from chem_highlighter.hml import (
    HML,
    Document,
    HighlightBackendMolecule,
    InputFormat,
    InputFormatNotSupported,
    OutputFormat,
    OutputFormatNotSupported,
)
from chem_highlighter.modify import apply_transform, flip_bond
from chem_highlighter.utils import get_atoms, get_high_precision_v3000, get_png_render_options

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Geometry import Point2D
logger = logging.getLogger(__name__)

Drawer: TypeAlias = "Draw.MolDraw2DCairo | Draw.MolDraw2DSVG"

SENTINEL_ISOTOPE = 999


def highlight_without_rings(
    drawer: Drawer,
    hml: HML,
    mol: Chem.Mol,
    legend: str | None = None,
    atomrad: float = 0.4,
    widthmult: float = 2,
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

    drawer.DrawMoleculeWithHighlights(mol, legend or "", atoms, bonds, atomrads, widthmults)


LEGEND_PATH_RE = re.compile(r'(<path class="legend"[^>]*?fill="([^"]+)")\s*/>')


def _bold_legend(svg: str) -> str:
    """Fake a bold legend by outlining its (already vector-path) glyphs with a matching stroke.

    RDKit draws the legend as filled vector paths (`class='legend'`), not real `<text>`, so
    there's no font weight to select -- a small stroke in the same color visually thickens the
    glyph outlines instead.
    """
    return LEGEND_PATH_RE.sub(
        lambda m: f'{m.group(1)} stroke="{m.group(2)}" stroke-width="1.2" />', svg
    )


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
    # TODO(tihoph): assert the drawing is actually clean, sometimes with the drawing twice
    conf = mol.GetConformer()
    for ring_ix, group_ix in hml.highlighted_rings.items():
        color = hml.palette[group_ix]
        ring_atom_ixs = hml.rings[ring_ix]
        draw_polygon(drawer, conf, ring_atom_ixs, color)


def draw_mol(
    hml: HML | None,
    mol: Chem.Mol,
    output: Literal["png", "svg"],
    legend: str | None = None,
    fill_rings: bool = True,
    opts: Draw.MolDrawOptions | None = None,
) -> bytes:
    """Draw the molecule with the provided highlighting options in the given output format."""
    from rdkit.Chem import Draw
    from rdkit_svg.draw import get_rdkit_svg
    from rdkit_svg.utils import assure_elem, hide_objects, read_tree, remove_whitespace, to_string

    if not hml:
        hml = HML()

    transparent, width, height = get_png_render_options()

    if output == "png" and width is not None and height is not None:
        drawer = Draw.MolDraw2DCairo(round(width), round(height))
        auto_fit = True
    elif output == "png":
        drawer = Draw.MolDraw2DCairo(-1, -1)
        auto_fit = False
    else:
        drawer = Draw.MolDraw2DSVG(-1, -1)
        auto_fit = False

    if not opts:  # pragma: no branch
        opts = drawer.drawOptions()
        if auto_fit:
            # TODO(tihoph): but i want to still have the ACS1996 look,
            # maybe creating svg and then rendering to png? keeping aspect ratio intact

            # a fixed canvas size auto-fits/centers the molecule to it (unlike ACS1996 mode,
            # which draws at a fixed absolute scale regardless of canvas size), matching the
            # bounding-box behavior of another backend's resvg-based PNG renderer. Only used
            # once a size is actually configured -- unconfigured, PNG keeps its previous -1,-1
            # ACS1996 sizing.
            opts.prepareMolsBeforeDrawing = True  # type: ignore[assignment]
        else:
            mean_bond_length = Draw.MeanBondLength(mol) or 1.0
            Draw.SetACS1996Mode(opts, mean_bond_length)
            opts.prepareMolsBeforeDrawing = False  # type: ignore[assignment]

    if output == "svg" or transparent:
        opts.clearBackground = True  # type: ignore[assignment]
        opts.setBackgroundColour((0.0, 0.0, 0.0, 0.0))

    if fill_rings:  # pragma: no branch
        # if we are filling rings, go ahead and do that first so that we draw
        # the molecule on top of the filled rings
        highlight_rings(drawer, hml, mol)

    highlight_without_rings(drawer, hml, mol, legend)

    if isinstance(drawer, Draw.MolDraw2DSVG):
        svg = get_rdkit_svg(drawer)
        tree = read_tree(svg)
        if not assure_elem(tree):
            raise ValueError("No root element found")
        hide_objects(tree, ["rect"])
        if not legend:
            # rdkit_svg's bbox math (remove_whitespace/find_bounds) undercounts the legend's
            # bezier-curve glyph paths, cropping the canvas back down to just the molecule and
            # clipping the legend out of view -- so skip cropping whenever a legend is present
            # and keep RDKit's own (correctly-sized) canvas instead.
            remove_whitespace(tree)
        svg = to_string(tree)
        if legend:
            svg = _bold_legend(svg)
        return svg.encode()  # type: ignore[no-any-return,unused-ignore]
    return drawer.GetDrawingText()  # type: ignore[no-any-return]


def export_rdkit_molecule(mol_backend: RDKitMolecule, fmt: OutputFormat, use_v2000: bool) -> bytes:  # noqa: PLR0911
    """Serialize one RDKit-backed molecule to `fmt`.

    Shared by `RDKitMolecule.export` (standalone use) and `RDKitDocument` (whole-document/
    single-molecule export), so the format-conversion logic lives in exactly one place.
    """
    from rdkit import Chem

    kekulized, _, _ = mol_backend.get_edit_state()
    hml_json = mol_backend.get_hml_json()
    hml = msgspec.json.Decoder(HML).decode(hml_json) if hml_json else None
    if fmt == "SDF":
        buffer = StringIO()
        with Chem.SDWriter(buffer) as sdw:
            sdw.SetForceV3000(not use_v2000)
            sdw.SetKekulize(kekulized)
            sdw.write(mol_backend.mol)
        return buffer.getvalue().encode()
    if fmt == "Mol":
        if use_v2000:
            return Chem.MolToMolBlock(mol_backend.mol, kekulize=kekulized).encode()
        return get_high_precision_v3000(mol_backend.mol, kekulize=kekulized).encode()
    if fmt == "SMILES":
        return Chem.MolToSmiles(mol_backend.mol, kekuleSmiles=kekulized, canonical=False).encode()
    if fmt == "InChI":
        return Chem.MolToInchi(mol_backend.mol).encode()  # type: ignore[no-any-return,no-untyped-call]
    if fmt == "InChIKey":
        return Chem.MolToInchiKey(mol_backend.mol).encode()  # type: ignore[no-any-return,no-untyped-call]
    if fmt == "SVG":
        return draw_mol(hml, mol_backend.mol, "svg", mol_backend._label)  # noqa: SLF001
    if fmt == "PNG":
        return draw_mol(hml, mol_backend.mol, "png", mol_backend._label)  # noqa: SLF001
    # RXN, CDX, CDXML, EPS
    name = type(mol_backend).__name__
    raise OutputFormatNotSupported(f"{name} does not support exporting to {fmt}")


class RDKitMolecule(HighlightBackendMolecule):
    """A structure to store a molecule for RDKit backend."""

    def __init__(self, mol: Chem.Mol) -> None:
        """Initialize the RDKitMolecule from a RDKit molecule.

        Args:
            mol: The RDKit molecule to wrap.
        """
        from rdkit import Chem
        from rdkit.Chem import rdDepictor

        self.mol = Chem.Mol(mol)
        self._hml_json: str | None = None
        self._label: str | None = None
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
    def get_label(self) -> str | None:
        """Get the label text, if set."""
        return self._label

    @override
    def set_label(self, label: str) -> None:
        """Set the label text."""
        self._label = label

    @override
    def add_label_callback(self, text: str) -> None:
        """Do nothing: the label is rendered directly from state during `export`."""

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
                raise ValueError(f"Invalid {fmt} input: Multiple groups/fragments found")
            else:
                mol = mols[0]
        elif fmt == "SMILES":
            params = Chem.SmilesParserParams()
            params.removeHs = False  # type: ignore[assignment]
            mol = Chem.MolFromSmiles(data.decode(), params)
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
    def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:
        """Export the molecule to the specified format as bytes."""
        return export_rdkit_molecule(self, fmt, use_v2000)

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
    def highlight_from_json_callback(self, hml_json: str, hide_hydrogens: bool = False) -> None:
        """Do nothing as highlighting occurs during visualization only."""
        if hide_hydrogens:
            self.hide_hydrogens_callback()


class RDKitDocument(Document[RDKitMolecule]):
    """A collection of one or more molecules parsed from a single input, using the RDKit backend."""

    @classmethod
    @override
    def _split_bytes(cls, data: bytes, fmt: InputFormat) -> list[RDKitMolecule]:
        from rdkit import Chem

        mols: list[Chem.Mol]
        if fmt == "SDF":
            buffer = BytesIO(data)
            with Chem.ForwardSDMolSupplier(buffer) as sds:
                mols = list(sds)
        elif fmt == "RXN":
            from rdkit.Chem import rdChemReactions

            try:
                rxn = rdChemReactions.ReactionFromRxnBlock(data.decode())
            except (ValueError, RuntimeError):
                rxn = None
            mols = [*rxn.GetReactants(), *rxn.GetAgents(), *rxn.GetProducts()] if rxn else []
        elif fmt == "CDXML":
            mols = list(Chem.MolsFromCDXML(data.decode(), removeHs=False))
        elif fmt == "SMILES":
            params = Chem.SmilesParserParams()
            params.removeHs = False  # type: ignore[assignment]
            smiles_mol = Chem.MolFromSmiles(data.decode(), params)
            mols = (
                list(Chem.GetMolFrags(smiles_mol, asMols=True, sanitizeFrags=False))
                if smiles_mol
                else []
            )
        else:
            # Mol, CDX, InChI: always a single molecule (or unsupported) -- reuse the
            # single-molecule backend's own construction/restrictions unchanged.
            return [RDKitMolecule.from_bytes(data, fmt)]

        mols = [mol for mol in mols if mol is not None and mol.GetNumAtoms() >= 1]
        if not mols:
            raise ValueError(f"Invalid {fmt} input")
        return [RDKitMolecule(mol) for mol in mols]

    @override
    def _combined_export(self, fmt: Literal["SVG", "PNG"]) -> bytes:
        combined = self._combine_for_export()
        return draw_mol(None, combined, "svg" if fmt == "SVG" else "png")

    def _combine_for_export(self) -> Chem.Mol:
        """Combine every molecule into one ephemeral `Chem.Mol`, translated by its offset."""
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        combined: Chem.Mol | None = None
        for mol_backend, (dx, dy) in zip(self.molecules, self.offsets, strict=True):
            mol = Chem.Mol(mol_backend.mol)
            if (dx, dy) != (0.0, 0.0):
                conf = mol.GetConformer()
                for atom_ix in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_ix)
                    conf.SetAtomPosition(atom_ix, Point3D(pos.x + dx, pos.y + dy, pos.z))
            combined = mol if combined is None else Chem.CombineMols(combined, mol)
        if combined is None:  # pragma: no cover -- `molecules` is never empty, see `__init__`
            raise ValueError("Document requires at least one molecule")
        # `CombineMols` does not carry ring perception over onto the new `Chem.Mol`, so
        # rendering it (which needs ring info to fill/highlight rings) would otherwise fail.
        Chem.GetSSSR(combined)
        return combined
