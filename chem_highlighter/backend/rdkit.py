"""Highlight molecules using RDKit."""

from __future__ import annotations

import asyncio
import logging
import re
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Literal, TypeAlias

import matplotlib as mpl
import msgspec
from typing_extensions import Self, override

from chem_highlighter.align import get_2d_mol, get_alignment_flips_and_transform, get_best_fit_angle
from chem_highlighter.hml import (
    HML,
    AsyncDocument,
    AsyncHighlightBackendMolecule,
    Document,
    HighlightBackendMolecule,
    InputFormat,
    InputFormatNotSupported,
    OutputFormat,
    OutputFormatNotSupported,
)
from chem_highlighter.modify import apply_transform, flip_bond, parse_transform
from chem_highlighter.utils import get_atoms, get_high_precision_v3000, get_png_render_options

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Geometry import Point2D
logger = logging.getLogger(__name__)

Drawer: TypeAlias = "Draw.MolDraw2DCairo | Draw.MolDraw2DSVG"

SENTINEL_ISOTOPE = 999


def mol_from_molblock(molblock: str) -> Chem.Mol:
    """Parse mol block into sanitized RDKit molecule.

    Raises:
        ValueError: If molblock is invalid.
    """
    from rdkit import Chem

    mol: Chem.Mol | None = Chem.MolFromMolBlock(molblock, removeHs=False)
    if not mol:
        raise ValueError("Invalid molblock")
    return mol


def highlight_without_rings(  # noqa: PLR0913,PLR0917
    drawer: Drawer,
    hml: HML,
    mol: Chem.Mol,
    legend: str | None = None,
    atomrad: float = 0.4,
    widthmult: float = 2,
) -> None:
    """Draw molecule with highlights and no ring fills."""
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
    """Simulate bold legend text by adding matching strokes to vector-path glyphs."""
    return LEGEND_PATH_RE.sub(
        lambda m: f'{m.group(1)} stroke="{m.group(2)}" stroke-width="1.2" />', svg
    )


def draw_polygon(
    drawer: Drawer,
    conf: Chem.Conformer,
    atom_ixs: Sequence[int],
    color: str,
) -> None:
    """Draw polygon connecting specified atoms."""
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
    """Fill highlighted rings with polygons."""
    # set molecule scale.
    highlight_without_rings(drawer, hml, mol)
    drawer.ClearDrawing()
    # prevent duplicate rendering.
    conf = mol.GetConformer()
    for ring_ix, group_ix in hml.highlighted_rings.items():
        color = hml.palette[group_ix]
        ring_atom_ixs = hml.rings[ring_ix]
        draw_polygon(drawer, conf, ring_atom_ixs, color)


def draw_mol(  # noqa: C901,PLR0912,PLR0913,PLR0917
    hml: HML | None,
    mol: Chem.Mol,
    output: Literal["png", "svg"],
    legend: str | None = None,
    fill_rings: bool = True,
    opts: Draw.MolDrawOptions | None = None,
) -> bytes:
    """Draw molecule with specified highlighting and output format."""
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
            # maintain ACS1996 style by rendering SVG to PNG to preserve aspect ratio.

            # center and auto-fit molecule to match resvg behavior.
            opts.prepareMolsBeforeDrawing = True  # type: ignore[assignment]
        else:
            mean_bond_length = Draw.MeanBondLength(mol) or 1.0
            Draw.SetACS1996Mode(opts, mean_bond_length)
            opts.prepareMolsBeforeDrawing = False  # type: ignore[assignment]

    if output == "svg" or transparent:
        opts.clearBackground = True  # type: ignore[assignment]
        opts.setBackgroundColour((0.0, 0.0, 0.0, 0.0))

    if fill_rings:  # pragma: no branch
        # fill rings first to ensure molecule is drawn on top.
        highlight_rings(drawer, hml, mol)

    highlight_without_rings(drawer, hml, mol, legend)

    if isinstance(drawer, Draw.MolDraw2DSVG):
        svg = get_rdkit_svg(drawer)
        tree = read_tree(svg)
        if not assure_elem(tree):  # pragma: no cover
            # defensive: RDKit-generated SVG always parses to a root element.
            raise ValueError("No root element found")
        hide_objects(tree, ["rect"])
        if not legend:
            # skip cropping when legend is present to prevent clipping.
            remove_whitespace(tree)
        svg = to_string(tree)
        if legend:
            svg = _bold_legend(svg)
        return svg.encode()  # type: ignore[no-any-return,unused-ignore]
    return drawer.GetDrawingText()  # type: ignore[no-any-return]


def export_rdkit_molecule(mol_backend: RDKitMolecule, fmt: OutputFormat, use_v2000: bool) -> bytes:  # noqa: PLR0911
    """Serialize RDKit molecule to specified format."""
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
    """Wraps RDKit molecules for highlight backend compatibility."""

    def __init__(self, mol: Chem.Mol) -> None:
        """Initialize wrapper with RDKit molecule."""
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

        return self._edit_state

    @override
    def set_edit_state(self, kekulized: bool, aligned: bool, hydrogens_hidden: bool) -> None:
        """Update edit state."""
        self._edit_state = kekulized, aligned, hydrogens_hidden

    @override
    def get_hml_json(self) -> str | None:
        """Return JSON-encoded HML object."""
        return self._hml_json

    @override
    def set_hml_json(self, hml_json: str) -> None:
        """Update JSON-encoded HML object."""
        self._hml_json = hml_json

    @override
    def get_label(self) -> str | None:
        """Return label text if set."""
        return self._label

    @override
    def set_label(self, label: str) -> None:
        """Update label text."""
        self._label = label

    @override
    def add_label_callback(self, text: str) -> None:
        """No-op; label is rendered from state during export."""

    @override
    @classmethod
    def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Create document from byte data."""
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
            # RXN, CDX.
            raise InputFormatNotSupported(f"{cls.__name__} does not support importing from {fmt}")
        if not mol or mol.GetNumAtoms() < 1:
            raise ValueError(f"Invalid {fmt} input")
        return cls(mol)

    @override
    def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:
        """Export molecule to specified format."""
        return export_rdkit_molecule(self, fmt, use_v2000)

    @override
    def align_to_reference_ratio_callback(self, width: float, height: float) -> None:
        """Align molecule to specified bounding box ratio."""
        atol = 1e-5
        query = get_2d_mol(mol_from_molblock(self.to_molblock()), atol=atol)
        angle = get_best_fit_angle(query, (width, height))
        self.mol = apply_transform(query, angle, flip_horizontal=False, atol=atol)

    @override
    def align_to_reference_mol_callback(self, reference_molblock: str) -> None:
        """Align molecule to reference molblock."""
        atol = 1e-5
        query = get_2d_mol(mol_from_molblock(self.to_molblock()), atol=atol)
        reference = get_2d_mol(mol_from_molblock(reference_molblock), atol=atol)
        flips, transform = get_alignment_flips_and_transform(query, reference, atol=atol)
        global_flip, angle = parse_transform(transform, atol=atol)
        for bond_ix, anchor_atom_ix in flips:
            query = flip_bond(query, bond_ix, anchor_atom_ix, atol=atol)
        query = apply_transform(query, angle, flip_horizontal=global_flip, atol=atol)
        self.mol = query

    @override
    def cleanup_callback(self) -> None:
        """Sanitize molecule and compute 2D coordinates."""
        from rdkit import Chem
        from rdkit.Chem.rdDepictor import Compute2DCoords
        from rdkit.Chem.rdmolops import SanitizeMol

        SanitizeMol(self.mol, Chem.SANITIZE_ALL)
        Compute2DCoords(self.mol, clearConfs=True)

    @override
    def kekulize_callback(self, kekulize: bool) -> None:
        """Kekulize or dekekulize molecule."""
        from rdkit import Chem
        from rdkit.Chem.rdmolops import SanitizeMol

        if kekulize:
            Chem.Kekulize(self.mol)
        else:
            SanitizeMol(self.mol, Chem.SANITIZE_SETAROMATICITY)

    @override
    def hide_hydrogens_callback(self) -> None:
        """Hide featureless hydrogens."""
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

        if hide_hydrogens:
            self.hide_hydrogens_callback()


class RDKitDocument(Document[RDKitMolecule]):
    """Container for molecules parsed using RDKit."""

    @classmethod
    @override
    def _split_bytes(cls, data: bytes, fmt: InputFormat) -> list[RDKitMolecule]:
        from rdkit import Chem

        mols: list[Chem.Mol | None]
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
            smiles_mol: Chem.Mol | None = Chem.MolFromSmiles(data.decode(), params)
            mols = (
                list(Chem.GetMolFrags(smiles_mol, asMols=True, sanitizeFrags=False))
                if smiles_mol
                else []
            )
        else:
            # reuse single-molecule backend construction and restrictions for MOL, CDX, and InChI.
            return [RDKitMolecule.from_bytes(data, fmt)]

        valid_mols = [mol for mol in mols if mol is not None and mol.GetNumAtoms() >= 1]
        if not valid_mols:
            raise ValueError(f"Invalid {fmt} input")
        return [RDKitMolecule(mol) for mol in valid_mols]

    @override
    def _combined_export(self, fmt: Literal["SVG", "PNG"]) -> bytes:
        combined = self._combine_for_export()
        return draw_mol(None, combined, "svg" if fmt == "SVG" else "png")

    def _combine_for_export(self) -> Chem.Mol:
        """Combine molecules into single translated Chem.Mol."""
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
        # perform ring perception for correct rendering.
        Chem.GetSSSR(combined)
        return combined


class AsyncRDKitMolecule(AsyncHighlightBackendMolecule):
    """RDKit backend that executes calls in a thread pool to avoid blocking event loop."""

    def __init__(self, sync: RDKitMolecule) -> None:
        """Wrap synchronous RDKit molecule."""
        self._sync = sync

    @override
    def get_hml_json(self) -> str | None:
        return self._sync.get_hml_json()

    @override
    def set_hml_json(self, hml_json: str) -> None:
        self._sync.set_hml_json(hml_json)

    @override
    def get_edit_state(self) -> tuple[bool, bool, bool]:
        return self._sync.get_edit_state()

    @override
    def set_edit_state(self, kekulized: bool, aligned: bool, hydrogens_hidden: bool) -> None:
        self._sync.set_edit_state(kekulized, aligned, hydrogens_hidden)

    @override
    def get_label(self) -> str | None:
        return self._sync.get_label()

    @override
    def set_label(self, label: str) -> None:
        self._sync.set_label(label)

    @override
    @classmethod
    async def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        loop = asyncio.get_running_loop()
        sync = await loop.run_in_executor(None, RDKitMolecule.from_bytes, data, fmt)
        return cls(sync)

    @override
    async def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.export, fmt, use_v2000)

    @override
    async def align_to_reference_ratio_callback(self, width: float, height: float) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sync.align_to_reference_ratio_callback, width, height
        )

    @override
    async def align_to_reference_mol_callback(self, reference_molblock: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sync.align_to_reference_mol_callback, reference_molblock
        )

    @override
    async def cleanup_callback(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync.cleanup_callback)

    @override
    async def kekulize_callback(self, kekulize: bool) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync.kekulize_callback, kekulize)

    @override
    async def hide_hydrogens_callback(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync.hide_hydrogens_callback)

    @override
    async def highlight_from_json_callback(
        self, hml_json: str, hide_hydrogens: bool = False
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sync.highlight_from_json_callback, hml_json, hide_hydrogens
        )

    @override
    async def add_label_callback(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync.add_label_callback, text)


class AsyncRDKitDocument(AsyncDocument[AsyncRDKitMolecule]):
    """RDKit backend for `AsyncDocument` that delegates logic to `RDKitDocument` via thread pool."""

    @classmethod
    @override
    async def _split_bytes(cls, data: bytes, fmt: InputFormat) -> list[AsyncRDKitMolecule]:
        loop = asyncio.get_running_loop()
        sync_mols = await loop.run_in_executor(None, RDKitDocument._split_bytes, data, fmt)  # noqa: SLF001
        return [AsyncRDKitMolecule(m) for m in sync_mols]

    @override
    async def _combined_export(self, fmt: Literal["SVG", "PNG"]) -> bytes:
        sync_doc = RDKitDocument(
            [m._sync for m in self.molecules],  # noqa: SLF001
            self.source_format,
            self.offsets,
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, sync_doc._combined_export, fmt)  # noqa: SLF001
