"""Utilities for highlighting chemical molecules."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from statistics import mean
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, final

import msgspec
from typing_extensions import Self, TypeVar

from chem_highlighter.align import parse_possible_ratio
from chem_highlighter.backend.map_tokens import map_smiles_tokens
from chem_highlighter.utils import (
    RESET_COLOR,
    get_ansi_color,
    get_high_precision_v3000,
    is_same_layout,
    mol_from_molblock,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from rdkit import Chem

    from chem_highlighter.backend.rdkit import AsyncRDKitMolecule, RDKitMolecule
    from chem_highlighter.utils import RGBA

MAX_CLEANUPS = 100
CLEANUP_ATOL = 0.02
"""Largest atom deviation for two cleanups to count as the same layout."""

HighlightBackendMoleculeT_co = TypeVar(
    "HighlightBackendMoleculeT_co",
    bound="HighlightBackendMolecule",
    default="RDKitMolecule",
    covariant=True,
)
AsyncHighlightBackendMoleculeT_co = TypeVar(
    "AsyncHighlightBackendMoleculeT_co",
    bound="AsyncHighlightBackendMolecule",
    default="AsyncRDKitMolecule",
    covariant=True,
)

InputFormat: TypeAlias = Literal["SDF", "Mol", "RXN", "CDX", "CDXML", "SMILES", "InChI"]
OutputFormat: TypeAlias = InputFormat | Literal["InChIKey", "SVG", "EPS", "PNG"]


class InputFormatNotSupported(Exception):  # noqa: N818
    """Raised when backend does not support specified input format."""


class OutputFormatNotSupported(Exception):  # noqa: N818
    """Raised when backend does not support specified output format."""


class HML(msgspec.Struct, kw_only=True):
    """Encapsulates highlighting options for molecule."""

    highlighted_atoms: dict[int, int] = {}
    highlighted_bonds: dict[int, int] = {}
    highlighted_rings: dict[int, int] = {}
    rings: list[list[int]] = []
    palette: list[str] = []

    def get_rgba(self, group_ix: int) -> RGBA:
        """Retrieve RGBA color for specified group index."""
        import matplotlib as mpl

        return mpl.colors.to_rgba(self.palette[group_ix])

    @staticmethod
    def _from_multicolor(
        data: Mapping[int, Sequence[RGBA]], palette: MutableMapping[str, int]
    ) -> dict[int, int]:
        import matplotlib as mpl

        result: dict[int, int] = {}
        for ix, colors in data.items():
            mean_color: RGBA = tuple(mean(x) for x in zip(*colors, strict=True))
            hex_color = mpl.colors.to_hex(mean_color)
            result[ix] = palette.setdefault(hex_color, len(palette))
        return result

    @classmethod
    def from_multicolor(
        cls,
        atoms: Mapping[int, Sequence[RGBA]],
        bonds: Mapping[int, Sequence[RGBA]],
        rings: Mapping[int, Sequence[RGBA]],
        rings_ixs: Sequence[Sequence[int]],
    ) -> Self:
        """Create highlighted objects from multi-colored highlight mappings."""
        palette: dict[str, int] = {}
        return cls(
            highlighted_atoms=cls._from_multicolor(atoms, palette),
            highlighted_bonds=cls._from_multicolor(bonds, palette),
            highlighted_rings=cls._from_multicolor(rings, palette),
            rings=[list(r) for r in rings_ixs],
            palette=list(palette),
        )


class HighlightBackendMolecule(ABC):
    """Base class for highlight backend molecules."""

    @abstractmethod
    def get_hml_json(self) -> str | None:
        """Return JSON-encoded HML object."""

    @abstractmethod
    def set_hml_json(self, hml_json: str) -> None:
        """Update JSON-encoded HML object."""

    @abstractmethod
    def get_edit_state(self) -> tuple[bool, bool, bool]:
        """Return edit state."""

    @abstractmethod
    def set_edit_state(self, kekulized: bool, aligned: bool, hydrogens_hidden: bool) -> None:
        """Update edit state."""

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Initialize molecule from bytes."""

    @final
    @classmethod
    def from_string(cls, data: str, fmt: InputFormat) -> Self:
        """Initialize molecule from string."""
        return cls.from_bytes(data.encode(), fmt)

    @abstractmethod
    def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:
        """Serialize molecule to specified format."""

    @final
    def export_string(self, fmt: OutputFormat, use_v2000: bool = False) -> str:
        """Serialize molecule to specified format as string."""
        return self.export(fmt, use_v2000).decode()

    @abstractmethod
    def align_to_reference_ratio_callback(self, width: float, height: float) -> None:
        """Align bounding box to reference ratio."""

    @abstractmethod
    def align_to_reference_mol_callback(self, reference_molblock: str) -> None:
        """Align molecule to reference molblock."""

    @final
    def align_to_reference(self, reference: str) -> Self:
        """Align molecule to reference molblock or bounding box ratio."""
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if aligned:
            raise ValueError("Already aligned")
        ratio = parse_possible_ratio(reference)
        if ratio is not None:
            self.align_to_reference_ratio_callback(*ratio)
        else:
            self.align_to_reference_mol_callback(reference)
        self.set_edit_state(kekulized, True, hydrogens_hidden)
        return self

    @abstractmethod
    def cleanup_callback(self) -> None:
        """Execute post-cleanup operations."""

    @final
    def cleanup(self) -> Self:
        """Clean molecule using backend features."""
        kekulized, aligned, _ = self.get_edit_state()
        if aligned:
            raise ValueError("Cleanup after alignment not supported")

        self.kekulize_callback(kekulized)
        prev_cleans: list[str] = [self.to_molblock()]
        for _ in range(MAX_CLEANUPS):
            self.cleanup_callback()
            self.kekulize_callback(kekulized)
            curr_clean = self.to_molblock()
            if any(is_same_layout(curr_clean, c, atol=CLEANUP_ATOL) for c in reversed(prev_cleans)):
                break
            prev_cleans.append(curr_clean)
        else:  # pragma: no cover
            raise ValueError("Cleanup does not converge")

        return self

    @abstractmethod
    def kekulize_callback(self, kekulize: bool) -> None:
        """Execute post-kekulization operations."""

    @final
    def kekulize(self, kekulize: bool) -> Self:
        """Kekulize or dekekulize molecule."""
        self.kekulize_callback(kekulize)
        _, aligned, hydrogens_hidden = self.get_edit_state()
        self.set_edit_state(kekulize, aligned, hydrogens_hidden)
        return self

    @abstractmethod
    def hide_hydrogens_callback(self) -> None:
        """Execute post-hydrogen-hiding operations."""

    @final
    def hide_hydrogens(self) -> Self:
        """Hide featureless hydrogens."""
        if self.get_hml_json():
            raise ValueError("Setting hydrogen display after highlighting not supported")
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if hydrogens_hidden:
            raise ValueError("Hydrogen display already set")
        self.hide_hydrogens_callback()
        self.set_edit_state(kekulized, aligned, True)
        return self

    @abstractmethod
    def highlight_from_json_callback(self, hml_json: str, hide_hydrogens: bool = False) -> None:
        """Execute post-highlighting operations."""

    @final
    def highlight_from_json(self, hml_json: str, hide_hydrogens: bool = False) -> Self:
        """Apply highlighting options via backend callback."""
        if self.get_hml_json():
            raise ValueError("Already highlighted")
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if hydrogens_hidden:
            raise ValueError("Highlighting after setting hydrogen display not supported")
        self.set_hml_json(hml_json)
        self.highlight_from_json_callback(hml_json, hide_hydrogens)
        self.set_edit_state(kekulized, aligned, hide_hydrogens)
        return self

    @abstractmethod
    def get_label(self) -> str | None:
        """Return label text if defined."""

    @abstractmethod
    def set_label(self, label: str) -> None:
        """Set label text."""

    @abstractmethod
    def add_label_callback(self, text: str) -> None:
        """Execute callback after label validation."""

    @final
    def add_label(self, text: str) -> Self:
        """Add bold, centered label below molecule."""
        if self.get_label() is not None:
            raise ValueError("Already labeled")
        self.set_label(text)
        self.add_label_callback(text)
        return self

    @final
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create molecule from RDKit molecule."""
        return cls.from_molblock(get_high_precision_v3000(mol))

    @final
    @classmethod
    def from_molblock(cls, molblock: str) -> Self:
        """Create molecule from molblock string."""
        return cls.from_string(molblock, "Mol")

    def to_molblock(self) -> str:
        """Export molecule as molblock string."""
        return self.export_string("Mol")

    def to_svg(self) -> str:
        """Export molecule as SVG visualization with optional highlighting."""
        return self.export_string("SVG")

    def to_png(self) -> bytes:
        """Export molecule as PNG visualization with optional highlighting."""
        return self.export("PNG")

    def to_console(self, canonical: bool = True) -> str:
        """Export molecule as string visualization with optional highlighting."""
        os.environ["FORCE_COLOR"] = "1"
        kekulized, _, _ = self.get_edit_state()
        return _render_console(self.to_molblock(), kekulized, self.get_hml_json(), canonical)


def _render_console(molblock: str, kekulized: bool, hml_json: str | None, canonical: bool) -> str:
    """Render molecule as console-compatible string visualization."""
    from rdkit import Chem

    mol = mol_from_molblock(molblock)

    smiles = Chem.MolToSmiles(
        mol, kekuleSmiles=kekulized, canonical=canonical, allBondsExplicit=True
    )

    char_maps = map_smiles_tokens(smiles, mol)

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


class AsyncHighlightBackendMolecule(ABC):
    """Interface for asynchronous molecule highlighting backends."""

    @abstractmethod
    def get_hml_json(self) -> str | None:
        """Return JSON-encoded HML object."""

    @abstractmethod
    def set_hml_json(self, hml_json: str) -> None:
        """Set JSON-encoded HML object."""

    @abstractmethod
    def get_edit_state(self) -> tuple[bool, bool, bool]:
        """Return edit state."""

    @abstractmethod
    def set_edit_state(self, kekulized: bool, aligned: bool, hydrogens_hidden: bool) -> None:
        """Update edit state."""

    @classmethod
    @abstractmethod
    async def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Initialize molecule from byte data."""

    @final
    @classmethod
    async def from_string(cls, data: str, fmt: InputFormat) -> Self:
        """Initialize molecule from string data."""
        return await cls.from_bytes(data.encode(), fmt)

    @abstractmethod
    async def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:
        """Export molecule as bytes in specified format."""

    @final
    async def export_string(self, fmt: OutputFormat, use_v2000: bool = False) -> str:
        """Export molecule as string in specified format."""
        return (await self.export(fmt, use_v2000)).decode()

    @abstractmethod
    async def align_to_reference_ratio_callback(self, width: float, height: float) -> None:
        """Align bounding box to reference ratio."""

    @abstractmethod
    async def align_to_reference_mol_callback(self, reference_molblock: str) -> None:
        """Align molecule to reference molblock."""

    @final
    async def align_to_reference(self, reference: str) -> Self:
        """Align molecule to reference molblock or bounding box ratio."""
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if aligned:
            raise ValueError("Already aligned")
        ratio = parse_possible_ratio(reference)
        if ratio is not None:
            await self.align_to_reference_ratio_callback(*ratio)
        else:
            await self.align_to_reference_mol_callback(reference)
        self.set_edit_state(kekulized, True, hydrogens_hidden)
        return self

    @abstractmethod
    async def cleanup_callback(self) -> None:
        """Execute post-cleanup logic."""

    @final
    async def cleanup(self) -> Self:
        """Clean molecule using backend features."""
        kekulized, aligned, _ = self.get_edit_state()
        if aligned:
            raise ValueError("Cleanup after alignment not supported")

        await self.kekulize_callback(kekulized)
        prev_cleans: list[str] = [await self.to_molblock()]
        for _ in range(MAX_CLEANUPS):
            await self.cleanup_callback()
            await self.kekulize_callback(kekulized)
            curr_clean = await self.to_molblock()
            if any(is_same_layout(curr_clean, c, atol=CLEANUP_ATOL) for c in reversed(prev_cleans)):
                break
            prev_cleans.append(curr_clean)
        else:  # pragma: no cover
            raise ValueError("Cleanup does not converge")

        return self

    @abstractmethod
    async def kekulize_callback(self, kekulize: bool) -> None:
        """Execute post-kekulization logic."""

    @final
    async def kekulize(self, kekulize: bool) -> Self:
        """Set kekulization state of molecule."""
        await self.kekulize_callback(kekulize)
        _, aligned, hydrogens_hidden = self.get_edit_state()
        self.set_edit_state(kekulize, aligned, hydrogens_hidden)
        return self

    @abstractmethod
    async def hide_hydrogens_callback(self) -> None:
        """Execute post-hydrogen-hiding logic."""

    @final
    async def hide_hydrogens(self) -> Self:
        """Hide featureless hydrogens."""
        if self.get_hml_json():
            raise ValueError("Setting hydrogen display after highlighting not supported")
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if hydrogens_hidden:
            raise ValueError("Hydrogen display already set")
        await self.hide_hydrogens_callback()
        self.set_edit_state(kekulized, aligned, True)
        return self

    @abstractmethod
    async def highlight_from_json_callback(
        self, hml_json: str, hide_hydrogens: bool = False
    ) -> None:
        """Execute post-highlighting logic."""

    @final
    async def highlight_from_json(self, hml_json: str, hide_hydrogens: bool = False) -> Self:
        """Apply highlighting from JSON and execute backend callback."""
        if self.get_hml_json():
            raise ValueError("Already highlighted")
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if hydrogens_hidden:
            raise ValueError("Highlighting after setting hydrogen display not supported")
        self.set_hml_json(hml_json)
        await self.highlight_from_json_callback(hml_json, hide_hydrogens)
        self.set_edit_state(kekulized, aligned, hide_hydrogens)
        return self

    @abstractmethod
    def get_label(self) -> str | None:
        """Return label text if set."""

    @abstractmethod
    def set_label(self, label: str) -> None:
        """Set label text."""

    @abstractmethod
    async def add_label_callback(self, text: str) -> None:
        """Execute post-label validation logic."""

    @final
    async def add_label(self, text: str) -> Self:
        """Add bold, centered label below molecule."""
        if self.get_label() is not None:
            raise ValueError("Already labeled")
        self.set_label(text)
        await self.add_label_callback(text)
        return self

    @final
    @classmethod
    async def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create molecule from RDKit molecule."""
        return await cls.from_molblock(get_high_precision_v3000(mol))

    @final
    @classmethod
    async def from_molblock(cls, molblock: str) -> Self:
        """Create molecule from molblock string."""
        return await cls.from_string(molblock, "Mol")

    async def to_molblock(self) -> str:
        """Export molecule as molblock string."""
        return await self.export_string("Mol")

    async def to_svg(self) -> str:
        """Export molecule as SVG visualization with optional highlights."""
        return await self.export_string("SVG")

    async def to_png(self) -> bytes:
        """Export molecule as PNG visualization with optional highlights."""
        return await self.export("PNG")

    async def to_console(self, canonical: bool = True) -> str:
        """Export molecule as console string visualization with optional highlights."""
        os.environ["FORCE_COLOR"] = "1"
        kekulized, _, _ = self.get_edit_state()
        molblock = await self.to_molblock()
        hml_json = self.get_hml_json()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _render_console, molblock, kekulized, hml_json, canonical
        )


class Document(ABC, Generic[HighlightBackendMoleculeT_co]):
    """Base class for a collection of molecules parsed from single input and placed on a canvas."""

    def __init__(
        self,
        molecules: list[HighlightBackendMoleculeT_co],
        source_format: InputFormat,
        offsets: list[tuple[float, float]] | None = None,
    ) -> None:
        """Initialize document with a sequence of pre-constructed molecules."""
        if not molecules:
            raise ValueError("Document requires at least one molecule")
        if offsets is not None and len(offsets) != len(molecules):
            raise ValueError("offsets must have the same length as molecules")
        self.molecules = molecules
        self.source_format = source_format
        self.offsets = list(offsets) if offsets is not None else [(0.0, 0.0)] * len(molecules)

    def __len__(self) -> int:
        """Return number of molecules in document."""
        return len(self.molecules)

    def molecule(self, ix: int) -> HighlightBackendMoleculeT_co:
        """Retrieve molecule at index."""
        return self.molecules[ix]

    @classmethod
    @abstractmethod
    def _split_bytes(cls, data: bytes, fmt: InputFormat) -> list[HighlightBackendMoleculeT_co]:
        """Parse data into sequence of molecules."""

    @final
    @classmethod
    def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Create document from byte data by splitting multi-molecule input."""
        return cls(cls._split_bytes(data, fmt), fmt)

    @final
    @classmethod
    def from_string(cls, data: str, fmt: InputFormat) -> Self:
        """Create document from string."""
        return cls.from_bytes(data.encode(), fmt)

    @final
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create document from RDKit molecule."""
        return cls.from_molblock(get_high_precision_v3000(mol))

    @final
    @classmethod
    def from_molblock(cls, molblock: str) -> Self:
        """Create document from molblock."""
        return cls.from_string(molblock, "Mol")

    @abstractmethod
    def _combined_export(self, fmt: Literal["SVG", "PNG"]) -> bytes:
        """Render all molecules on single canvas using offsets."""

    def export(
        self, fmt: OutputFormat, use_v2000: bool = False, molecule_ix: int | None = None
    ) -> bytes:
        """Export document or specific molecule as bytes."""
        if molecule_ix is not None:
            return self.molecule(molecule_ix).export(fmt, use_v2000)
        if fmt == "SDF":
            return "".join(m.export_string("SDF", use_v2000) for m in self.molecules).encode()
        if fmt == "SMILES":
            return ".".join(m.export_string("SMILES") for m in self.molecules).encode()
        if fmt in ("SVG", "PNG"):
            return self._combined_export(fmt)
        if len(self.molecules) == 1:
            return self.molecule(0).export(fmt, use_v2000)
        raise OutputFormatNotSupported(
            f"{type(self).__name__} does not support exporting the whole document to {fmt}"
        )

    def export_string(
        self, fmt: OutputFormat, use_v2000: bool = False, molecule_ix: int | None = None
    ) -> str:
        """Export document or specific molecule as string."""
        return self.export(fmt, use_v2000, molecule_ix).decode()

    @final
    def to_molblock(self, molecule_ix: int | None = None) -> str:
        """Export document or specific molecule as molblock."""
        return self.export_string("Mol", molecule_ix=molecule_ix)

    @final
    def to_svg(self, molecule_ix: int | None = None) -> str:
        """Export document or specific molecule as SVG."""
        return self.export_string("SVG", molecule_ix=molecule_ix)

    @final
    def to_png(self, molecule_ix: int | None = None) -> bytes:
        """Export document or specific molecule as PNG."""
        return self.export("PNG", molecule_ix=molecule_ix)

    def to_console(self, canonical: bool = True, molecule_ix: int | None = None) -> str:
        """Export document or specific molecule as string."""
        if molecule_ix is not None:
            return self.molecule(molecule_ix).to_console(canonical=canonical)
        return ".".join(m.to_console(canonical=canonical) for m in self.molecules)


class AsyncDocument(ABC, Generic[AsyncHighlightBackendMoleculeT_co]):
    """Asynchronous base class for `Document`."""

    def __init__(
        self,
        molecules: list[AsyncHighlightBackendMoleculeT_co],
        source_format: InputFormat,
        offsets: list[tuple[float, float]] | None = None,
    ) -> None:
        """Initialize document with sequence of pre-constructed molecules."""
        if not molecules:
            raise ValueError("Document requires at least one molecule")
        if offsets is not None and len(offsets) != len(molecules):
            raise ValueError("offsets must have the same length as molecules")
        self.molecules = molecules
        self.source_format = source_format
        self.offsets = list(offsets) if offsets is not None else [(0.0, 0.0)] * len(molecules)

    def __len__(self) -> int:
        """Return number of molecules in document."""
        return len(self.molecules)

    def molecule(self, ix: int) -> AsyncHighlightBackendMoleculeT_co:
        """Retrieve molecule at index."""
        return self.molecules[ix]

    @classmethod
    @abstractmethod
    async def _split_bytes(
        cls, data: bytes, fmt: InputFormat
    ) -> list[AsyncHighlightBackendMoleculeT_co]:
        """Parse data into sequence of molecules."""

    @final
    @classmethod
    async def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Create document from byte data by splitting multi-molecule input."""
        return cls(await cls._split_bytes(data, fmt), fmt)

    @final
    @classmethod
    async def from_string(cls, data: str, fmt: InputFormat) -> Self:
        """Create document from string."""
        return await cls.from_bytes(data.encode(), fmt)

    @final
    @classmethod
    async def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create document from RDKit molecule."""
        return await cls.from_molblock(get_high_precision_v3000(mol))

    @final
    @classmethod
    async def from_molblock(cls, molblock: str) -> Self:
        """Create document from molblock."""
        return await cls.from_string(molblock, "Mol")

    @abstractmethod
    async def _combined_export(self, fmt: Literal["SVG", "PNG"]) -> bytes:
        """Render all molecules on single canvas using offsets."""

    async def export(
        self, fmt: OutputFormat, use_v2000: bool = False, molecule_ix: int | None = None
    ) -> bytes:
        """Export document or specific molecule as bytes."""
        if molecule_ix is not None:
            return await self.molecule(molecule_ix).export(fmt, use_v2000)
        if fmt == "SDF":
            parts = [await m.export_string("SDF", use_v2000) for m in self.molecules]
            return "".join(parts).encode()
        if fmt == "SMILES":
            parts = [await m.export_string("SMILES") for m in self.molecules]
            return ".".join(parts).encode()
        if fmt in ("SVG", "PNG"):
            return await self._combined_export(fmt)
        if len(self.molecules) == 1:
            return await self.molecule(0).export(fmt, use_v2000)
        raise OutputFormatNotSupported(
            f"{type(self).__name__} does not support exporting the whole document to {fmt}"
        )

    async def export_string(
        self, fmt: OutputFormat, use_v2000: bool = False, molecule_ix: int | None = None
    ) -> str:
        """Export document or specific molecule as string."""
        return (await self.export(fmt, use_v2000, molecule_ix)).decode()

    @final
    async def to_molblock(self, molecule_ix: int | None = None) -> str:
        """Export document or specific molecule as molblock string."""
        return await self.export_string("Mol", molecule_ix=molecule_ix)

    @final
    async def to_svg(self, molecule_ix: int | None = None) -> str:
        """Export document or specific molecule as SVG string."""
        return await self.export_string("SVG", molecule_ix=molecule_ix)

    @final
    async def to_png(self, molecule_ix: int | None = None) -> bytes:
        """Export document or specific molecule as PNG."""
        return await self.export("PNG", molecule_ix=molecule_ix)

    async def to_console(self, canonical: bool = True, molecule_ix: int | None = None) -> str:
        """Export document or specific molecule as console string."""
        if molecule_ix is not None:
            return await self.molecule(molecule_ix).to_console(canonical=canonical)
        parts = [await m.to_console(canonical=canonical) for m in self.molecules]
        return ".".join(parts)
