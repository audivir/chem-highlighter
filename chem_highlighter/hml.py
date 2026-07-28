"""Utilities for highlighting chemical molecules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean
from typing import TYPE_CHECKING, Literal, TypeAlias, final

import msgspec
from typing_extensions import Self, TypeVar

from chem_highlighter.align import get_alignment_ops_from_molblock

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from rdkit import Chem

    from chem_highlighter.backend.rdkit import RDKitDocument
    from chem_highlighter.utils import RGBA


HighlightBackendDocumentT_co = TypeVar(
    "HighlightBackendDocumentT_co",
    bound="HighlightBackendDocument",
    default="RDKitDocument",
    covariant=True,
)

InputFormat: TypeAlias = Literal["SDF", "Mol", "RXN", "CDX", "CDXML", "SMILES", "InChI"]
OutputFormat: TypeAlias = InputFormat | Literal["InChIKey", "SVG", "EPS", "PNG"]


class InputFormatNotSupported(Exception):  # noqa: N818
    """The specified input format is not supported by the backend."""


class OutputFormatNotSupported(Exception):  # noqa: N818
    """The specified output format is not supported by the backend."""


class HML(msgspec.Struct, kw_only=True):
    """A structure highlighting options for a molecule."""

    highlighted_atoms: dict[int, int] = {}
    highlighted_bonds: dict[int, int] = {}
    highlighted_rings: dict[int, int] = {}
    rings: list[list[int]] = []
    palette: list[str] = []

    def get_rgba(self, group_ix: int) -> RGBA:
        """Return the color for the group as RGBA tuple."""
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
        """Build highlighted objects from multi-colored highlights."""
        palette: dict[str, int] = {}
        return cls(
            highlighted_atoms=cls._from_multicolor(atoms, palette),
            highlighted_bonds=cls._from_multicolor(bonds, palette),
            highlighted_rings=cls._from_multicolor(rings, palette),
            rings=[list(r) for r in rings_ixs],
            palette=list(palette),
        )


class HMol(HML):
    """A structure to hold the serialized molecule as Mol block and its highlighting options."""

    mol: str


class HighlightBackendDocument(ABC):
    """Abstract base class for highlight backend documents."""

    @abstractmethod
    def get_hml_json(self) -> str | None:
        """Get the JSON-encoded HML object."""

    @abstractmethod
    def set_hml_json(self, hml_json: str) -> None:
        """Set the JSON-encoded HML object."""

    @abstractmethod
    def get_edit_state(self) -> tuple[bool, bool, bool]:
        """Get the edit state tuple."""

    @abstractmethod
    def set_edit_state(self, kekulized: bool, aligned: bool, hydrogens_hidden: bool) -> None:
        """Set the edit state."""

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes, fmt: InputFormat) -> Self:
        """Create a document from byte data."""

    @final
    @classmethod
    def from_string(cls, data: str, fmt: InputFormat) -> Self:
        """Create a document from string data."""
        return cls.from_bytes(data.encode(), fmt)

    @abstractmethod
    def export(self, fmt: OutputFormat, use_v2000: bool = False) -> bytes:
        """Export the document to the specified format as bytes."""

    @final
    def export_string(self, fmt: OutputFormat, use_v2000: bool = False) -> str:
        """Export the document to the specified format as a string."""
        return self.export(fmt, use_v2000).decode()

    @abstractmethod
    def align_to_reference_callback(
        self, flips: list[tuple[int, int]], global_flip: bool, angle: float
    ) -> None:
        """Run after alignment checks are done by `align_to_reference`."""

    @final
    def align_to_reference(self, reference: str) -> Self:
        """Align the underlying molecule to a reference molecule as molblock."""
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if aligned:
            raise ValueError("Already aligned")
        flips, global_flip, angle = get_alignment_ops_from_molblock(
            self.to_molblock(), reference, atol=1e-5
        )
        self.align_to_reference_callback(flips, global_flip, angle)
        self.set_edit_state(kekulized, True, hydrogens_hidden)
        return self

    @abstractmethod
    def cleanup_callback(self) -> None:
        """Run after cleanup checks are done by `cleanup`."""

    @final
    def cleanup(self) -> Self:
        """Cleanup the molecule using the backend's features."""
        kekulized, aligned, _ = self.get_edit_state()
        if aligned:
            raise ValueError("Cleanup after alignment not supported")
        self.cleanup_callback()
        self.kekulize(kekulized)
        return self

    @abstractmethod
    def kekulize_callback(self, kekulize: bool) -> None:
        """Run after kekulization checks are done by `kekulize`."""

    @final
    def kekulize(self, kekulize: bool) -> Self:
        """Kekulize or dekekulize the underlying molecule."""
        self.kekulize_callback(kekulize)
        return self

    @abstractmethod
    def hide_hydrogens_callback(self) -> None:
        """Run after the checks in `hide_hydrogens` pass."""

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
    def highlight_from_json_callback(self, hml_json: str, hide_hydrogens: bool) -> None:
        """Run after highlighting options are set by `highlight_from_json`."""

    @final
    def highlight_from_json(self, hml_json: str, hide_hydrogens: bool) -> Self:
        """Set the underlying highlighting options and run the backend's callback."""
        if self.get_hml_json():
            raise ValueError("Already highlighted")
        kekulized, aligned, hydrogens_hidden = self.get_edit_state()
        if hydrogens_hidden:
            raise ValueError("Highlighting after setting hydrogen display not supported")
        self.highlight_from_json_callback(hml_json, hide_hydrogens)
        self.set_hml_json(hml_json)
        self.set_edit_state(kekulized, aligned, hide_hydrogens)
        return self

    @final
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create a document from a provided molecule as RDkit molecule."""
        from rdkit import Chem

        return cls.from_molblock(Chem.MolToMolBlock(mol, forceV3000=True))

    @final
    @classmethod
    def from_molblock(cls, molblock: str) -> Self:
        """Create a document from a provided molecule as molblock."""
        return cls.from_string(molblock, "Mol")

    @final
    def to_molblock(self) -> str:
        """Return the underlying molecule as molblock."""
        return self.export_string("Mol")

    @final
    def to_svg(self) -> str:
        """Return a highlighted (if set) SVG visualization of the molecule."""
        return self.export_string("SVG")

    @final
    def to_png(self) -> bytes:
        """Return a highlighted (if set) PNG visualization of the molecule."""
        return self.export("PNG")

    def to_console(self, canonical: bool = True) -> str:
        """Return a highlighted (if set) string visualization of the molecule."""
        from rdkit import Chem

        smiles = self.export_string("SMILES")
        if canonical:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        return smiles

    @final
    def to_hmol_json(self) -> str:
        """Return a JSON-encoded of the molecule including its highlighting options."""
        hmol = HMol(mol=self.to_molblock())
        if hml_json := self.get_hml_json():
            hml = msgspec.json.Decoder(HML).decode(hml_json)
            for field in hml.__struct_fields__:
                setattr(hmol, field, getattr(hml, field))
        return msgspec.json.encode(hmol).decode()
