"""Utilities for highlighting chemical molecules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean
from typing import TYPE_CHECKING

import msgspec
from typing_extensions import Self, TypeVar

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
    """A structure to store a molecule for highlighting."""

    hml: HML | None
    """Highlighting options or None if not highlighted."""

    @classmethod
    @abstractmethod
    def from_mol(cls, mol: Chem.Mol) -> Self:
        """Create a document from a provided molecule as RDkit molecule."""

    @classmethod
    @abstractmethod
    def from_molblock(cls, molblock: str) -> Self:
        """Create a document from a provided molecule as Mol block.

        Args:
            molblock: The Mol block to import.
        """

    @abstractmethod
    def to_molblock(self) -> str:
        """Return the underlying molecule as Mol block."""

    @abstractmethod
    def to_svg(self) -> str:
        """Return a highlighted (if set) SVG visualization of the molecule."""

    @abstractmethod
    def to_png(self) -> bytes:
        """Return a highlighted (if set) PNG visualization of the molecule."""

    def to_console(self, canonical: bool = True) -> str:
        """Return a highlighted (if set) string visualization of the molecule."""
        raise NotImplementedError

    @abstractmethod
    def align_to_reference(self, reference: str) -> None:
        """Align the underlying molecule to a reference molecule.

        Args:
            reference: The reference molecule as Mol block.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the molecule using the backend's features."""

    @abstractmethod
    def kekulize(self, kekulize: bool) -> None:
        """Kekulize or dekekulize the underlying molecule."""

    @abstractmethod
    def set_hydrogen_display(self, show_hydrogens: bool) -> None:
        """Show or hide featureless and non-highlighted hydrogens."""

    @abstractmethod
    def highlight_from_json_callback(self, hml_json: str) -> None:
        """Run after highlighting options are set by `highlight_from_json`."""

    def to_hmol_json(self) -> str:
        """Return a JSON-encoded of the molecule including its highlighting options."""
        hmol = HMol(mol=self.to_molblock())
        if self.hml:
            for field in self.hml.__struct_fields__:
                setattr(hmol, field, getattr(self.hml, field))
        return msgspec.json.encode(hmol).decode()

    def highlight_from_json(self, hml_json: str) -> None:
        """Set the underlying highlighting options and run the backend's callback."""
        self.hml = msgspec.json.Decoder(HML).decode(hml_json)
        self.highlight_from_json_callback(hml_json)
