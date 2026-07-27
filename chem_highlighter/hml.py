"""Utilities for highlighting chemical molecules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean
from typing import TYPE_CHECKING, NamedTuple, final

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


class EditState(NamedTuple):
    """A tuple to store the edit state."""

    kekulized: bool | None
    aligned: bool
    hydrogen_display_set: bool


class HighlightBackendDocument(ABC):
    """A structure to store a molecule for highlighting."""

    @abstractmethod
    def get_edit_state(self) -> EditState:
        """Get the edit state tuple."""

    @abstractmethod
    def set_edit_state(self, edit_state: EditState) -> None:
        """Set the edit state."""

    @abstractmethod
    def get_hml(self) -> HML | None:
        """Get the HML object."""

    @abstractmethod
    def set_hml(self, hml: HML) -> None:
        """Set the HML object."""

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
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def align_to_reference_callback(self, reference: str) -> None:
        """Run after alignment checks are done by `align_to_reference`."""

    @final
    def align_to_reference(self, reference: str) -> Self:
        """Align the underlying molecule to a reference molecule.

        Args:
            reference: The reference molecule as Mol block.
        """
        edit_state = self.get_edit_state()
        if edit_state.aligned:
            raise ValueError("Already aligned")
        self.set_edit_state(edit_state._replace(aligned=True))
        self.align_to_reference_callback(reference)
        return self

    @abstractmethod
    def cleanup_callback(self) -> None:
        """Run after cleanup checks are done by `cleanup`."""

    @final
    def cleanup(self) -> Self:
        """Cleanup the molecule using the backend's features."""
        edit_state = self.get_edit_state()
        if edit_state.kekulized is not None or edit_state.aligned:
            raise ValueError("Cleanup after kekulization or alignment not supported")
        self.cleanup_callback()
        return self

    @abstractmethod
    def kekulize_callback(self, kekulize: bool) -> None:
        """Run after kekulization checks are done by `kekulize`."""

    @final
    def kekulize(self, kekulize: bool) -> Self:
        """Kekulize or dekekulize the underlying molecule."""
        edit_state = self.get_edit_state()
        if edit_state.kekulized is not None:
            raise ValueError("Already kekulized")
        self.set_edit_state(edit_state._replace(kekulized=kekulize))
        self.kekulize_callback(kekulize)
        return self

    @abstractmethod
    def set_hydrogen_display_callback(self, show_hydrogens: bool) -> None:
        """Run after hydrogen options are set by `set_hydrogen_display`."""

    @final
    def set_hydrogen_display(self, show_hydrogens: bool) -> Self:
        """Show or hide hydrogens."""
        if self.get_hml():
            raise ValueError("Setting hydrogen display after highlighting not supported")
        edit_state = self.get_edit_state()
        if edit_state.hydrogen_display_set:
            raise ValueError("Hydrogen display already set")
        self.set_edit_state(edit_state._replace(hydrogen_display_set=True))
        self.set_hydrogen_display_callback(show_hydrogens)
        return self

    @abstractmethod
    def highlight_from_json_callback(self, hml_json: str, show_hydrogens: bool | None) -> None:
        """Run after highlighting options are set by `highlight_from_json`."""

    @final
    def highlight_from_json(self, hml_json: str, show_hydrogens: bool | None) -> Self:
        """Set the underlying highlighting options and run the backend's callback."""
        if self.get_hml():
            raise ValueError("Already highlighted")
        edit_state = self.get_edit_state()
        if edit_state.hydrogen_display_set:
            raise ValueError("Highlighting after setting hydrogen display not supported")
        if show_hydrogens is not None:
            self.set_edit_state(edit_state._replace(hydrogen_display_set=True))
        self.set_hml(msgspec.json.Decoder(HML).decode(hml_json))
        self.highlight_from_json_callback(hml_json, show_hydrogens)
        return self

    @final
    def to_hmol_json(self) -> str:
        """Return a JSON-encoded of the molecule including its highlighting options."""
        hmol = HMol(mol=self.to_molblock())
        if hml := self.get_hml():
            for field in hml.__struct_fields__:
                setattr(hmol, field, getattr(hml, field))
        return msgspec.json.encode(hmol).decode()
