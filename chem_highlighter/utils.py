"""Utilities for the decomposer."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem

RGBA: TypeAlias = tuple[float, float, float, float]  # pragma: no cover

RED_COLOR = "\033[91m"
GREEN_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"


class SmilesMolPair(NamedTuple):
    """Tuple to hold SMILES string and RDKit molecule."""

    smiles: str
    mol: Chem.Mol


def get_atoms(mol: Chem.Mol) -> tuple[Chem.Atom, ...]:
    """Return a tuple of the atoms of a RDKit molecule."""
    return mol.GetAtoms()  # type: ignore[no-any-return,no-untyped-call]


def get_bonds(mol: Chem.Mol) -> tuple[Chem.Bond, ...]:
    """Return a tuple of the bonds of a RDKit molecule."""
    return mol.GetBonds()  # type: ignore[no-any-return,no-untyped-call]


def get_neighbors(atom: Chem.Atom) -> tuple[Chem.Atom, ...]:
    """Return a tuple of the neighbor atoms of a RDKit atom."""
    return atom.GetNeighbors()


def add_hydrogens(data: Sequence[str | Chem.Mol]) -> list[Chem.Mol]:
    """Add hydrogens to the molecules."""
    from rdkit import Chem

    return [Chem.AddHs(get_smiles_mol_pair(d).mol, addCoords=True) for d in data]


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Convert a SMILES string safely to a RDKit molecule."""
    from rdkit import Chem

    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    if not mol or mol.GetNumAtoms() < 1:
        raise ValueError("Invalid SMILES")
    return mol


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Convert a RDKit molecule safely to a SMILES string."""
    from rdkit import Chem

    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:  # pragma: no cover
        raise ValueError("Conversion to SMILES failed") from e
    if not smiles:
        raise ValueError("Empty SMILES")
    return smiles


def get_smiles_mol_pair(data: str | Chem.Mol) -> SmilesMolPair:
    """Return a pair of SMILES string and corresponding RDKit molecule."""
    from rdkit import Chem

    if isinstance(data, Chem.Mol):
        return SmilesMolPair(mol_to_smiles(data), data)
    if isinstance(data, str):
        return SmilesMolPair(data, mol_from_smiles(data))
    raise TypeError("Invalid input")  # pragma: no cover


def setup_cmap() -> list[RGBA]:  # pragma: no cover
    """Set up the colormap."""
    int_colors: list[tuple[int, int, int]] = [
        (86, 180, 233),
        (240, 228, 66),
        (0, 114, 178),
        (0, 158, 115),
        (204, 121, 167),
        (230, 159, 0),
        (213, 94, 0),
    ]

    return [(a / 255, b / 255, c / 255, 1.0) for a, b, c in int_colors]


def get_ansi_color(palette: Sequence[str], group_ix: int) -> str:
    """Get the ANSI color from color palette."""
    import matplotlib as mpl

    hex_color = palette[group_ix]
    r, g, b = [int(x * 255) for x in mpl.colors.hex2color(hex_color)]
    return f"\033[38;2;{r};{g};{b}m"
