"""Rejoin decomposed molecular fragments into a complete structure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from chem_highlighter.utils import get_atoms, get_neighbors, mol_from_smiles, mol_to_smiles

if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType


def get_sole_neighbor(atom: Chem.Atom) -> Chem.Atom:
    """Retrieve the single neighboring atom of the given atom.

    Raises:
        ValueError: If the atom has zero or multiple neighbors.
    """
    neighbors = get_neighbors(atom)
    if len(neighbors) != 1:
        raise ValueError(f"The atom with the index {atom.GetIdx()} has {len(neighbors)} neighbors.")
    return neighbors[0]


@overload
def get_mapped_atoms(mol: Chem.Mol, on: int, num: Literal[1] = ...) -> Chem.Atom: ...


@overload
def get_mapped_atoms(mol: Chem.Mol, on: int, num: Literal[2]) -> tuple[Chem.Atom, Chem.Atom]: ...


def get_mapped_atoms(
    mol: Chem.Mol, on: int, num: Literal[1, 2] = 2
) -> tuple[Chem.Atom, Chem.Atom] | Chem.Atom:
    """Retrieve atoms with the specified atom map number.

    Args:
        mol: Target molecule.
        on: Atom map number to search for.
        num: Required number of atoms to find.

    Raises:
        ValueError: If the count of atoms with map number `on` does not equal `num`.
    """
    mapped_atoms = [atom for atom in get_atoms(mol) if atom.GetAtomMapNum() == on]
    if len(mapped_atoms) != num:
        raise ValueError(f"The number of atoms with the atom map number {on} is not {num}.")

    if num == 1:
        return mapped_atoms[0]

    if num == 2:  # noqa: PLR2004
        return mapped_atoms[0], mapped_atoms[1]

    raise ValueError("Invalid number of atoms.")  # pragma: no cover


def mask_hydrogens(mol: Chem.Mol, on: int, symbol: str = "I") -> Chem.Mol:
    """Replace hydrogens adjacent to mapped atoms with a specified non-hydrogen symbol.

    Args:
        mol: Target molecule.
        on: Atom map number to identify the target atom.
        symbol: Element symbol used for masking.

    Raises:
        ValueError: If the mask symbol is already present in the molecule.
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import GetPeriodicTable

    atomic_num = GetPeriodicTable().GetAtomicNumber(symbol)

    mol = Chem.Mol(mol)

    if any(
        atom.GetAtomicNum() == atomic_num and atom.GetIsotope() < 200  # noqa: PLR2004
        for atom in get_atoms(mol)
    ):
        raise ValueError("Mask symbol already in the molecule")

    atom = get_mapped_atoms(mol, on, num=1)
    neighbor = get_sole_neighbor(atom)

    if neighbor.GetSymbol() == "H":
        neighbor.SetAtomicNum(atomic_num)
        neighbor.SetIsotope(400 + neighbor.GetIsotope())

    return mol


def clear_special_isotopes(mol: Chem.Mol) -> Chem.Mol:
    """Normalize special isotopes by removing offsets of 200 or 400."""
    from rdkit import Chem

    mol = Chem.Mol(mol)
    for atom in get_atoms(mol):
        isotope = atom.GetIsotope()
        if isotope < 200:  # noqa: PLR2004
            continue
        if isotope >= 400:  # noqa: PLR2004
            isotope -= 400
        else:
            isotope -= 200
        atom.SetIsotope(isotope)
    return mol


def join_mols(mol_a: Chem.Mol, mol_b: Chem.Mol, on: int, order: BondType | None = None) -> Chem.Mol:
    """Join two molecules at atoms sharing the specified atom map number.

    Args:
        mol_a: First molecule.
        mol_b: Second molecule.
        on: Atom map number to join on.
        order: Bond order for the new connection.

    Returns:
        The joined molecule.
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType
    from rdkit.Chem.rdmolops import CombineMols

    if order is None:
        order = BondType.SINGLE

    if not on > 0:  # pragma: no cover
        raise ValueError("The atom map number must be greater than 0.")

    mol_c = CombineMols(mol_a, mol_b)
    mol_c = Chem.RWMol(mol_c)
    a_atom, b_atom = get_mapped_atoms(mol_c, on, 2)

    a_neighbor = get_sole_neighbor(a_atom)
    b_neighbor = get_sole_neighbor(b_atom)

    mol_c.RemoveBond(a_atom.GetIdx(), a_neighbor.GetIdx())
    mol_c.RemoveBond(b_atom.GetIdx(), b_neighbor.GetIdx())

    mol_c.AddBond(a_neighbor.GetIdx(), b_neighbor.GetIdx(), order=order)

    # remove mapped atoms.
    mol_c.RemoveAtom(a_atom.GetIdx())

    c_atom = get_mapped_atoms(mol_c, on, 1)

    mol_c.RemoveAtom(c_atom.GetIdx())

    return mol_c.GetMol()


def join_multiple(core: Chem.Mol, groups: dict[str, Chem.Mol]) -> Chem.Mol:
    """Join multiple molecules to a core molecule.

    Args:
        core: Core molecule containing atoms marked with map numbers.
        groups: Mapping of group identifiers (e.g., 'R1') to molecules containing
            corresponding map numbers.

    Returns:
        The joined molecule.
    """
    for key, mol in groups.items():
        if not key.startswith("R"):  # pragma: no cover
            continue
        on = int(key.removeprefix("R"))
        core = join_mols(core, mol, on=on)

    return mol_from_smiles(mol_to_smiles(core))
