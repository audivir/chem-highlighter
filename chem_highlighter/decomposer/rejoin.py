"""Rejoin decomposed molecules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from chem_highlighter.utils import get_atoms, get_neighbors, mol_from_smiles, mol_to_smiles

if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType


def get_sole_neighbor(atom: Chem.Atom) -> Chem.Atom:
    """Get the sole neighboring atom of the given RDKit atom.

    Raises:
        ValueError: If the molecule has no or more than one neighbor.
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
    """Get the indices of the atoms with the atom map number `on`.

    Args:
        mol: The molecule.
        on: The atom map number.
        num: The number of atoms. Defaults to 2.

    Raises:
        ValueError: If the number of atoms with the atom map number `on` is not `num`.
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
    """Mask hydrogens next to mapped atoms with a non-hydrogen symbol.

    Raises:
        ValueError: If the mask symbol is already in the molecule.
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
    """Remove all special (200+, 300+) isotopes from the molecule."""
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
    """Join a pair of molecules on a [*:`on`] marked atom.

    Usage:
        >>> mol_a = Chem.MolFromSmiles("CC[*:1]")
        >>> mol_b = Chem.MolFromSmiles("CO[*:1]")
        >>> mol_c = join_mols(mol_a, mol_b, on=1)
        >>> Chem.MolToSmiles(mol_c)
        'CCOC'

    Args:
        mol_a: The first molecule.
        mol_b: The second molecule.
        on: The atom map number.
        order: The bond order. Defaults to `BondType.SINGLE`.

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

    # Remove old bonds
    mol_c.RemoveBond(a_atom.GetIdx(), a_neighbor.GetIdx())
    mol_c.RemoveBond(b_atom.GetIdx(), b_neighbor.GetIdx())

    # Add new bond
    mol_c.AddBond(a_neighbor.GetIdx(), b_neighbor.GetIdx(), order=order)

    # Remove the mapped atoms
    mol_c.RemoveAtom(a_atom.GetIdx())

    c_atom = get_mapped_atoms(mol_c, on, 1)

    mol_c.RemoveAtom(c_atom.GetIdx())

    return mol_c.GetMol()


def join_multiple(core: Chem.Mol, groups: dict[str, Chem.Mol]) -> Chem.Mol:
    """Join multiple molecules to a core.

    Usage:
        >>> core = Chem.MolFromSmiles("C(O[*:2])C[*:1]")
        >>> groups = {
        ...     "R1": Chem.MolFromSmiles("CO[*:1]"),
        ...     "R2": Chem.MolFromSmiles("CC[*:2]"),
        ... }
        >>> joined = join_multiple(core, groups)
        >>> Chem.MolToSmiles(joined)
        'C(OCC)COC'

    Args:
        core: The core molecule with [*:`on`] marked atoms.
        groups: The groups to join as with [*:`on`] marked atoms and `R{on}` keys.

    Returns:
        The joined molecule.
    """
    for key, mol in groups.items():
        if not key.startswith("R"):  # pragma: no cover
            continue
        on = int(key.removeprefix("R"))
        core = join_mols(core, mol, on=on)

    return mol_from_smiles(mol_to_smiles(core))
