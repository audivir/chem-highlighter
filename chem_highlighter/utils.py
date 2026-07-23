"""Utilities for the decomposer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem

logger = logging.getLogger()

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


def is_same_conformer(  # noqa: PLR0911
    molblock_a: str,
    molblock_b: str,
    atol: float = 1e-3,
) -> bool:
    """Are two molblocks the same conformer."""
    import numpy as np
    from rdkit import Chem
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    mol_a: Chem.Mol | None = Chem.MolFromMolBlock(molblock_a, removeHs=False)
    mol_b: Chem.Mol | None = Chem.MolFromMolBlock(molblock_b, removeHs=False)

    if not mol_a or not mol_b:
        raise ValueError("Invalid molblocks")

    if Chem.MolToSmiles(mol_a) != Chem.MolToSmiles(mol_b):
        logger.error("Non-identical molecules")
        return False

    # Basic topology
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms() or mol_a.GetNumBonds() != mol_b.GetNumBonds():
        logger.error("Non-identical number of atoms or bonds")
        return False

    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()

    pos_a = np.array(conf_a.GetPositions())
    pos_b = np.array(conf_b.GetPositions())

    # Cost matrix of distances
    cost_matrix = cdist(pos_a, pos_b)

    # Find minimum-cost one-to-one assignment
    rows, cols = linear_sum_assignment(cost_matrix)

    mapping = {int(c): int(r) for c, r in zip(cols, rows, strict=True)}  # expected -> actual

    max_dist = max(cost_matrix[rows, cols])
    if max_dist >= atol:
        logger.error("Positions off by %f", max_dist)
        return False

    for exp_idx, act_idx in mapping.items():
        atom_a = mol_a.GetAtomWithIdx(act_idx)
        atom_b = mol_b.GetAtomWithIdx(exp_idx)

        atom_data_a, atom_data_b = [
            (a.GetAtomicNum(), a.GetIsAromatic(), a.GetFormalCharge()) for a in (atom_a, atom_b)
        ]
        if atom_data_a != atom_data_b:
            logger.error("Non-identical atom data: %s != %s", atom_data_a, atom_data_b)
            return False

    for bond in get_bonds(mol_b):
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]

        other: Chem.Bond | None = mol_a.GetBondBetweenAtoms(a1, a2)
        if not other:
            logger.error("No bond found")
            return False

        bond_data_a, bond_data_b = [(b.GetBondType(), b.GetIsAromatic()) for b in (bond, other)]

        if bond_data_a != bond_data_b:
            logger.error("Non-identical bond data: %s != %s", bond_data_a, bond_data_b)
            return False

    return True
