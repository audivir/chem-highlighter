"""Utilities for the decomposer."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray
    from rdkit import Chem
    from rdkit.Geometry import Point3D

logger = logging.getLogger()

RGBA: TypeAlias = tuple[float, float, float, float]  # pragma: no cover

RED_COLOR = "\033[91m"
GREEN_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_png_render_options() -> tuple[bool, float | None, float | None]:
    """Return `(transparent, width, height)` for PNG rendering.

    Read from CHEM_HIGHLIGHTER_PNG_TRANSPARENT/CHEM_HIGHLIGHTER_PNG_WIDTH/
    CHEM_HIGHLIGHTER_PNG_HEIGHT, so every backend (other implementations' own equivalent, this
    package's RDKit backend) renders PNGs to the same bounding box from the same knobs, with no
    per-call channel for it (neither `export(fmt)` nor other backends' own export requests carry
    per-call arguments). `width`/`height` are `None` when unset -- callers should then fall back
    to their own natural/unbounded size (RDKit's `-1, -1`; another backend's own SVG size) rather
    than a hardcoded default, so an unconfigured install renders the same as before this existed.
    If only one of width/height is set, the other mirrors it (a square bound).
    """
    transparent = os.environ.get("CHEM_HIGHLIGHTER_PNG_TRANSPARENT") == "true"
    width = _float_env("CHEM_HIGHLIGHTER_PNG_WIDTH")
    height = _float_env("CHEM_HIGHLIGHTER_PNG_HEIGHT")
    if width is None:
        width = height
    if height is None:
        height = width
    return transparent, width, height


class SmilesMolPair(NamedTuple):
    """Tuple to hold SMILES string and RDKit molecule."""

    smiles: str
    mol: Chem.Mol


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


def get_atom_position(conf: Chem.Conformer, ix: int) -> Point3D:
    """Get the 3D position of atom at `ix` of the conformer `conf`."""
    return conf.GetAtomPosition(ix)  # type: ignore[no-any-return]


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


def get_high_precision_v3000(mol: Chem.Mol, kekulize: bool = False) -> str:
    """Get an unrounded V3000 molblock."""
    from rdkit import Chem

    mol_block = Chem.MolToMolBlock(mol, kekulize=kekulize, forceV3000=True)

    if mol.GetNumConformers() == 0:
        return mol_block

    conf = mol.GetConformer()
    lines = mol_block.split("\n")

    in_atom_block = False
    atom_idx = 0

    for i, line in enumerate(lines):
        if line.startswith("M  V30 BEGIN ATOM"):
            in_atom_block = True
            continue
        if line.startswith("M  V30 END ATOM"):
            in_atom_block = False
            continue

        if in_atom_block and line.startswith("M  V30 "):
            pos = conf.GetAtomPosition(atom_idx)

            # M  V30 [idx] [element] [x] [y] [z] [aamap] ...
            parts = re.split(r"(\s+)", line)

            # Replace the rounded X, Y, Z (indices 4, 5, 6 in the split)
            # with 8-decimal precision floats
            parts[8] = f"{pos.x:.8f}"
            parts[10] = f"{pos.y:.8f}"
            parts[12] = f"{pos.z:.8f}"

            # Reconstruct the line
            lines[i] = "".join(parts)
            atom_idx += 1

    return "\n".join(lines)


def is_same_conformer(  # noqa: C901,PLR0912
    mol_or_molblock_a: Chem.Mol | str,
    mol_or_molblock_b: Chem.Mol | str,
    atol: float,
    quiet: bool = False,
) -> bool:
    """Are two molblocks the same conformer."""
    import numpy as np
    from rdkit import Chem
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    mol_a: Chem.Mol | None = (
        Chem.MolFromMolBlock(mol_or_molblock_a, removeHs=False)
        if isinstance(mol_or_molblock_a, str)
        else mol_or_molblock_a
    )
    mol_b: Chem.Mol | None = (
        Chem.MolFromMolBlock(mol_or_molblock_b, removeHs=False)
        if isinstance(mol_or_molblock_b, str)
        else mol_or_molblock_b
    )

    if not mol_a or not mol_b:
        raise ValueError("Invalid molblocks")

    if Chem.MolToSmiles(mol_a) != Chem.MolToSmiles(mol_b):
        raise ValueError("Non-identical molecules")

    # Basic topology. Unreachable in practice: identical canonical SMILES already
    # implies identical atom/bond counts, but this stays as a defensive guard.
    if (
        mol_a.GetNumAtoms() != mol_b.GetNumAtoms() or mol_a.GetNumBonds() != mol_b.GetNumBonds()
    ):  # pragma: no cover
        if not quiet:
            logger.error("Non-identical number of atoms or bonds")
        return False

    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()

    pos_a = np.array(conf_a.GetPositions())
    pos_b = np.array(conf_b.GetPositions())

    # Cost matrix of distances
    cost_matrix = cdist(pos_a, pos_b)

    # Apply Scoring Function:
    # Add a massive penalty for any atom pair that is chemically incompatible.
    # This prevents linear_sum_assignment from pairing a Carbon with an Oxygen
    # just because they are close in space.
    penalty = 1e7
    for i, atom_a in enumerate(get_atoms(mol_a)):
        for j, atom_b in enumerate(get_atoms(mol_b)):
            if not are_atoms_equal(atom_a, atom_b):
                cost_matrix[i, j] += penalty

    # Find minimum-cost one-to-one assignment
    rows, cols = linear_sum_assignment(cost_matrix)

    mapping = {int(c): int(r) for c, r in zip(cols, rows, strict=True)}  # expected -> actual

    max_dist = max(cost_matrix[rows, cols])
    if max_dist >= atol:
        if not quiet:
            logger.error("Positions off by %f", max_dist)
        return False

    # Unreachable in practice: the 1e7 mismatch penalty above means the optimal
    # assignment always prefers a type-preserving pairing when one exists, and
    # identical canonical SMILES guarantees one does. Kept as a defensive guard.
    for exp_idx, act_idx in mapping.items():
        atom_a = mol_a.GetAtomWithIdx(act_idx)
        atom_b = mol_b.GetAtomWithIdx(exp_idx)

        if not are_atoms_equal(atom_a, atom_b):  # pragma: no cover
            if not quiet:
                logger.error("Non-identical atom data")
            return False

    for bond in get_bonds(mol_b):
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]

        # Unreachable in practice for the same reason: identical canonical SMILES
        # guarantees a graph-isomorphic mapping exists between mol_a and mol_b.
        other: Chem.Bond | None = mol_a.GetBondBetweenAtoms(a1, a2)
        if not other:  # pragma: no cover
            if not quiet:
                logger.error("No bond found")
            return False

        if not are_bonds_equal(bond, other):
            if not quiet:
                logger.error("Non-identical bond data")
            return False

    return True


def get_mol_center(mol_or_conf: Chem.Mol | Chem.Conformer) -> NDArray[np.float64]:
    """Get the center of the atom."""
    from rdkit import Chem

    conf = mol_or_conf.GetConformer() if isinstance(mol_or_conf, Chem.Mol) else mol_or_conf
    positions = conf.GetPositions()
    return (positions.min(axis=0) + positions.max(axis=0)) / 2.0  # type: ignore[no-any-return]


def move_molecule(mol_or_conf: Chem.Mol | Chem.Conformer, offset: NDArray[np.float64]) -> None:
    """Translate every atom position of a molecule by `offset` (shape (3,))."""
    from rdkit import Chem

    if offset.shape != (3,):  # pragma: no cover
        raise ValueError("Offset has wrong shape")

    conf = mol_or_conf.GetConformer() if isinstance(mol_or_conf, Chem.Mol) else mol_or_conf
    conf.SetPositions(conf.GetPositions() + offset)


def recenter_mol(
    mol_or_conf: Chem.Mol | Chem.Conformer,
    new_center: NDArray[np.float64],
    check_for_shift: NDArray[np.float64] | None,
    atol: float,
) -> None:
    """Recenter a molecule to new center coordinates."""
    import numpy as np

    if new_center.shape != (3,):  # pragma: no cover
        raise ValueError("New center has wrong shape")

    if check_for_shift is not None:
        if check_for_shift.shape != (3,):  # pragma: no cover
            raise ValueError("Previous center has wrong shape")
        if not np.allclose(check_for_shift, new_center, atol=atol):
            logger.warning("The molecule was shifted")

    move_molecule(mol_or_conf, new_center - get_mol_center(mol_or_conf))


def flatten_conformer_z(mol: Chem.Mol, conf: Chem.Conformer, atol: float) -> None:
    """Force every atom's Z-coordinate in `conf` back to exactly 0.0.

    Logs a warning if an atom's Z deviates from 0 by more than `atol`.
    """
    from rdkit.Geometry import Point3D

    for ix in range(mol.GetNumAtoms()):
        pos = get_atom_position(conf, ix)
        if abs(pos.z) > atol:  # pragma: no cover
            logger.warning("Operation resulted in a non-zero z-coordinate, resetting...")
        conf.SetAtomPosition(ix, Point3D(pos.x, pos.y, 0.0))


def raise_if_3d_molecule(conf: Chem.Conformer, atol: float) -> None:
    """Raise a ValueError if the conformer has any non-zero z coordinates."""
    import numpy as np

    if not np.isclose(conf.GetPositions()[:, 2], 0.0, atol=atol).all():
        raise ValueError("Molecule is a 3D molecule")


def are_atoms_equal(atom_a: Chem.Atom, atom_b: Chem.Atom) -> bool:
    """Whether to atoms have the same atomic number, aromaticity and charge."""
    return (
        atom_a.GetAtomicNum() == atom_b.GetAtomicNum()
        and atom_a.GetIsAromatic() == atom_b.GetIsAromatic()
        and atom_a.GetFormalCharge() == atom_b.GetFormalCharge()
    )


def are_bonds_equal(bond_a: Chem.Bond, bond_b: Chem.Bond) -> bool:
    """Whether to bonds have the same bond type and aromaticity."""
    return (
        bond_a.GetBondType() == bond_b.GetBondType()
        and bond_a.GetIsAromatic() == bond_b.GetIsAromatic()
    )


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
