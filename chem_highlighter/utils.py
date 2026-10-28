"""Utility functions for decomposer."""

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


class SmilesMolPair(NamedTuple):
    """Container for SMILES string and corresponding RDKit molecule."""

    smiles: str
    mol: Chem.Mol


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Create RDKit molecule from SMILES string."""
    from rdkit import Chem

    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    if not mol or mol.GetNumAtoms() < 1:
        raise ValueError("Invalid SMILES")
    return mol


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Create SMILES string from RDKit molecule."""
    from rdkit import Chem

    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:  # pragma: no cover
        raise ValueError("Conversion to SMILES failed") from e
    if not smiles:
        raise ValueError("Empty SMILES")
    return smiles


def get_smiles_mol_pair(data: str | Chem.Mol) -> SmilesMolPair:
    """Return SMILES string and corresponding RDKit molecule."""
    from rdkit import Chem

    if isinstance(data, Chem.Mol):
        return SmilesMolPair(mol_to_smiles(data), data)
    if isinstance(data, str):
        return SmilesMolPair(data, mol_from_smiles(data))
    raise TypeError("Invalid input")  # pragma: no cover


def get_atom_position(conf: Chem.Conformer, ix: int) -> Point3D:
    """Get 3D position of atom at index ix in conformer conf."""
    return conf.GetAtomPosition(ix)  # type: ignore[no-any-return]


def get_atoms(mol: Chem.Mol) -> tuple[Chem.Atom, ...]:
    """Return atoms from RDKit molecule."""
    return tuple(mol.GetAtoms())


def get_bonds(mol: Chem.Mol) -> tuple[Chem.Bond, ...]:
    """Return bonds from RDKit molecule."""
    return tuple(mol.GetBonds())


def get_neighbors(atom: Chem.Atom) -> tuple[Chem.Atom, ...]:
    """Return neighbor atoms from RDKit atom."""
    return atom.GetNeighbors()


def add_hydrogens(data: Sequence[str | Chem.Mol]) -> list[Chem.Mol]:
    """Add hydrogens to molecules."""
    from rdkit import Chem

    return [Chem.AddHs(get_smiles_mol_pair(d).mol, addCoords=True) for d in data]


def add_aromatic_hcounts(mol: Chem.Mol, mol_block: str) -> str:
    """Add `HCOUNT` to aromatic heteroatoms with hydrogens in V3000 molblock.

    Aromatic bonds do not tell ChemDraw which ring atom carries the hydrogen of `[nH]`, so it
    draws an invalid valence without hydrogen. Kekulized molblocks do not need this.
    """
    lines = mol_block.split("\n")
    in_atom_block = False
    atom_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("M  V30 BEGIN ATOM"):
            in_atom_block = True
        elif line.startswith("M  V30 END ATOM"):
            in_atom_block = False
        elif in_atom_block and line.startswith("M  V30 "):
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetIsAromatic() and atom.GetSymbol() != "C" and atom.GetTotalNumHs() > 0:
                lines[i] = f"{line} HCOUNT={atom.GetTotalNumHs()}"
            atom_idx += 1
    return "\n".join(lines)


def get_high_precision_v3000(mol: Chem.Mol, kekulize: bool = False) -> str:
    """Generate unrounded V3000 molblock."""
    from rdkit import Chem

    mol_block = Chem.MolToMolBlock(mol, kekulize=kekulize, forceV3000=True)
    if not kekulize:
        mol_block = add_aromatic_hcounts(mol, mol_block)

    if mol.GetNumConformers() == 0:  # pragma: no cover
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

            # v3000 atom line format: [idx] [element] [x] [y] [z] [aamap].
            parts = re.split(r"(\s+)", line)

            # replace rounded X, Y, Z coordinates with 8-decimal precision floats.
            parts[8] = f"{pos.x:.8f}"
            parts[10] = f"{pos.y:.8f}"
            parts[12] = f"{pos.z:.8f}"

            # reconstruct line.
            lines[i] = "".join(parts)
            atom_idx += 1

    return "\n".join(lines)


def is_same_layout(molblock_a: str, molblock_b: str, atol: float) -> bool:
    """Check if molblocks with the same atom order differ only by rotation and translation."""
    import numpy as np
    from rdkit import Chem

    mol_a: Chem.Mol | None = Chem.MolFromMolBlock(molblock_a, sanitize=False, removeHs=False)
    mol_b: Chem.Mol | None = Chem.MolFromMolBlock(molblock_b, sanitize=False, removeHs=False)
    if mol_a is None or mol_b is None:  # pragma: no cover
        raise ValueError("Invalid molblocks")
    if [a.GetAtomicNum() for a in mol_a.GetAtoms()] != [a.GetAtomicNum() for a in mol_b.GetAtoms()]:
        return False

    pos_a = np.array(mol_a.GetConformer().GetPositions())
    pos_b = np.array(mol_b.GetConformer().GetPositions())
    pos_a -= pos_a.mean(axis=0)
    pos_b -= pos_b.mean(axis=0)

    # kabsch rotation without reflection.
    u, _, vt = np.linalg.svd(pos_b.T @ pos_a)
    rotation = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return bool(np.abs(pos_b @ rotation - pos_a).max() < atol)


def is_same_conformer(  # noqa: C901,PLR0911,PLR0912
    mol_or_molblock_a: Chem.Mol | str,
    mol_or_molblock_b: Chem.Mol | str,
    atol: float,
    quiet: bool = False,
) -> bool:
    """Determine if two molecules represent same conformer."""
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
        if not quiet:
            logger.error("Non-identical molecules")
            raise ValueError("Non-identical molecules")
        return False

    # ensure topology consistency.
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

    # distance cost matrix.
    cost_matrix = cdist(pos_a, pos_b)

    # penalize chemically incompatible atom pairs to prevent incorrect spatial assignments.
    penalty = 1e7
    for i, atom_a in enumerate(get_atoms(mol_a)):
        for j, atom_b in enumerate(get_atoms(mol_b)):
            if not are_atoms_equal(atom_a, atom_b):
                cost_matrix[i, j] += penalty

    # find minimum-cost one-to-one assignment.
    rows, cols = linear_sum_assignment(cost_matrix)

    mapping = {
        int(c): int(r) for c, r in zip(cols, rows, strict=True)
    }  # compare expected and actual values.

    max_dist = max(cost_matrix[rows, cols])
    if max_dist >= atol:
        if not quiet:  # pragma: no branch
            logger.error("Positions off by %e", max_dist)
        return False

    # prevent unexpected assignment failures.
    for exp_idx, act_idx in mapping.items():
        atom_a = mol_a.GetAtomWithIdx(act_idx)
        atom_b = mol_b.GetAtomWithIdx(exp_idx)

        if not are_atoms_equal(atom_a, atom_b):  # pragma: no cover
            if not quiet:  # pragma: no branch
                logger.error("Non-identical atom data")
            return False

    for bond in get_bonds(mol_b):
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]

        # prevent missing isomorphic mappings.
        other: Chem.Bond | None = mol_a.GetBondBetweenAtoms(a1, a2)
        if not other:  # pragma: no cover
            if not quiet:  # pragma: no branch
                logger.error("No bond found")
            return False

        if not are_bonds_equal(bond, other):
            if not quiet:  # pragma: no branch
                logger.error("Non-identical bond data")
            return False

    return True


def get_mol_center(mol_or_conf: Chem.Mol | Chem.Conformer) -> NDArray[np.float64]:
    """Compute geometric center of molecule or conformer."""
    from rdkit import Chem

    conf = mol_or_conf.GetConformer() if isinstance(mol_or_conf, Chem.Mol) else mol_or_conf
    positions = conf.GetPositions()
    return (positions.min(axis=0) + positions.max(axis=0)) / 2.0  # type: ignore[no-any-return]


def move_molecule(mol_or_conf: Chem.Mol | Chem.Conformer, offset: NDArray[np.float64]) -> None:
    """Translate atom positions of molecule or conformer by offset."""
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
    """Recenter molecule or conformer to specified coordinates."""
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
    """Set atom Z-coordinates in conformer to 0.0."""
    from rdkit.Geometry import Point3D

    for ix in range(mol.GetNumAtoms()):
        pos = get_atom_position(conf, ix)
        if abs(pos.z) > atol:  # pragma: no cover
            logger.warning("Operation resulted in a non-zero z-coordinate, resetting...")
        conf.SetAtomPosition(ix, Point3D(pos.x, pos.y, 0.0))


def raise_if_3d_molecule(conf: Chem.Conformer, atol: float) -> None:
    """Raise ValueError if conformer contains non-zero Z-coordinates."""
    import numpy as np

    if not np.isclose(conf.GetPositions()[:, 2], 0.0, atol=atol).all():
        raise ValueError("Molecule is a 3D molecule")


def are_atoms_equal(atom_a: Chem.Atom, atom_b: Chem.Atom) -> bool:
    """Compare atomic number, aromaticity, and charge of two atoms."""
    return (
        atom_a.GetAtomicNum() == atom_b.GetAtomicNum()
        and atom_a.GetIsAromatic() == atom_b.GetIsAromatic()
        and atom_a.GetFormalCharge() == atom_b.GetFormalCharge()
    )


def are_bonds_equal(bond_a: Chem.Bond, bond_b: Chem.Bond) -> bool:
    """Compare bond type and aromaticity of two bonds."""
    return (
        bond_a.GetBondType() == bond_b.GetBondType()
        and bond_a.GetIsAromatic() == bond_b.GetIsAromatic()
    )


def setup_cmap() -> list[RGBA]:  # pragma: no cover
    """Initialize colormap."""
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
    """Convert palette hex color to ANSI escape sequence."""
    import matplotlib as mpl

    hex_color = palette[group_ix]
    r, g, b = [int(x * 255) for x in mpl.colors.hex2color(hex_color)]
    return f"\033[38;2;{r};{g};{b}m"


def get_float_env(name: str) -> float | None:
    """Retrieve float from environment variable."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_png_render_options() -> tuple[bool, float | None, float | None]:
    """Retrieve PNG rendering options from environment variables."""
    transparent = bool(os.environ.get("CHEM_HIGHLIGHTER_PNG_TRANSPARENT"))
    width = get_float_env("CHEM_HIGHLIGHTER_PNG_WIDTH")
    height = get_float_env("CHEM_HIGHLIGHTER_PNG_HEIGHT")
    if width is None:
        width = height
    if height is None:
        height = width
    return transparent, width, height
