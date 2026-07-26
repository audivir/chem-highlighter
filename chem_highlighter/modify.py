"""Modify a molecule, namely rotate, mirror, or flip bonds."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit.Chem.rdMolTransforms import TransformConformer

from chem_highlighter.state import AtomState
from chem_highlighter.utils import Position3D

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from rdkit import Chem

logger = logging.getLogger(__name__)


def make_transform(
    angle_deg: float = 0.0,
    *,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> NDArray[np.float64]:
    """Create a 2D transform matrix.

    Args:
        angle_deg: Counterclockwise rotation angle in degrees.
        flip_horizontal: Mirror about the vertical axis.
        flip_vertical: Mirror about the horizontal axis.
    """
    import numpy as np

    theta = np.radians(angle_deg % 360.0)
    c = np.cos(theta)
    s = np.sin(theta)

    matrix = np.diag([0.0, 0.0, 1.0, 1.0])
    matrix[:2, :2] = [[c, s], [-s, c]]

    rotate_around_x = np.diag([1.0, -1.0, -1.0, 1.0])
    rotate_around_y = np.diag([-1.0, 1.0, -1.0, 1.0])

    if flip_horizontal:
        matrix = matrix @ rotate_around_y

    if flip_vertical:
        matrix = matrix @ rotate_around_x

    return matrix


def parse_transform(matrix: NDArray[np.float64], tol: float = 1e-5) -> tuple[bool, float]:
    """Extract the reflection state and 2D rotation angle from a 4x4 matrix.

    Arguments:
        matrix: 4x4 transformation matrix.
        tol: Tolerance to match our constraints.

    Returns:
        A tuple whether a horizontal flip is necessary and the rotation angle in degrees.

    Raises:
        ValueError if the matrix is not 4x4 or does not represent a valid rigid transformation.
    """
    import numpy as np

    if matrix.shape != (4, 4):  # pragma: no cover
        raise ValueError("Matrix must be exactly 4x4.")

    m_2d = matrix[:2, :2]

    # Check that there is no coupling between XY and Z.
    m_3d = matrix[:3, :3]
    if not np.allclose(m_3d[:2, 2], 0, atol=tol) or not np.allclose(
        m_3d[2, :2], 0, atol=tol
    ):  # pragma: no cover
        logger.warning("Matrix contains out-of-plane rotation components.")

    det = np.linalg.det(m_2d)

    horizontal_flip = False

    # Remove a reflection, if present, before extracting the angle.
    # We standardize all 2D reflections to a horizontal flip (negating the X column).
    if det < 0:
        horizontal_flip = True
        # Undo the horizontal flip by multiplying by a matrix that negates the first column
        m_2d = m_2d @ np.diag([-1.0, 1.0])

    if not np.isclose(np.linalg.det(m_2d), 1.0, atol=tol):
        raise ValueError("Matrix 2D component does not represent a valid rigid transformation.")

    # Based on make_transform structure [[c, s], [-s, c]]
    # c = m_2d[0, 0] and s = m_2d[0, 1]
    angle_rad = np.arctan2(m_2d[0, 1], m_2d[0, 0])

    return horizontal_flip, float(np.degrees(angle_rad) % 360.0)


def apply_transform(
    mol: Chem.Mol, angle_deg: float, *, flip_horizontal: bool = False, flip_vertical: bool = False
) -> Chem.Mol:
    """Rotate and/or mirror a (flat, z=0) conformer within its own plane.

    Args:
        mol: Molecule to transform.
        angle_deg: Counterclockwise rotation angle in degrees.
        flip_horizontal: Mirror about the vertical axis.
        flip_vertical: Mirror about the horizontal axis.
    """
    import numpy as np
    from rdkit import Chem

    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    matrix = make_transform(angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
    TransformConformer(conf, matrix)

    positions = np.array(conf.GetPositions())
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    new_positions = [Position3D(*p) for p in positions - center]
    for i, new_pos in enumerate(new_positions):
        conf.SetAtomPosition(i, new_pos)

    return mol


def flip_bond(
    mol: Chem.Mol,
    bond_ix: int,
    anchor_atom_ix: int,
) -> Chem.Mol:
    """Flip a rotatable bond by 180°.

    Args:
        mol: Molecule whose conformer will be modified in-place.
        bond_ix: Bond index to rotate.
        anchor_atom_ix: Atom on the bond that should remain fixed.

    Returns:
        Mol with flipped bond.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    bond = mol.GetBondWithIdx(bond_ix)

    if bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Only single bonds can be flipped")

    if bond.GetIsAromatic():
        raise ValueError("Cannot rotate aromatic bonds")

    begin = bond.GetBeginAtomIdx()
    end = bond.GetEndAtomIdx()

    if anchor_atom_ix == begin:
        idx_b = begin
        idx_c = end
    elif anchor_atom_ix == end:
        idx_b = end
        idx_c = begin
    else:
        raise ValueError("anchor_atom_ix is not part of the specified bond")

    tag = AtomState.freeze(mol)

    mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=[idx_b, idx_c])

    atom_b = mol.GetAtomWithIdx(idx_b)
    atom_c = mol.GetAtomWithIdx(idx_c)

    neighbors_b = [n.GetIdx() for n in atom_b.GetNeighbors() if n.GetIdx() != idx_c]
    neighbors_c = [n.GetIdx() for n in atom_c.GetNeighbors() if n.GetIdx() != idx_b]

    if not neighbors_b:
        raise ValueError("Anchor atom has no suitable neighboring atom")
    if not neighbors_c:
        raise ValueError("Rotating atom has no suitable neighboring atom")

    idx_a = neighbors_b[0]
    idx_d = neighbors_c[0]

    conf = mol.GetConformer()

    angle = rdMolTransforms.GetDihedralDeg(conf, idx_a, idx_b, idx_c, idx_d)

    rdMolTransforms.SetDihedralDeg(
        conf,
        idx_a,
        idx_b,
        idx_c,
        idx_d,
        angle + 180.0,
    )

    return AtomState.unfreeze(mol, tag)
