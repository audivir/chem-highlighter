"""Modify a molecule, namely rotate, mirror, or flip bonds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit.Chem.rdMolTransforms import TransformConformer

from chem_highlighter.state import AtomState
from chem_highlighter.utils import Position3D

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from rdkit import Chem


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

    matrix = np.array(
        [
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    rotate_around_x = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    rotate_around_y = np.array(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    if flip_horizontal:
        matrix = matrix @ rotate_around_y

    if flip_vertical:
        matrix = matrix @ rotate_around_x

    return matrix


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
