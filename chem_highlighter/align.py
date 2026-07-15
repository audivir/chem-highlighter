"""Align two RDKit molecules using flips around rotatable bonds and rotation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray
    from rdkit import Chem

logger = logging.getLogger(__name__)

ALIGN_WEIGHT_DEFAULT = 1
ALIGN_WEIGHT_MAP = {1: 0.1, 6: 1}


def get_2d_mol(molblock_or_mol: str | Chem.Mol) -> Chem.Mol:
    """Return a RDKit molecule with a 2D conformer.

    Raises:
        ValueError: If any molecule is not convertable, has no coordinate or is a 3D molecule.
    """
    from rdkit import Chem

    if isinstance(molblock_or_mol, Chem.Mol):
        mol = molblock_or_mol
    else:
        maybe_mol: Chem.Mol | None = Chem.MolFromMolBlock(molblock_or_mol, removeHs=False)
        if not maybe_mol:
            raise ValueError("Invalid molblock")
        mol = maybe_mol
    try:
        conf = mol.GetConformer()
    except ValueError as e:
        raise ValueError("No coordinates available for molecule") from e
    if any(conf.GetAtomPosition(ix).z != 0.0 for ix in range(conf.GetNumAtoms())):
        raise ValueError("Molecule is a 3D molecule")
    return mol


def find_mcs(query: Chem.Mol, reference: Chem.Mol) -> dict[int, int]:
    """Find the maximum common substructure between the two molecules.

    Returns:
        A mapping between the corresponding atom indices for query and reference, respectively.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    mcs = rdFMCS.FindMCS([query, reference], completeRingsOnly=True, ringMatchesRingOnly=True)

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

    query_match = query.GetSubstructMatch(mcs_mol)
    reference_match = reference.GetSubstructMatch(mcs_mol)

    if not query_match or not reference_match:  # pragma: no cover
        raise ValueError("No common substructure found")

    return dict(zip(query_match, reference_match, strict=True))


def flip_bonds(
    query: Chem.Mol, reference: Chem.Mol, mcs_match: Mapping[int, int]
) -> list[tuple[int, int]]:
    """Flip misaligned rotatable bonds based on an MCS match.

    Returns:
        A list of the bond index and the anchor atom index where the flip happened.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    flips: list[tuple[int, int]] = []
    # find rotatable bonds
    rot_bond_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    rot_bonds_ref = reference.GetSubstructMatches(rot_bond_smarts)

    conf_q = query.GetConformer()
    conf_r = reference.GetConformer()

    mcs_match_rev = {v: k for k, v in mcs_match.items()}

    for bond_atoms in rot_bonds_ref:
        idx_b, idx_c = bond_atoms

        # both atoms must be in MCS match
        if idx_b not in mcs_match_rev or idx_c not in mcs_match_rev:
            continue

        atom_b = reference.GetAtomWithIdx(idx_b)
        atom_c = reference.GetAtomWithIdx(idx_c)

        # find neighbors in mcs match
        neighbors_b = [
            n.GetIdx()
            for n in atom_b.GetNeighbors()
            if n.GetIdx() != idx_c and n.GetIdx() in mcs_match_rev
        ]
        neighbors_c = [
            n.GetIdx()
            for n in atom_c.GetNeighbors()
            if n.GetIdx() != idx_b and n.GetIdx() in mcs_match_rev
        ]

        # both neighbors must be in mcs match
        if not neighbors_b or not neighbors_c:
            continue

        idx_a = neighbors_b[0]
        idx_d = neighbors_c[0]

        q_idx_a = mcs_match_rev[idx_a]
        q_idx_b = mcs_match_rev[idx_b]
        q_idx_c = mcs_match_rev[idx_c]
        q_idx_d = mcs_match_rev[idx_d]

        angle_r = rdMolTransforms.GetDihedralDeg(conf_r, idx_a, idx_b, idx_c, idx_d)
        angle_q = rdMolTransforms.GetDihedralDeg(conf_q, q_idx_a, q_idx_b, q_idx_c, q_idx_d)

        diff = abs(angle_r - angle_q)
        if diff > 180.0:  # noqa: PLR2004  # pragma: no cover
            diff = 360.0 - diff

        if diff > 90.0:  # noqa: PLR2004
            rdMolTransforms.SetDihedralDeg(
                conf_q, q_idx_a, q_idx_b, q_idx_c, q_idx_d, angle_q + 180.0
            )
            bond_ix = query.GetBondBetweenAtoms(q_idx_b, q_idx_c).GetIdx()
            flips.append((bond_ix, q_idx_b))

    # safety: force all Z-coordinates back to 0.0
    for ix in range(query.GetNumAtoms()):
        pos = conf_q.GetAtomPosition(ix)
        conf_q.SetAtomPosition(ix, (pos.x, pos.y, 0.0))

    return flips


def get_alignment_flips_and_transform(
    query: Chem.Mol, reference: Chem.Mol
) -> tuple[list[tuple[int, int]], NDArray[np.float64]]:
    """Finds the necessary flips and tranformation matrix to align two molecules.

    Returns:
        A tuple with a list of necessary flips around rotatable bonds
        (each described by a bond index and an anchor atom index)
        and the 4x4 tranformation matrix to rotate the query molecule.
    """
    from rdkit.Chem import rdMolAlign

    mcs_match = find_mcs(query, reference)

    flips = flip_bonds(query, reference, mcs_match)

    weights: list[float] = []
    for atom_ix in mcs_match.values():
        atom = reference.GetAtomWithIdx(atom_ix)
        weights.append(ALIGN_WEIGHT_MAP.get(atom.GetAtomicNum(), ALIGN_WEIGHT_DEFAULT))

    _, transform = rdMolAlign.GetAlignmentTransform(
        query, reference, atomMap=list(mcs_match.items()), weights=weights
    )

    return flips, transform


def get_2d_rotation_angle(matrix: NDArray[np.float64], tol: float = 1e-5) -> float:
    """Extract the 2D rotation angle from a 4x4 matrix.

    Arguments:
        matrix: 4x4 transformation matrix.
        tol: Tolerance to match our constraints.

    Returns:
        rotation angle in radians

    Raises:
        ValueError if the matrix contains 3D rotations, shearing, or scaling.
    """
    import numpy as np

    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be exactly 4x4.")

    zero_indices = [
        (0, 2),
        (1, 2),  # Z does not affect X or Y
        (2, 0),
        (2, 1),  # X or Y do not affect Z
        (3, 0),
        (3, 1),
        (3, 2),  # Bottom row must be [0, 0, 0, 1]
    ]

    fixed_matrix = [matrix[row, col] for row, col in zero_indices] + [
        matrix[2, 2],
        matrix[3, 3],
    ]
    target_matrix = [0.0] * len(zero_indices) + [1.0, 1.0]

    det = (matrix[0, 0] * matrix[1, 1]) - (matrix[0, 1] * matrix[1, 0])
    fixed_matrix += [matrix[0, 0], matrix[1, 0], det]
    target_matrix += [matrix[1, 1], -matrix[0, 1], 1.0]

    if not np.allclose(fixed_matrix, target_matrix, atol=tol):
        logger.warning(
            "Matrix contains scaling, shearing, non-uniform scaling, 3D rotations,"
            " Z-axis scaling or invalid homogeneous scaling."
        )

    return float(-np.arctan2(matrix[1, 0], matrix[0, 0]))


def get_alignment_ops_from_molblock(
    query_mol: str, reference_mol: str
) -> tuple[list[tuple[int, int]], float]:
    """Finds the necessary flips and rotation angle to align two molecules as Mol blocks.

    Returns:
        A tuple with a list of necessary flips around rotatable bonds
        (each described by a bond index and an anchor atom index)
        and the rotation angle based on the reference molecule in radians.
    """
    query = get_2d_mol(query_mol)
    reference = get_2d_mol(reference_mol)
    flips, transform = get_alignment_flips_and_transform(query, reference)
    angle = get_2d_rotation_angle(transform)
    return flips, angle
