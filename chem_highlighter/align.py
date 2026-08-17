"""Align two RDKit molecules using rotatable bond flips and rotation."""

from __future__ import annotations

import functools
import logging
import re
from typing import TYPE_CHECKING, TypeAlias

from chem_highlighter.modify import parse_transform
from chem_highlighter.utils import (
    flatten_conformer_z,
    get_atom_position,
    get_mol_center,
    get_neighbors,
    raise_if_3d_molecule,
    recenter_mol,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray
    from rdkit import Chem

logger = logging.getLogger(__name__)

Flip: TypeAlias = tuple[int, int]
Flips: TypeAlias = list[Flip]

RATIO_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")

# default angle increment for bonds in 2D depictions.
BOND_ANGLE_STEP_DEG = 30.0


def get_2d_mol(mol: Chem.Mol, atol: float) -> Chem.Mol:
    """Validate that molecule has usable 2D conformer.

    Raises:
        ValueError: If molecule lacks coordinates or is 3D.
    """
    try:
        conf = mol.GetConformer()
    except ValueError as e:
        raise ValueError("No coordinates available for molecule") from e

    raise_if_3d_molecule(conf, atol=atol)
    return mol


def _parse_molblock(molblock: str) -> Chem.Mol:
    """Parse mol block into RDKit molecule.

    Raises:
        ValueError: If mol block is invalid.
    """
    from rdkit import Chem

    mol: Chem.Mol | None = Chem.MolFromMolBlock(molblock, removeHs=False)
    if not mol:
        raise ValueError("Invalid molblock")
    return mol


def find_mcs(query: Chem.Mol, reference: Chem.Mol) -> dict[int, int]:
    """Find maximum common substructure between two molecules.

    Returns:
        Mapping of atom indices from query molecule to reference molecule.

    Raises:
        ValueError: If no common substructure is found.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    mcs = rdFMCS.FindMCS(
        [query, reference], matchValences=True, completeRingsOnly=True, ringMatchesRingOnly=True
    )
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

    query_match = query.GetSubstructMatch(mcs_mol)
    reference_match = reference.GetSubstructMatch(mcs_mol)

    if not query_match or not reference_match:  # pragma: no cover
        raise ValueError("No common substructure found")

    return dict(zip(query_match, reference_match, strict=True))


@functools.lru_cache(maxsize=1)
def _rotatable_bond_query() -> Chem.Mol:
    """Return rotatable-bond SMARTS query."""
    from rdkit import Chem

    return Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def flip_misaligned_bonds(
    query: Chem.Mol, reference: Chem.Mol, mcs_match: Mapping[int, int], atol: float
) -> list[tuple[int, int]]:
    """Flip misaligned rotatable bonds based on MCS match.

    Returns:
        List of tuples containing bond index and anchor atom index for each flip.
    """
    from rdkit.Chem import rdMolTransforms

    prev_center = get_mol_center(query)

    flips: list[tuple[int, int]] = []
    # find rotatable bonds.
    rot_bonds_ref = reference.GetSubstructMatches(_rotatable_bond_query())

    conf_q = query.GetConformer()
    conf_r = reference.GetConformer()

    mcs_match_rev = {v: k for k, v in mcs_match.items()}

    for bond_atoms in rot_bonds_ref:
        idx_b, idx_c = bond_atoms

        # ensure both atoms are in MCS match.
        if idx_b not in mcs_match_rev or idx_c not in mcs_match_rev:  # pragma: no cover
            continue

        atom_b = reference.GetAtomWithIdx(idx_b)
        atom_c = reference.GetAtomWithIdx(idx_c)

        # find neighbors in MCS match.
        neighbors_b = [
            n.GetIdx()
            for n in get_neighbors(atom_b)
            if n.GetIdx() != idx_c and n.GetIdx() in mcs_match_rev
        ]
        neighbors_c = [
            n.GetIdx()
            for n in get_neighbors(atom_c)
            if n.GetIdx() != idx_b and n.GetIdx() in mcs_match_rev
        ]

        # ensure both neighbors are in MCS match.
        if not neighbors_b or not neighbors_c:  # pragma: no cover
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
            # measure displacement relative to bond atoms to avoid global translation bias.
            vec_a_q = get_atom_position(conf_q, q_idx_a) - get_atom_position(conf_q, q_idx_b)
            vec_a_r = get_atom_position(conf_r, idx_a) - get_atom_position(conf_r, idx_b)
            dist_a = vec_a_q.Distance(vec_a_r)

            vec_d_q = get_atom_position(conf_q, q_idx_d) - get_atom_position(conf_q, q_idx_c)
            vec_d_r = get_atom_position(conf_r, idx_d) - get_atom_position(conf_r, idx_c)
            dist_d = vec_d_q.Distance(vec_d_r)

            bond_ix = query.GetBondBetweenAtoms(q_idx_b, q_idx_c).GetIdx()

            if dist_a > dist_d:
                # keep c-d side stationary and anchor at c.
                rdMolTransforms.SetDihedralDeg(
                    conf_q, q_idx_d, q_idx_c, q_idx_b, q_idx_a, angle_q + 180.0
                )
                flips.append((bond_ix, q_idx_c))
            else:
                # keep a-b side stationary and anchor at b.
                rdMolTransforms.SetDihedralDeg(
                    conf_q, q_idx_a, q_idx_b, q_idx_c, q_idx_d, angle_q + 180.0
                )
                flips.append((bond_ix, q_idx_b))

    flatten_conformer_z(query, conf_q, atol=atol)
    recenter_mol(query, prev_center, get_mol_center(query), atol=atol)

    return flips


def get_alignment_flips_and_transform(
    query: Chem.Mol, reference: Chem.Mol, atol: float
) -> tuple[list[tuple[int, int]], NDArray[np.float64]]:
    """Compute bond flips and transformation matrix to align two molecules.

    Returns:
        Tuple of bond flips (bond index and anchor atom index) and 4x4 transformation matrix.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    from chem_highlighter.utils import is_same_conformer

    # short-circuit if molecules are already aligned to avoid symmetry-equivalent non-identity
    # transforms from Kabsch algorithm.
    if is_same_conformer(query, reference, atol=atol, quiet=True):
        return [], np.eye(4)

    bare_mcs_match = find_mcs(query, reference)

    heavy_bare_mcs_match = {
        q_ix: r_ix
        for q_ix, r_ix in bare_mcs_match.items()
        if reference.GetAtomWithIdx(r_ix).GetAtomicNum() != 1
    }

    bare_rmsd, bare_transform = rdMolAlign.GetAlignmentTransform(
        query, reference, atomMap=list(heavy_bare_mcs_match.items())
    )

    if np.isclose(bare_rmsd, 0.0, atol=atol):
        return [], bare_transform

    query = Chem.AddHs(query, addCoords=True)
    reference = Chem.AddHs(reference, addCoords=True)

    mcs_match = find_mcs(query, reference)

    flips = flip_misaligned_bonds(query, reference, mcs_match, atol=atol)

    heavy_mcs_match = {
        q_ix: r_ix
        for q_ix, r_ix in mcs_match.items()
        if reference.GetAtomWithIdx(r_ix).GetAtomicNum() != 1
    }

    _, transform = rdMolAlign.GetAlignmentTransform(
        query, reference, atomMap=list(heavy_mcs_match.items())
    )

    return flips, transform


def parse_possible_ratio(reference: str) -> tuple[float, float] | None:
    """Parse bounding box aspect ratio string (e.g., "2:1").

    Returns:
        Tuple of width and height, or None if reference is not a ratio.

    Raises:
        ValueError: If ratio contains non-positive values.
    """
    match = RATIO_PATTERN.fullmatch(reference.strip())
    if not match:
        return None

    width, height = float(match.group(1)), float(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"Aspect ratio must be positive: {reference!r}")
    return width, height


def get_best_fit_angle(
    mol: Chem.Mol, ratio: tuple[float, float], angle_step_deg: float = BOND_ANGLE_STEP_DEG
) -> float:
    """Compute rotation angle that best fits molecule 2D bounding box to target ratio."""
    import numpy as np

    positions = mol.GetConformer().GetPositions()[:, :2]
    target_w, target_h = ratio

    def fit_cost(angle_deg: float) -> float:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        rotated_x = positions[:, 0] * c + positions[:, 1] * s
        rotated_y = -positions[:, 0] * s + positions[:, 1] * c
        width = rotated_x.max() - rotated_x.min()
        height = rotated_y.max() - rotated_y.min()
        return float(max(width / target_w, height / target_h))

    candidates = np.arange(0.0, 180.0, angle_step_deg)
    costs = [fit_cost(float(angle)) for angle in candidates]
    return float(candidates[int(np.argmin(costs))])


def get_alignment_ops_from_molblock(
    query_molblock: str, reference: str, atol: float
) -> tuple[list[tuple[int, int]], bool, float]:
    """Compute flips and rotation angle required to align molecule to reference.

    Reference may be molblock or bounding box aspect ratio (e.g., "2:1").

    Returns:
        Tuple of bond flips (bond index and anchor atom index), boolean indicating
        global horizontal flip, and rotation angle in degrees.
    """
    query = get_2d_mol(_parse_molblock(query_molblock), atol=atol)

    ratio = parse_possible_ratio(reference)
    if ratio is not None:
        return [], False, get_best_fit_angle(query, ratio)

    reference_mol = get_2d_mol(_parse_molblock(reference), atol=atol)
    flips, transform = get_alignment_flips_and_transform(query, reference_mol, atol=atol)
    global_flip, angle = parse_transform(transform, atol=atol)
    return flips, global_flip, angle
