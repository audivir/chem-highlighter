"""Align two RDKit molecules using flips around rotatable bonds and rotation."""

from __future__ import annotations

import logging
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


def get_2d_mol(mol_or_molblock: Chem.Mol | str, atol: float) -> Chem.Mol:
    """Return a RDKit molecule with a 2D conformer.

    Raises:
        ValueError: If any molecule is not convertable, has no coordinate or is a 3D molecule.
    """
    from rdkit import Chem

    if isinstance(mol_or_molblock, Chem.Mol):
        mol = mol_or_molblock
    else:
        maybe_mol: Chem.Mol | None = Chem.MolFromMolBlock(mol_or_molblock, removeHs=False)

        if not maybe_mol:
            raise ValueError("Invalid molblock")

        mol = maybe_mol

    try:
        conf = mol.GetConformer()
    except ValueError as e:
        raise ValueError("No coordinates available for molecule") from e

    raise_if_3d_molecule(conf, atol=atol)
    return mol


def find_mcs(query: Chem.Mol, reference: Chem.Mol) -> dict[int, int]:
    """Find the maximum common substructure between the two molecules.

    Returns:
        A mapping between the corresponding atom indices for query and reference, respectively.

    Raises:
        ValueError if no common substructure is found.
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


def flip_misaligned_bonds(
    query: Chem.Mol, reference: Chem.Mol, mcs_match: Mapping[int, int], atol: float
) -> list[tuple[int, int]]:
    """Flip misaligned rotatable bonds based on an MCS match.

    Returns:
        A list of the bond index and the anchor atom index where the flip happened.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    prev_center = get_mol_center(query)

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
        if idx_b not in mcs_match_rev or idx_c not in mcs_match_rev:  # pragma: no cover
            continue

        atom_b = reference.GetAtomWithIdx(idx_b)
        atom_c = reference.GetAtomWithIdx(idx_c)

        # find neighbors in mcs match
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

        # both neighbors must be in mcs match
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
            # Measure how far each side moved *relative to its own bond atom*
            # (a relative to b, d relative to c), not in absolute coordinates.
            # query may have been globally translated relative to reference
            # (e.g. by a prior flip_bond's bounding-box recentering), which
            # would otherwise bias an absolute-position comparison toward
            # picking the wrong side.
            vec_a_q = get_atom_position(conf_q, q_idx_a) - get_atom_position(conf_q, q_idx_b)
            vec_a_r = get_atom_position(conf_r, q_idx_a) - get_atom_position(conf_r, q_idx_b)
            dist_a = vec_a_q.Distance(vec_a_r)

            vec_d_q = get_atom_position(conf_q, q_idx_d) - get_atom_position(conf_q, q_idx_c)
            vec_d_r = get_atom_position(conf_r, q_idx_d) - get_atom_position(conf_r, q_idx_c)
            dist_d = vec_d_q.Distance(vec_d_r)

            bond_ix = query.GetBondBetweenAtoms(q_idx_b, q_idx_c).GetIdx()

            if dist_a > dist_d:
                # The 'a' side moved; keep the 'c-d' side stationary (anchor at c)
                rdMolTransforms.SetDihedralDeg(
                    conf_q, q_idx_d, q_idx_c, q_idx_b, q_idx_a, angle_q + 180.0
                )
                flips.append((bond_ix, q_idx_c))
            else:
                # The 'd' side moved; keep the 'a-b' side stationary (anchor at b)
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
    """Finds the necessary flips and tranformation matrix to align two molecules.

    Returns:
        A tuple with a list of necessary flips around rotatable bonds
        (each described by a bond index and an anchor atom index)
        and the 4x4 tranformation matrix to rotate the query molecule.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

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


def get_alignment_ops_from_molblock(
    query_molblock: str, reference_molblock: str, atol: float
) -> tuple[list[tuple[int, int]], bool, float]:
    """Finds the necessary flips and rotation angle to align two molecules as Mol blocks.

    Returns:
        A tuple with a list of necessary flips around rotatable bonds
        (each described by a bond index and an anchor atom index),
        whether a global horizontal flip is necessary
        and the rotation angle based on the reference molecule in degrees.
    """
    query = get_2d_mol(query_molblock, atol=atol)
    reference = get_2d_mol(reference_molblock, atol=atol)
    flips, transform = get_alignment_flips_and_transform(query, reference, atol=atol)
    global_flip, angle = parse_transform(transform, atol=atol)
    return flips, global_flip, angle
