"""Tests for chem_highlighter.align."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pytest
from conftest import FIXTURES
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom

from chem_highlighter.align import (
    Flips,
    find_mcs,
    flip_bonds,
    get_2d_global_flip_and_angle,
    get_2d_mol,
    get_alignment_flips_and_transform,
    get_alignment_ops_from_molblock,
    get_atom_position,
)
from chem_highlighter.utils import mol_from_smiles

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Query(NamedTuple):
    """A query including the expected values."""

    mol: Chem.Mol
    expected_flips: Flips
    expected_global_flip: bool
    expected_angle: float


@pytest.fixture
def queries() -> tuple[Query, Query, Chem.Mol]:
    """Two ethylbenzene 2D mols with the ethyl chain in different orientations."""
    q1 = Query(get_2d_mol((FIXTURES / "query1.mol").read_text()), [(6, 4), (7, 6)], True, 0.0)
    q2 = Query(get_2d_mol((FIXTURES / "query2.mol").read_text()), [], True, 180.0)
    r = get_2d_mol((FIXTURES / "ref.mol").read_text())

    return q1, q2, r


def _mol(smiles: str) -> tuple[Chem.Mol, str]:
    """Return a RDKit molecule and a V2000 mol block with 2D coordinates for *smiles*."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    return mol, Chem.MolToMolBlock(mol)


def _3d_mol(smiles: str) -> tuple[Chem.Mol, str]:
    """Like _mol but with 3D coordinates (EmbedMolecule)."""
    mol = Chem.MolFromSmiles(smiles)
    rdDistGeom.EmbedMolecule(mol)
    return mol, Chem.MolToMolBlock(mol)


def _rotation_matrix_4x4(angle_rad: float) -> NDArray[np.float64]:
    """Build a pure 2D rotation by *angle_rad* embedded in a 4x4 homogeneous matrix.

    Standard form:
        [[cos th, -sin th, 0, 0],
         [sin th,  cos th, 0, 0],
         [0,       0,      1, 0],
         [0,       0,      0, 1]]
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def test_get_atom_position() -> None:
    mol, _ = _mol("CCC")
    conf = mol.GetConformer()
    pos = get_atom_position(conf, 0)
    assert pos.z == 0


def test_get_2d_mol() -> None:
    mol, molblock = _mol("CCC")
    assert get_2d_mol(molblock).GetNumAtoms() == 3  # from mol block string
    assert get_2d_mol(mol).GetNumAtoms() == 3  # from Chem.Mol directly


def test_get_2d_mol_raises_when_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid molblock"):
        get_2d_mol("this is not a molblock")
    with pytest.raises(ValueError, match="No coordinates available for molecule"):
        get_2d_mol(mol_from_smiles("CCC"))  # SMILES-parsed mol has no conformer
    with pytest.raises(ValueError, match="Molecule is a 3D molecule"):
        get_2d_mol(_3d_mol("C1CCCCC1")[1])  # cyclohexane chair has non-zero Z


@pytest.mark.parametrize(
    ("smiles_q", "smiles_r", "n_mcs"),
    [
        ("CCO", "CCCO", 3),  # ethanol in propanol: 3 atoms shared
        ("c1ccccc1", "Cc1ccccc1", 6),  # benzene ring in toluene: 6 atoms
        ("CCCO", "CCCCO", 4),  # propanol in butanol: 4 atoms
    ],
)
def test_find_mcs(smiles_q: str, smiles_r: str, n_mcs: int) -> None:
    q, r = _mol(smiles_q)[0], _mol(smiles_r)[0]
    mapping = find_mcs(q, r)
    assert len(mapping) == n_mcs
    for q_idx, r_idx in mapping.items():
        assert q.GetAtomWithIdx(q_idx).GetAtomicNum() == r.GetAtomWithIdx(r_idx).GetAtomicNum()
        assert 0 <= q_idx < q.GetNumAtoms()
        assert 0 <= r_idx < r.GetNumAtoms()


@pytest.mark.parametrize(
    ("smiles_q", "smiles_r"),
    [
        ("CCO", "CCO"),  # all terminal bonds, none are rotatable
        ("c1ccccc1", "c1ccccc1"),  # ring bonds excluded by the rotatable-bond SMARTS
        ("c1ccccc1", "CCc1ccccc1"),  # MCS = ring only; rotatable ring→chain bond skipped
    ],
)
def test_flip_bonds(smiles_q: str, smiles_r: str) -> None:
    q, r = _mol(smiles_q)[0], _mol(smiles_r)[0]
    assert flip_bonds(q, r, find_mcs(q, r)) == []


@pytest.mark.parametrize("query_ix", [1, 2])
def test_flip_bonds_detects_flipped_ethyl_chain(
    query_ix: Literal[1, 2],
    queries: tuple[Query, Query, Chem.Mol],
) -> None:
    q1, q2, ref = queries
    query = q1 if query_ix == 1 else q2
    flips = flip_bonds(query.mol, ref, find_mcs(query.mol, ref))
    assert flips == query.expected_flips


@pytest.mark.parametrize(
    "theta",
    [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, -math.pi / 4, -math.pi / 2],
)
def test_get_2d_global_flip_and_angle(theta: float) -> None:
    # Returns -arctan2(sin θ, cos θ) = -θ.
    global_flip, result = get_2d_global_flip_and_angle(_rotation_matrix_4x4(theta))
    assert not global_flip
    assert math.isclose(result, -theta * 180 / math.pi % 360.0, abs_tol=1e-10)


def test_get_2d_global_flip_and_angle_raises_when_not_4x4() -> None:
    with pytest.raises(ValueError, match="4x4"):
        get_2d_global_flip_and_angle(np.eye(3))


@pytest.mark.parametrize(
    ("smiles_q", "smiles_r"),
    [
        ("CCO", "CCO"),
        ("c1ccccc1", "Cc1ccccc1"),
        ("CCCO", "CCCCO"),
    ],
)
def test_get_alignment_flips_and_transform_noops(smiles_q: str, smiles_r: str) -> None:
    q, r = _mol(smiles_q)[0], _mol(smiles_r)[0]
    flips, transform = get_alignment_flips_and_transform(q, r)
    transform[0:2, 3] = 0.0
    np.testing.assert_allclose(np.abs(transform), np.eye(4), atol=1e-2)
    assert flips == []


@pytest.mark.parametrize("query_ix", [1, 2])
def test_get_alignment_flips_and_transform(
    query_ix: Literal[1, 2],
    queries: tuple[Query, Query, Chem.Mol],
) -> None:
    q1, q2, ref = queries
    query = q1 if query_ix == 1 else q2
    flips, transform = get_alignment_flips_and_transform(query.mol, ref)
    global_flip, angle = get_2d_global_flip_and_angle(transform)
    assert flips == query.expected_flips
    assert global_flip == query.expected_global_flip
    assert math.isclose(angle, query.expected_angle, abs_tol=1e-2)


@pytest.mark.parametrize(
    ("smiles_q", "smiles_r", "expected_global_flip", "expected_angle"),
    [
        ("CCO", "CCO", False, 0.0),
        ("c1ccccc1", "Cc1ccccc1", False, 0.0),
        ("CCCO", "CCCCO", True, 180.0),
    ],
)
def test_get_alignment_ops_from_molblock_noops(
    smiles_q: str, smiles_r: str, expected_global_flip: bool, expected_angle: float
) -> None:
    flips, global_flip, angle = get_alignment_ops_from_molblock(
        _mol(smiles_q)[1], _mol(smiles_r)[1]
    )
    assert flips == []
    assert global_flip == expected_global_flip
    assert math.isclose(np.abs(angle), expected_angle, abs_tol=1e-1)


@pytest.mark.parametrize("query_ix", [1, 2])
def test_get_alignment_ops_from_molblock_ethyl(
    query_ix: Literal[1, 2],
    queries: tuple[Query, Chem.Mol, Query],
) -> None:
    q1, q2, ref = queries
    query = q1 if query_ix == 1 else q2
    flips, global_flip, angle = get_alignment_ops_from_molblock(
        Chem.MolToMolBlock(query.mol), Chem.MolToMolBlock(ref)
    )
    assert flips == query.expected_flips
    assert global_flip == query.expected_global_flip
    assert math.isclose(angle, query.expected_angle, abs_tol=1e-2)
