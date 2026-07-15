"""Tests for chem_highlighter.align."""

from __future__ import annotations

import math
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom

if TYPE_CHECKING:
    from numpy.typing import NDArray

from chem_highlighter.align import (
    find_mcs,
    flip_bonds,
    get_2d_mol,
    get_2d_rotation_angle,
    get_alignment_flips_and_transform,
    get_alignment_ops_from_molblock,
)
from chem_highlighter.utils import mol_from_smiles

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def ethylbenzene() -> tuple[Chem.Mol, Chem.Mol]:
    """Two ethylbenzene 2D mols with the ethyl chain in different orientations."""
    q = get_2d_mol(_FIXTURES.joinpath("ethylbenzene1.mol").read_text())
    r = get_2d_mol(_FIXTURES.joinpath("ethylbenzene2.mol").read_text())
    return q, r


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


def test_flip_bonds_detects_flipped_ethyl_chain(ethylbenzene: tuple[Chem.Mol, Chem.Mol]) -> None:
    q, r = ethylbenzene
    flips = flip_bonds(q, r, find_mcs(q, r))
    assert flips == [(6, 4)]


@pytest.mark.parametrize(
    "theta",
    [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, -math.pi / 4, -math.pi / 2],
)
def test_get_2d_rotation_angle(theta: float) -> None:
    # Returns -arctan2(sin θ, cos θ) = -θ.
    result = get_2d_rotation_angle(_rotation_matrix_4x4(theta))
    assert math.isclose(result, -theta, abs_tol=1e-10)


def test_get_2d_rotation_angle_raises_when_not_4x4() -> None:
    with pytest.raises(ValueError, match="4x4"):
        get_2d_rotation_angle(np.eye(3))


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


def test_get_alignment_flips_and_transform(ethylbenzene: tuple[Chem.Mol, Chem.Mol]) -> None:
    q, r = ethylbenzene
    flips, transform = get_alignment_flips_and_transform(q, r)
    assert math.isclose(get_2d_rotation_angle(transform), math.pi, abs_tol=1e-3)
    assert flips == [(6, 4)]


@pytest.mark.parametrize(
    ("smiles_q", "smiles_r", "expected"),
    [
        ("CCO", "CCO", 0.0),
        ("c1ccccc1", "Cc1ccccc1", 0.0),
        ("CCCO", "CCCCO", math.pi),
    ],
)
def test_get_alignment_ops_from_molblock_noops(
    smiles_q: str, smiles_r: str, expected: float
) -> None:
    flips, angle = get_alignment_ops_from_molblock(_mol(smiles_q)[1], _mol(smiles_r)[1])
    assert flips == []
    assert math.isclose(np.abs(angle), expected, abs_tol=1e-2)


def test_get_alignment_ops_from_molblock_ethyl(ethylbenzene: tuple[Chem.Mol, Chem.Mol]) -> None:
    q, r = ethylbenzene
    flips, angle = get_alignment_ops_from_molblock(Chem.MolToMolBlock(q), Chem.MolToMolBlock(r))
    assert flips == [(6, 4)]
    assert math.isclose(angle, math.pi, abs_tol=1e-3)
