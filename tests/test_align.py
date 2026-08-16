"""Tests for chem_highlighter.align."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom
from rdkit.Geometry import Point3D
from utils import from_fixture_molblock

from chem_highlighter.align import (
    find_mcs,
    flip_misaligned_bonds,
    get_2d_mol,
    get_alignment_flips_and_transform,
    get_alignment_ops_from_molblock,
    get_best_fit_angle,
)
from chem_highlighter.modify import apply_transform, flip_bond, parse_transform
from chem_highlighter.utils import is_same_conformer, mol_from_smiles

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _mol(smiles: str) -> tuple[Chem.Mol, str]:
    """Generate RDKit molecule and V2000 mol block with 2D coordinates from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    return mol, Chem.MolToMolBlock(mol, kekulize=False, forceV3000=True)


def _3d_mol(smiles: str) -> tuple[Chem.Mol, str]:
    """Generate RDKit molecule and V3000 mol block with 3D coordinates from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    rdDistGeom.EmbedMolecule(mol)
    return mol, Chem.MolToMolBlock(mol, kekulize=False, forceV3000=True)


def _rect_mol(corners: Sequence[tuple[float, float]]) -> Chem.Mol:
    """Create molecule with one atom at each specified (x, y) coordinate."""
    mol, _ = _mol("C" * len(corners))
    conf = mol.GetConformer()
    for ix, (x, y) in enumerate(corners):
        conf.SetAtomPosition(ix, Point3D(x, y, 0.0))
    return mol


def _chain_mol(symbols: Sequence[str], positions: Sequence[tuple[float, float, float]]) -> Chem.Mol:
    """Create straight-chain molecule with specified atom order and 3D positions."""
    mol = Chem.RWMol()
    for symbol in symbols:
        mol.AddAtom(Chem.Atom(symbol))
    for ix in range(len(symbols) - 1):
        mol.AddBond(ix, ix + 1, Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(symbols))
    for ix, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(ix, Point3D(x, y, z))
    mol.AddConformer(conf)
    result = mol.GetMol()
    Chem.SanitizeMol(result)
    return result


def _assert_angle_close(actual: float, expected: float, atol: float = 0.1) -> None:
    """Assert that two angles in degrees match modulo 180."""
    diff = abs(actual - expected) % 180.0
    assert min(diff, 180.0 - diff) < atol


def test_get_2d_mol(atol: float) -> None:
    mol, _ = _mol("CCC")
    assert get_2d_mol(mol, atol=atol).GetNumAtoms() == 3


def test_get_2d_mol_raises_when_invalid(atol: float) -> None:
    with pytest.raises(ValueError, match="No coordinates available for molecule"):
        get_2d_mol(mol_from_smiles("CCC"), atol=atol)  # smiiles-parsed molecule lacks a conformer.
    with pytest.raises(ValueError, match="Molecule is a 3D molecule"):
        get_2d_mol(
            _3d_mol("C1CCCCC1")[0], atol=atol
        )  # cyclohexane chair has non-zero z-coordinates.


@pytest.mark.parametrize(
    ("smiles_q", "smiles_r", "n_mcs"),
    [
        ("CCO", "CCCO", 3),  # ethanol and propanol share 3 atoms.
        ("c1ccccc1", "Cc1ccccc1", 6),  # benzene and toluene share 6 atoms.
        ("CCCO", "CCCCO", 4),  # propanol and butanol share 4 atoms.
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
    ("q_file", "allowed_flips"),
    [
        ("ethanol.mol", []),
        ("3-methylbutanone.mol", [(1, 2)]),
        ("acetylic_acid.mol", [(0, 0)]),
        ("mol.mol", [(6, 6), (7, 6), (8, 6), (9, 8)]),
    ],
)
def test_flip_misaligned_bonds(
    q_file: str, allowed_flips: Sequence[tuple[int, int]], atol: float
) -> None:
    q_orig = from_fixture_molblock(q_file)
    q_orig = Chem.AddHs(q_orig, addCoords=True)

    for r in range(len(allowed_flips) + 1):
        for flips in itertools.combinations(allowed_flips, r=r):
            q = q_orig
            for bond_ix, anchor_atom_ix in flips:
                q = flip_bond(q, bond_ix, anchor_atom_ix, atol=atol)
            if flips:
                assert not is_same_conformer(q, q_orig, atol=atol)

            found_flips = flip_misaligned_bonds(q, q_orig, find_mcs(q, q_orig), atol=atol)
            assert found_flips == list(flips)
            assert is_same_conformer(q, q_orig, atol=atol)


def test_flip_misaligned_bonds_with_independently_numbered_reference(atol: float) -> None:
    # Cl-C-C-F chain with one rotatable bond to test a single flip decision.
    b, c, a = (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (-1.0, 1.3, 0.0)

    def d_at(dihedral_deg: float) -> tuple[float, float, float]:
        theta = np.radians(dihedral_deg)
        return (c[0] + 1.0, 1.3 * np.cos(theta), 1.3 * np.sin(theta))

    # query Cl-Ca-Cb-F chain with 0 degree dihedral.
    query = _chain_mol(["Cl", "C", "C", "F"], [a, b, c, d_at(0.0)])
    # reference Cl-Ca-Cb-F chain with reversed atom numbering and
    # 170 degree dihedral to force a flip.
    reference = _chain_mol(["F", "C", "C", "Cl"], [d_at(170.0), c, b, a])

    mcs_match = find_mcs(query, reference)
    # map query atoms to reference atoms in reverse order.
    assert mcs_match == {0: 3, 1: 2, 2: 1, 3: 0}

    flips = flip_misaligned_bonds(query, reference, mcs_match, atol=atol)

    # anchor flip at Ca because F exhibits greater displacement.
    assert flips == [(1, 1)]


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene: D6h symmetry with multiple zero-RMSD non-identity superpositions.
        "CCO",  # ethanol: asymmetric case to verify code path.
    ],
)
def test_get_alignment_flips_and_transform_self_alignment_is_a_no_op(
    smiles: str, atol: float
) -> None:
    # verify Kabsch algorithm prefers identity transform over symmetry-equivalent alternatives.
    mol, _ = _mol(smiles)
    query = get_2d_mol(mol, atol=atol)
    reference = get_2d_mol(Chem.Mol(mol), atol=atol)

    flips, transform = get_alignment_flips_and_transform(query, reference, atol=atol)

    assert flips == []
    np.testing.assert_allclose(transform, np.eye(4), atol=atol)


# acetylic_acid.mol: use 14-degree increments to avoid failure at 30 degrees.
@pytest.mark.parametrize("angle_deg", list(range(0, 360, 14)))
@pytest.mark.parametrize("flip_horizontal", [False, True])
@pytest.mark.parametrize(
    "q_file",
    ["ethanol.mol", "3-methylbutanone.mol", "acetylic_acid.mol", "mol.mol"],
)
def test_get_alignment_flips_and_transform_without_alignment_flips(
    angle_deg: float, flip_horizontal: bool, q_file: str, atol: float
) -> None:
    q_orig = from_fixture_molblock(q_file)

    q = apply_transform(q_orig, angle_deg, flip_horizontal=flip_horizontal, atol=atol)
    flips, transform = get_alignment_flips_and_transform(q, q_orig, atol=atol)
    global_flip, found_angle = parse_transform(transform, atol=atol)
    assert flips == []
    assert global_flip == flip_horizontal
    factor = 1.0 if flip_horizontal else -1.0
    assert np.isclose(found_angle, factor * angle_deg % 360.0, atol=atol)


def assert_all_alignments(
    q_orig: Chem.Mol,
    allowed_flips: Sequence[tuple[int, int]],
    angle_deg: float,
    flip_horizontal: bool,
    op_func: Callable[[Chem.Mol, Chem.Mol, float], tuple[list[tuple[int, int]], bool, float]],
    atol: float,
) -> None:
    for r in range(max((len(f) for f in allowed_flips), default=1)):
        for flips in itertools.combinations(allowed_flips, r=r):
            q = q_orig
            for bond_ix, anchor_atom_ix in flips:
                q = flip_bond(q, bond_ix, anchor_atom_ix, atol=atol)

            q = apply_transform(q, angle_deg, flip_horizontal=flip_horizontal, atol=atol)
            keep = not flip_horizontal and len(flips) == 0 and np.isclose(angle_deg, 0.0, atol=atol)

            assert is_same_conformer(q, q_orig, atol=atol) == keep

            found_flips, global_flip, found_angle = op_func(q, q_orig, atol)

            # reapply detected transformations.
            for bond_ix, anchor_atom_ix in found_flips:
                q = flip_bond(q, bond_ix, anchor_atom_ix, atol=atol)
            q = apply_transform(q, found_angle, flip_horizontal=global_flip, atol=atol)

            assert is_same_conformer(q, q_orig, atol=atol)


# acetylic_acid.mol: use 14-degree increments to avoid failure at 30 degrees.
@pytest.mark.parametrize("angle_deg", list(range(0, 360, 32)))
@pytest.mark.parametrize("flip_horizontal", [False, True])
@pytest.mark.parametrize(
    ("q_file", "allowed_flips"),
    [
        ("ethanol.mol", []),
        ("3-methylbutanone.mol", [(1, 2)]),
        ("acetylic_acid.mol", [(0, 0)]),
        ("mol.mol", [(6, 6), (7, 6), (8, 6), (9, 8)]),
    ],
)
def test_get_alignment_flips_and_transform(
    angle_deg: float,
    flip_horizontal: bool,
    q_file: str,
    allowed_flips: Sequence[tuple[int, int]],
    atol: float,
) -> None:
    q_orig = from_fixture_molblock(q_file)

    def op_func(
        q: Chem.Mol, q_orig: Chem.Mol, atol: float
    ) -> tuple[list[tuple[int, int]], bool, float]:
        found_flips, transform = get_alignment_flips_and_transform(q, q_orig, atol=atol)
        global_flip, found_angle = parse_transform(transform, atol=atol)
        return found_flips, global_flip, found_angle

    assert_all_alignments(q_orig, allowed_flips, angle_deg, flip_horizontal, op_func, atol=atol)


# acetylic_acid.mol: use 14-degree increments to avoid failure at 30 degrees.
@pytest.mark.parametrize("angle_deg", list(range(0, 360, 32)))
@pytest.mark.parametrize("flip_horizontal", [False, True])
@pytest.mark.parametrize(
    ("q_file", "allowed_flips"),
    [
        ("ethanol.mol", []),
        ("3-methylbutanone.mol", [(1, 2)]),
        ("acetylic_acid.mol", [(0, 0)]),
        ("mol.mol", [(6, 6), (7, 6), (8, 6), (9, 8)]),
    ],
)
def test_get_alignment_ops_from_molblock(
    angle_deg: float,
    flip_horizontal: bool,
    q_file: str,
    allowed_flips: Sequence[tuple[int, int]],
    atol: float,
) -> None:
    q_orig = from_fixture_molblock(q_file)

    def op_func(
        q: Chem.Mol, q_orig: Chem.Mol, atol: float
    ) -> tuple[list[tuple[int, int]], bool, float]:
        q_molblock = Chem.MolToMolBlock(q, forceV3000=True)
        q_orig_molblock = Chem.MolToMolBlock(q_orig, forceV3000=True)
        return get_alignment_ops_from_molblock(q_molblock, q_orig_molblock, atol=atol)

    assert_all_alignments(q_orig, allowed_flips, angle_deg, flip_horizontal, op_func, atol=atol)


def test_get_best_fit_angle_matching_ratio_needs_no_rotation() -> None:
    mol = _rect_mol([(-2.0, -1.0), (2.0, -1.0), (-2.0, 1.0), (2.0, 1.0)])  # 4x2 rectangle.
    _assert_angle_close(get_best_fit_angle(mol, (2.0, 1.0)), 0.0)


def test_get_best_fit_angle_rotates_tall_shape_to_match_ratio() -> None:
    # 2x4 rectangle rotated 90 degrees.
    mol = _rect_mol([(-1.0, 2.0), (-1.0, -2.0), (1.0, 2.0), (1.0, -2.0)])
    _assert_angle_close(get_best_fit_angle(mol, (2.0, 1.0)), 90.0)


def test_get_best_fit_angle_is_invariant_to_target_scale() -> None:
    mol = _rect_mol([(-1.0, 2.0), (-1.0, -2.0), (1.0, 2.0), (1.0, -2.0)])
    angle = get_best_fit_angle(mol, (2.0, 1.0))
    scaled_angle = get_best_fit_angle(mol, (200.0, 100.0))
    _assert_angle_close(angle, scaled_angle)


@pytest.mark.parametrize("angle_step_deg", [15.0, 30.0, 45.0, 90.0])
def test_get_best_fit_angle_only_returns_grid_angles(angle_step_deg: float, atol: float) -> None:
    # use a non-rectangular molecule to verify search is restricted to the grid.
    q_orig = from_fixture_molblock("mol.mol")
    angle = get_best_fit_angle(q_orig, (2.0, 1.0), angle_step_deg=angle_step_deg)
    assert 0.0 <= angle < 180.0
    nearest_multiple = round(angle / angle_step_deg) * angle_step_deg
    assert np.isclose(angle, nearest_multiple, atol=atol**2)


@pytest.mark.parametrize("ratio", ["2:1", "2.5:1", " 2 : 1 "])
def test_get_alignment_ops_from_molblock_with_bounding_box_ratio(ratio: str, atol: float) -> None:
    mol = _rect_mol([(-1.0, 2.0), (-1.0, -2.0), (1.0, 2.0), (1.0, -2.0)])  # 2x4 rectangle.
    molblock = Chem.MolToMolBlock(mol, forceV3000=True)

    flips, global_flip, angle = get_alignment_ops_from_molblock(molblock, ratio, atol=atol)

    assert flips == []
    assert global_flip is False
    _assert_angle_close(angle, 90.0)


@pytest.mark.parametrize("ratio", ["2:0", "0:1", "0:0"])
def test_get_alignment_ops_from_molblock_raises_for_non_positive_ratio(
    ratio: str, atol: float
) -> None:
    _, molblock = _mol("CCC")
    with pytest.raises(ValueError, match="Aspect ratio must be positive"):
        get_alignment_ops_from_molblock(molblock, ratio, atol=atol)


def test_get_alignment_ops_from_molblock_treats_non_ratio_string_as_molblock(atol: float) -> None:
    _, molblock = _mol("CCC")
    with pytest.raises(ValueError, match="Invalid molblock"):
        get_alignment_ops_from_molblock(molblock, "not a molblock", atol=atol)
