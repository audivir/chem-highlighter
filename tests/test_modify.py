"""Test the chem_highlighter.modify module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdmolops import Kekulize
from rdkit.Chem.rdMolTransforms import TransformConformer
from utils import from_fixture_molblock, visualize_conformers

from chem_highlighter.modify import apply_transform, flip_bond, make_transform, parse_transform
from chem_highlighter.utils import get_mol_center, is_same_conformer, move_molecule

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.parametrize(
    ("flip_horizontal", "flip_vertical", "expected_matrix"),
    [
        # flip_horizontal=True, flip_vertical=True
        (
            True,
            True,
            np.diag([-1.0, -1.0, 1.0, 1.0]),
        ),
        # flip_vertical=True (180-deg around Z-axis)
        (
            False,
            True,
            np.diag([1.0, -1.0, -1.0, 1.0]),
        ),
        # flip_horizontal=True (180-deg around X-axis)
        (
            True,
            False,
            np.diag([-1.0, 1.0, -1.0, 1.0]),
        ),
        # No flips (Identity)
        (
            False,
            False,
            np.diag([1.0, 1.0, 1.0, 1.0]),
        ),
    ],
    ids=["flip_horizontal_and_vertical", "flip_vertical_only", "flip_horizontal_only", "no_flips"],
)
def test_make_transform_flips_only(
    flip_horizontal: bool, flip_vertical: bool, expected_matrix: NDArray[np.float64], atol: float
) -> None:
    """Test standard axis inversions without any Z-axis rotations."""
    transform = make_transform(
        angle_deg=0.0, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )
    np.testing.assert_allclose(transform, expected_matrix, atol=atol**1.5)


@pytest.mark.parametrize(
    ("angle_deg", "flip_horizontal", "flip_vertical", "expected_matrix"),
    [
        # 90 deg, flip_horizontal=True, flip_vertical=True
        (
            90.0,
            True,
            True,
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 90 deg, flip_vertical=True
        (
            90.0,
            False,
            True,
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 90 deg, flip_horizontal=True
        (
            90.0,
            True,
            False,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 90 deg, no flips
        (
            90.0,
            False,
            False,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 180 deg, flip_horizontal=True, flip_vertical=True
        (
            180.0,
            True,
            True,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 180 deg, flip_horizontal=True
        (
            180.0,
            True,
            False,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 45 deg, no flips
        (
            45.0,
            False,
            False,
            np.array(
                [
                    [0.70710678, 0.70710678, 0.0, 0.0],
                    [-0.70710678, 0.70710678, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 45 deg, flip_horizontal=True
        (
            45.0,
            True,
            False,
            np.array(
                [
                    [-0.70710678, 0.70710678, 0.0, 0.0],
                    [0.70710678, 0.70710678, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 45 deg, flip_vertical=True
        (
            45.0,
            False,
            True,
            np.array(
                [
                    [0.70710678, -0.70710678, 0.0, 0.0],
                    [-0.70710678, -0.70710678, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 45 deg, flip_horizontal=True, flip_vertical=True
        (
            45.0,
            True,
            True,
            np.array(
                [
                    [-0.70710678, -0.70710678, 0.0, 0.0],
                    [0.70710678, -0.70710678, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # Negative angle: -30 deg, flip_horizontal=True
        (
            -30.0,
            True,
            False,
            np.array(
                [
                    [-0.86602540, -0.5, 0.0, 0.0],
                    [-0.5, 0.86602540, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # Fractional angle: 12.5 deg, no flips
        (
            12.5,
            False,
            False,
            np.array(
                [
                    [0.97629601, 0.21643961, 0.0, 0.0],
                    [-0.21643961, 0.97629601, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
    ids=[
        "90deg_flip_horizontal_and_y",
        "90deg_flip_vertical_only",
        "90deg_flip_horizontal_only",
        "90deg_no_flips",
        "180deg_flip_horizontal_and_y",
        "180deg_flip_horizontal_only",
        "45deg_no_flips",
        "45deg_flip_horizontal_only",
        "45deg_flip_vertical_only",
        "45deg_flip_horizontal_and_y",
        "neg30deg_flip_horizontal_only",
        "12_5deg_no_flips",
    ],
)
def test_make_transform_with_angles(
    angle_deg: float,
    flip_horizontal: bool,
    flip_vertical: bool,
    expected_matrix: NDArray[np.float64],
    atol: float,
) -> None:
    """Test axis inversions stacked with Z-axis angular rotations."""
    result = make_transform(
        angle_deg=angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )
    np.testing.assert_allclose(result, expected_matrix, atol=atol**1.5)


@pytest.mark.parametrize(
    ("matrix", "expected_horizontal_flip", "expected_angle_deg"),
    [
        # Original: 90 deg, flip_horizontal=True, flip_vertical=True
        # Equivalent to: 0 flips, 90 + 180 = 270 deg
        (
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0,
            270.0,
        ),
        # Original: 90 deg, flip_vertical=True
        # Equivalent to: 1 (Horizontal) flip, 90 + 180 = 270 deg
        (
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            1,
            270.0,
        ),
        # Original: 90 deg, flip_horizontal=True
        (
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            1,
            90.0,
        ),
        # Original: 90 deg, no flips
        (
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0,
            90.0,
        ),
        # Original: 180 deg, flip_horizontal=True, flip_vertical=True
        # Equivalent to: 0 flips, 180 + 180 = 0 deg
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0,
            0.0,
        ),
        # Original: 180 deg, flip_horizontal=True
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            1,
            180.0,
        ),
        # Original: 45 deg, no flips
        (
            np.array(
                [
                    [0.70710678, 0.70710678, 0.0, 0.0],
                    [-0.70710678, 0.70710678, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0,
            45.0,
        ),
        # Original: 45 deg, flip_horizontal=True
        (
            np.array(
                [
                    [-0.70710678, 0.70710678, 0.0, 0.0],
                    [0.70710678, 0.70710678, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            1,
            45.0,
        ),
        # Original: 45 deg, flip_vertical=True
        # Equivalent to: 1 (Horizontal) flip, 45 + 180 = 225 deg
        (
            np.array(
                [
                    [0.70710678, -0.70710678, 0.0, 0.0],
                    [-0.70710678, -0.70710678, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            1,
            225.0,
        ),
        # Original: 45 deg, flip_horizontal=True, flip_vertical=True
        # Equivalent to: 0 flips, 45 + 180 = 225 deg
        (
            np.array(
                [
                    [-0.70710678, -0.70710678, 0.0, 0.0],
                    [0.70710678, -0.70710678, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0,
            225.0,
        ),
        # Original: Negative angle: -30 deg, flip_horizontal=True
        # Equivalent to: 1 (Horizontal) flip, 330 deg
        (
            np.array(
                [
                    [-0.86602540, -0.5, 0.0, 0.0],
                    [-0.5, 0.86602540, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            1,
            330.0,
        ),
        # Original: Fractional angle: 12.5 deg, no flips
        (
            np.array(
                [
                    [0.97629601, 0.21643961, 0.0, 0.0],
                    [-0.21643961, 0.97629601, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0,
            12.5,
        ),
    ],
    ids=[
        "90deg_both_flips_resolves_to_270deg",
        "90deg_vflip_resolves_to_hflip_270deg",
        "90deg_hflip_resolves_to_hflip_90deg",
        "90deg_no_flips_resolves_to_90deg",
        "180deg_both_flips_resolves_to_0deg",
        "180deg_hflip_resolves_to_hflip_180deg",
        "45deg_no_flips_resolves_to_45deg",
        "45deg_hflip_resolves_to_hflip_45deg",
        "45deg_vflip_resolves_to_hflip_225deg",
        "45deg_both_flips_resolves_to_225deg",
        "neg30deg_hflip_resolves_to_hflip_330deg",
        "12_5deg_no_flips_resolves_to_12_5deg",
    ],
)
def test_parse_transform(
    matrix: NDArray[np.float64],
    expected_horizontal_flip: bool,
    expected_angle_deg: float,
    atol: float,
) -> None:
    """Test extracting angle and flips from transformation matrices."""
    horizontal_flip, angle_deg = parse_transform(matrix, atol=atol)

    assert horizontal_flip == expected_horizontal_flip
    np.testing.assert_allclose(angle_deg, expected_angle_deg, atol=atol)


@pytest.mark.parametrize(
    ("q_file", "r_file", "angle_deg", "flip_horizontal", "flip_vertical"),
    [
        ("3-methylbutanone.mol", "3-methylbutanone_44cw.mol", 44, False, False),
        ("3-methylbutanone_44cw.mol", "3-methylbutanone.mol", -44, False, False),
        ("3-methylbutanone.mol", "3-methylbutanone_bf.mol", 0, True, True),
        ("3-methylbutanone_bf.mol", "3-methylbutanone.mol", 0, True, True),
        ("3-methylbutanone.mol", "3-methylbutanone_hf.mol", 0, True, False),
        ("3-methylbutanone_hf.mol", "3-methylbutanone.mol", 0, True, False),
        ("3-methylbutanone.mol", "3-methylbutanone_vf.mol", 0, False, True),
        ("3-methylbutanone_vf.mol", "3-methylbutanone.mol", 0, False, True),
        ("acetylic_acid.mol", "acetylic_acid_44cw.mol", 44, False, False),
        ("acetylic_acid_44cw.mol", "acetylic_acid.mol", -44, False, False),
        ("acetylic_acid.mol", "acetylic_acid_bf.mol", 0, True, True),
        ("acetylic_acid_bf.mol", "acetylic_acid.mol", 0, True, True),
        ("acetylic_acid.mol", "acetylic_acid_hf.mol", 0, True, False),
        ("acetylic_acid_hf.mol", "acetylic_acid.mol", 0, True, False),
        ("acetylic_acid.mol", "acetylic_acid_vf.mol", 0, False, True),
        ("acetylic_acid_vf.mol", "acetylic_acid.mol", 0, False, True),
        ("ethanol.mol", "ethanol_44cw.mol", 44, False, False),
        ("ethanol_44cw.mol", "ethanol.mol", -44, False, False),
        ("ethanol.mol", "ethanol_bf.mol", 0, True, True),
        ("ethanol_bf.mol", "ethanol.mol", 0, True, True),
        ("ethanol.mol", "ethanol_hf.mol", 0, True, False),
        ("ethanol_hf.mol", "ethanol.mol", 0, True, False),
        ("ethanol.mol", "ethanol_vf.mol", 0, False, True),
        ("ethanol_vf.mol", "ethanol.mol", 0, False, True),
        ("mol.mol", "mol_44cw.mol", 44, False, False),
        ("mol_44cw.mol", "mol.mol", -44, False, False),
        ("mol.mol", "mol_bf.mol", 0, True, True),
        ("mol_bf.mol", "mol.mol", 0, True, True),
        ("mol.mol", "mol_hf.mol", 0, True, False),
        ("mol_hf.mol", "mol.mol", 0, True, False),
        ("mol.mol", "mol_vf.mol", 0, False, True),
        ("mol_vf.mol", "mol.mol", 0, False, True),
    ],
)
def test_apply_transform(
    q_file: str,
    r_file: str,
    angle_deg: float,
    flip_horizontal: bool,
    flip_vertical: bool,
    atol: float,
) -> None:
    q_orig = from_fixture_molblock(q_file)
    r = from_fixture_molblock(r_file)
    assert not is_same_conformer(q_orig, r, atol=atol)

    q = apply_transform(
        q_orig, angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, atol=atol
    )
    assert is_same_conformer(q, r, atol=atol)

    q = apply_transform(
        q, -angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, atol=atol
    )
    assert is_same_conformer(q, q_orig, atol=atol)


@pytest.mark.parametrize(
    ("recenter", "expected_center"),
    [("origin", np.zeros(3)), ("previous", np.array([10.0, -4.0, 0.0])), ("none", None)],
)
def test_apply_transform_recenter_modes(
    recenter: Literal["origin", "previous", "none"],
    expected_center: NDArray[np.float64] | None,
    atol: float,
) -> None:
    q_orig = from_fixture_molblock("ethanol.mol")

    move_molecule(q_orig, np.array([10.0, -4.0, 0.0]))

    q = apply_transform(q_orig, 30.0, recenter=recenter, atol=atol)

    matrix = make_transform(30.0, flip_vertical=False)
    TransformConformer(q_orig.GetConformer(), matrix)
    raw_center = get_mol_center(q_orig)

    if expected_center is None:
        expected_center = raw_center

    np.testing.assert_allclose(get_mol_center(q), expected_center, atol=atol)


@pytest.mark.parametrize(
    ("q_file", "r_file", "bond_ix", "anchor_atom_ix"),
    [
        ("acetylic_acid.mol", "acetylic_acid_b00.mol", 0, 0),
        ("acetylic_acid_b00.mol", "acetylic_acid.mol", 0, 0),
        ("3-methylbutanone.mol", "3-methylbutanone_b12.mol", 1, 2),
        ("3-methylbutanone_b12.mol", "3-methylbutanone.mol", 1, 2),
        ("mol.mol", "mol_b98.mol", 9, 8),
        ("mol.mol", "mol_b76.mol", 7, 6),
        ("mol.mol", "mol_b66.mol", 6, 6),
        ("mol.mol", "mol_b86.mol", 8, 6),
    ],
)
def test_flip_bond(
    q_file: str, r_file: str, bond_ix: int, anchor_atom_ix: int, atol: float
) -> None:
    q_orig = from_fixture_molblock(q_file)
    r = from_fixture_molblock(r_file)
    assert not is_same_conformer(q_orig, r, atol=atol)

    q = flip_bond(q_orig, bond_ix, anchor_atom_ix, atol=atol)

    assert is_same_conformer(q, r, atol=atol), visualize_conformers(q, q_orig)

    q = flip_bond(q, bond_ix, anchor_atom_ix, atol=atol)
    assert is_same_conformer(q, q_orig, atol=atol)


def test_flip_bond_errors(atol: float) -> None:

    benzenelike = Chem.MolFromSmiles("ClCc1ccccc1")
    Compute2DCoords(benzenelike)
    with pytest.raises(ValueError, match="Only single bonds can be flipped"):
        flip_bond(benzenelike, 2, 3, atol=atol)
    Kekulize(benzenelike)
    with pytest.raises(ValueError, match="Cannot rotate aromatic bonds"):
        flip_bond(benzenelike, 3, 3, atol=atol)
    with pytest.raises(ValueError, match="anchor_atom_ix is not part of the specified bond"):
        flip_bond(benzenelike, 0, 7, atol=atol)
    with pytest.raises(ValueError, match="Anchor atom has no suitable neighboring atom"):
        flip_bond(benzenelike, 0, 0, atol=atol)
    with pytest.raises(ValueError, match="Rotating atom has no suitable neighboring atom"):
        flip_bond(benzenelike, 0, 1, atol=atol)

    cyclohexane = Chem.MolFromSmiles("C1CCCCC1")
    Compute2DCoords(cyclohexane)
    with pytest.raises(ValueError, match="Cannot flip a bond that is part of a ring"):
        flip_bond(cyclohexane, 0, 1, atol=atol)


@pytest.mark.parametrize("angle_deg", list(range(0, 360, 15)))
@pytest.mark.parametrize(
    "q_file",
    ["ethanol.mol", "3-methylbutanone.mol", "acetylic_acid.mol"],
)
def test_rotate_and_flip(angle_deg: float, q_file: str, atol: float) -> None:
    q_orig = from_fixture_molblock(q_file)

    q = apply_transform(q_orig, angle_deg, atol=atol)
    keep_conf = np.isclose(angle_deg, 0.0)
    assert is_same_conformer(q, q_orig, atol=atol) == keep_conf

    q = apply_transform(q, -angle_deg, atol=atol)
    assert is_same_conformer(q, q_orig, atol=atol)

    # flip
    horizontal = apply_transform(q_orig, angle_deg, flip_horizontal=True, atol=atol)
    vertical = apply_transform(q_orig, angle_deg + 180.0, flip_vertical=True, atol=atol)
    assert is_same_conformer(horizontal, vertical, atol=atol)
