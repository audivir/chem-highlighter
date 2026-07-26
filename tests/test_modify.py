"""Test the chem_highlighter.modify module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import from_fixture_molblock
from rdkit import Chem
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdmolops import Kekulize

from chem_highlighter.modify import apply_transform, flip_bond, make_transform, parse_transform
from chem_highlighter.utils import is_same_conformer

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
    flip_horizontal: bool, flip_vertical: bool, expected_matrix: NDArray[np.float64]
) -> None:
    """Test standard axis inversions without any Z-axis rotations."""
    transform = make_transform(
        angle_deg=0.0, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )
    np.testing.assert_allclose(transform, expected_matrix, atol=1e-7)


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
) -> None:
    """Test axis inversions stacked with Z-axis angular rotations."""
    result = make_transform(
        angle_deg=angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )
    np.testing.assert_allclose(result, expected_matrix, atol=1e-7)


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
    matrix: NDArray[np.float64], expected_horizontal_flip: bool, expected_angle_deg: float
) -> None:
    """Test extracting angle and flips from transformation matrices."""
    horizontal_flip, angle_deg = parse_transform(matrix)

    assert horizontal_flip == expected_horizontal_flip
    np.testing.assert_allclose(angle_deg, expected_angle_deg, atol=1e-4)


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
        ("acetylic_acid_hf.mol", "acetylic_acid.mol", 0, True, False),
        ("acetylic_acid.mol", "acetylic_acid_hf.mol", 0, True, False),
        ("acetylic_acid.mol", "acetylic_acid_vf.mol", 0, False, True),
        ("acetylic_acid_vf.mol", "acetylic_acid.mol", 0, False, True),
    ],
)
def test_apply_transform(
    q_file: str, r_file: str, angle_deg: float, flip_horizontal: bool, flip_vertical: bool
) -> None:
    q_orig = from_fixture_molblock(q_file)
    r = from_fixture_molblock(r_file)
    assert not is_same_conformer(q_orig, r)

    q = apply_transform(
        q_orig, angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )
    assert is_same_conformer(q, r)

    q = apply_transform(q, -angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
    assert is_same_conformer(q, q_orig)


@pytest.mark.parametrize(
    ("q_file", "r_file", "bond_ix", "anchor_atom_ix"),
    [
        ("acetylic_acid.mol", "acetylic_acid_b00.mol", 0, 0),
        ("acetylic_acid_b00.mol", "acetylic_acid.mol", 0, 0),
        ("3-methylbutanone.mol", "3-methylbutanone_b12.mol", 1, 2),
        ("3-methylbutanone_b12.mol", "3-methylbutanone.mol", 1, 2),
    ],
)
def test_flip_bond(q_file: str, r_file: str, bond_ix: int, anchor_atom_ix: int) -> None:

    q_orig = from_fixture_molblock(q_file)
    r = from_fixture_molblock(r_file)
    assert not is_same_conformer(q_orig, r)

    q = flip_bond(q_orig, bond_ix, anchor_atom_ix)
    assert is_same_conformer(q, r)

    q = flip_bond(q, bond_ix, anchor_atom_ix)
    assert is_same_conformer(q, q_orig)


def test_flip_bond_errors() -> None:
    benzenelike = Chem.MolFromSmiles("ClCc1ccccc1")
    Compute2DCoords(benzenelike)
    with pytest.raises(ValueError, match="Only single bonds can be flipped"):
        flip_bond(benzenelike, 2, 3)
    Kekulize(benzenelike)
    with pytest.raises(ValueError, match="Cannot rotate aromatic bonds"):
        flip_bond(benzenelike, 3, 3)
    with pytest.raises(ValueError, match="anchor_atom_ix is not part of the specified bond"):
        flip_bond(benzenelike, 0, 7)
    with pytest.raises(ValueError, match="Anchor atom has no suitable neighboring atom"):
        flip_bond(benzenelike, 0, 0)
    with pytest.raises(ValueError, match="Rotating atom has no suitable neighboring atom"):
        flip_bond(benzenelike, 0, 1)
