from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import from_fixture_molblock

from chem_highlighter.modify import apply_transform, flip_bond, make_transform
from chem_highlighter.utils import is_same_conformer

if TYPE_CHECKING:
    from rdkit import Chem


def visualize_conformers(a: Chem.Mol, b: Chem.Mol) -> None:
    import tempfile
    import webbrowser
    from pathlib import Path

    from chem_highlighter import RDKitDocument

    svg_a = RDKitDocument.from_mol(a).to_svg()
    svg_b = RDKitDocument.from_mol(b).to_svg()

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Conformer Comparison</title>

        <link
            rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"
        >

        <style>
            body {{
                max-width: 1100px;
                margin: 2rem auto;
                padding: 0 1rem;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 1.5rem;
            }}

            .mol {{
                padding: 1.5rem;
                border-radius: 1rem;
                background: var(--card-background-color);
                box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                overflow: hidden;
            }}

            .mol h2 {{
                text-align: center;
                margin-bottom: 1rem;
            }}

            .mol svg {{
                width: 100%;
                height: auto;
                display: block;
            }}

            /* RDKit uses black lines/text; invert in dark mode */
            @media (prefers-color-scheme: dark) {{
                .mol svg {{
                    filter: invert(1);
                }}
            }}
        </style>
    </head>

    <body>
        <main>
            <h1>Conformer Comparison</h1>

            <div class="grid">
                <article class="mol">
                    <h2>Conformer A</h2>
                    {svg_a}
                </article>

                <article class="mol">
                    <h2>Conformer B</h2>
                    {svg_b}
                </article>
            </div>
        </main>
    </body>
    </html>
    """

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        encoding="utf-8",
        delete=False,
    ) as f:
        f.write(html)
        path = Path(f.name)

    webbrowser.open(path.as_uri())

    breakpoint()

    path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "flip_x, flip_y, expected_matrix",
    [
        # flip_x=True, flip_y=True (Stacks Z-axis then X-axis 180-deg rotations)
        (
            True,
            True,
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # flip_y=True (180-deg around Z-axis)
        (
            False,
            True,
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # flip_x=True (180-deg around X-axis)
        (
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
        # No flips (Identity)
        (
            False,
            False,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
    ids=["flip_x_and_y", "flip_y_only", "flip_x_only", "no_flips"],
)
def test_make_transform_flips_only(flip_x, flip_y, expected_matrix):
    """Test standard axis inversions without any Z-axis rotations."""
    transform = make_transform(angle_deg=0.0, flip_x=flip_x, flip_y=flip_y)
    np.testing.assert_allclose(transform, expected_matrix, atol=1e-7)


@pytest.mark.parametrize(
    "angle_deg, flip_x, flip_y, expected_matrix",
    [
        # 90 deg, flip_x=True, flip_y=True
        (
            90.0,
            True,
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
        # 90 deg, flip_y=True
        (
            90.0,
            False,
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
        # 90 deg, flip_x=True
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
        # 180 deg, flip_x=True, flip_y=True
        (
            180.0,
            True,
            True,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # 180 deg, flip_x=True
        (
            180.0,
            True,
            False,
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
    ids=[
        "90deg_flip_x_and_y",
        "90deg_flip_y_only",
        "90deg_flip_x_only",
        "90deg_no_flips",
        "180deg_flip_x_and_y",
        "180deg_flip_x_only",
    ],
)
def test_make_transform_with_angles(angle_deg, flip_x, flip_y, expected_matrix):
    """Test axis inversions stacked with Z-axis angular rotations."""
    result = make_transform(angle_deg=angle_deg, flip_x=flip_x, flip_y=flip_y)
    np.testing.assert_allclose(result, expected_matrix, atol=1e-7)


@pytest.mark.parametrize(
    "angle_deg, flip_x, flip_y, expected_matrix",
    [
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
        # 45 deg, flip_x=True, flip_y=True
        (
            45.0,
            True,
            True,
            np.array(
                [
                    [-0.70710678, -0.70710678, 0.0, 0.0],
                    [-0.70710678, 0.70710678, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # Negative angle: -30 deg, flip_x=True
        # Base cos(330) = ~0.866, sin(330) = -0.5
        (
            -30.0,
            True,
            False,
            np.array(
                [
                    [0.86602540, -0.5, 0.0, 0.0],
                    [-0.5, -0.86602540, 0.0, 0.0],
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
        "45deg_no_flips",
        "45deg_flip_x_and_y",
        "negative_30deg_flip_x",
        "fractional_12_5deg",
    ],
)
def test_make_transform_weirder_angles(angle_deg, flip_x, flip_y, expected_matrix):
    """Test modulo math, negative angles, and floating point precision."""
    result = make_transform(angle_deg=angle_deg, flip_x=flip_x, flip_y=flip_y)
    np.testing.assert_allclose(result, expected_matrix, atol=1e-7)


@pytest.mark.parametrize(
    ("q_file", "r_file", "angle_deg", "flip_x", "flip_y"),
    [
        ("3-methylbutanone.mol", "3-methylbutanone_44cw.mol", 44, False, False),
        ("3-methylbutanone_44cw.mol", "3-methylbutanone.mol", -44, False, False),
        ("3-methylbutanone.mol", "3-methylbutanone_hf.mol", 0, False, True),
        ("3-methylbutanone_hf.mol", "3-methylbutanone.mol", 0, False, True),
        ("3-methylbutanone.mol", "3-methylbutanone_vf.mol", 0, True, False),
        ("3-methylbutanone_vf.mol", "3-methylbutanone.mol", 0, True, False),
        ("acetylic_acid.mol", "acetylic_acid_44cw.mol", 44, False, False),
        ("acetylic_acid_44cw.mol", "acetylic_acid.mol", -44, False, False),
        ("acetylic_acid_hf.mol", "acetylic_acid.mol", 0, False, True),
        ("acetylic_acid.mol", "acetylic_acid_hf.mol", 0, False, True),
        ("acetylic_acid.mol", "acetylic_acid_vf.mol", 0, True, False),
        ("acetylic_acid_vf.mol", "acetylic_acid.mol", 0, True, False),
    ],
)
def test_apply_transform(
    q_file: str, r_file: str, angle_deg: float, flip_x: bool, flip_y: bool
) -> None:
    q_orig = from_fixture_molblock(q_file)
    r = from_fixture_molblock(r_file)
    assert not is_same_conformer(q_orig, r)  # , visualize_conformers(q_orig, r)

    q = apply_transform(q_orig, angle_deg, flip_x=flip_x, flip_y=flip_y)
    assert is_same_conformer(q, r)  # , visualize_conformers(q, r)

    q = apply_transform(q, -angle_deg, flip_x=flip_x, flip_y=flip_y)
    assert is_same_conformer(q, q_orig)  # , visualize_conformers(q, q_orig)


@pytest.mark.parametrize(
    ("q_file", "r_file", "bond_ix", "anchor_atom_ix"),
    [
        ("acetylic_acid.mol", "acetylic_acid_b00.mol", 0, 0),
        ("acetylic_acid_b00.mol", "acetylic_acid.mol", 0, 0),
        ("3-methylbutanone.mol", "3-methylbutanone_b12.mol", 1, 2),
        ("3-methylbutanone_b12.mol", "3-methylbutanone.mol", 1, 2),
    ],
)
def test_flip_bond_single(q_file: str, r_file: str, bond_ix: int, anchor_atom_ix: int) -> None:

    q_orig = from_fixture_molblock(q_file)
    r = from_fixture_molblock(r_file)
    assert not is_same_conformer(q_orig, r)

    q = flip_bond(q_orig, bond_ix, anchor_atom_ix)
    assert is_same_conformer(q, r)

    q = flip_bond(q, bond_ix, anchor_atom_ix)
    assert is_same_conformer(q, q_orig)
