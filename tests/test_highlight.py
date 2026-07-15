"""Tests for chem_highlighter.decomposer.highlight."""

from __future__ import annotations

import pytest

from chem_highlighter.decomposer.core import Core
from chem_highlighter.decomposer.highlight import create_empty, draw_single, remove_empty_branches
from chem_highlighter.utils import mol_from_smiles


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C()C", "CC"),
        ("C(N)()", "C(N)"),
        ("CCC", "CCC"),
    ],
)
def test_remove_empty_branches(smiles: str, expected: str) -> None:
    assert remove_empty_branches(smiles) == expected


@pytest.mark.parametrize(
    ("core_smiles", "expected_atoms"),
    [
        ("C([*:1])([*:2])O", 4),  # two R-groups replaced by C → C(C)(C)O = 4 heavy atoms
        ("C[*:1]", 2),  # one R-group → C
    ],
)
def test_create_empty(core_smiles: str, expected_atoms: int) -> None:
    core = mol_from_smiles(core_smiles)
    empty = create_empty(core)
    assert empty.GetNumAtoms() == expected_atoms


@pytest.mark.parametrize(
    "smiles_list",
    [
        ["Cc1ccccc1", "CCc1ccccc1"],  # methyl and ethyl benzene → ring R-group coverage
        ["c1ccc(-c2ccccc2)cc1"],  # biphenyl → phenyl R-group with ring → get_ring_fill
    ],
)
def test_draw_single(smiles_list: list[str]) -> None:
    decomposed, _ = Core("c1ccccc1").decompose(smiles_list)
    doc = draw_single(decomposed[0])
    svg = doc.to_svg()
    assert "<svg" in svg
