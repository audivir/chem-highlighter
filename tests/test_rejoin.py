"""Tests for chem_highlighter.decomposer.rejoin."""

from __future__ import annotations

from typing import Literal

import pytest
from conftest import assert_mols_equal
from rdkit import Chem

from chem_highlighter.decomposer import rejoin
from chem_highlighter.decomposer.rejoin import (
    clear_special_isotopes,
    get_mapped_atoms,
    get_sole_neighbor,
    join_multiple,
    mask_hydrogens,
)
from chem_highlighter.utils import mol_from_smiles


@pytest.mark.parametrize(
    ("smiles", "atom_idx", "error_fragment"),
    [
        ("C", 0, "has 0 neighbors"),  # methane: no heavy-atom neighbors
        ("CCC", 1, "has 2 neighbors"),  # propane center C: two neighbors
    ],
)
def test_get_sole_neighbor_raises_when_not_exactly_one(
    smiles: str, atom_idx: int, error_fragment: str
) -> None:
    mol = mol_from_smiles(smiles)
    with pytest.raises(ValueError, match=error_fragment):
        get_sole_neighbor(mol.GetAtomWithIdx(atom_idx))


def test_get_sole_neighbor() -> None:
    # In C-I, carbon has exactly one neighbor: iodine.
    mol = mol_from_smiles("CI")
    neighbor = get_sole_neighbor(mol.GetAtomWithIdx(0))
    assert neighbor.GetSymbol() == "I"


@pytest.mark.parametrize(
    ("smiles", "map_num", "num", "error_fragment"),
    [
        ("C", 1, 1, "is not 1"),  # no [*:1] at all
        ("C[*:1]", 1, 2, "is not 2"),  # only one [*:1]; asking for two
    ],
)
def test_get_mapped_atoms_raises_when_count_mismatch(
    smiles: str, map_num: int, num: Literal[1, 2], error_fragment: str
) -> None:
    mol = mol_from_smiles(smiles)
    with pytest.raises(ValueError, match=error_fragment):
        get_mapped_atoms(mol, map_num, num=num)


def test_get_mapped_atoms_single() -> None:
    mol = mol_from_smiles("C[*:1]")
    atom = get_mapped_atoms(mol, 1, num=1)
    assert atom.GetAtomicNum() == 0  # dummy atom


def test_get_mapped_atoms_double() -> None:
    mol = mol_from_smiles("C[*:1].S[*:1]")
    atom1, atom2 = get_mapped_atoms(mol, 1, num=2)
    assert atom1.GetAtomicNum() == 0
    assert atom2.GetAtomicNum() == 0
    assert atom1.GetIdx() != atom2.GetIdx()


def test_mask_hydrogens_raises_when_symbol_already_present() -> None:
    # [H][*:1]I already contains iodine, so "I" cannot be used as the mask.
    to_mask = mol_from_smiles("[H][*:1]I")
    with pytest.raises(ValueError, match="Mask symbol already in the molecule"):
        mask_hydrogens(to_mask, on=1, symbol="I")


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("[H][*:1]", "[400I][*:1]"),  # replaces hydrogens
        (  # Calling again on an already-masked mol must leave it unchanged.
            "[400I][*:1]",
            "[400I][*:1]",
        ),
    ],
)
def test_mask_hydrogens(smiles: str, expected: str) -> None:
    to_mask = mol_from_smiles(smiles)
    masked = mask_hydrogens(to_mask, on=1, symbol="I")
    assert_mols_equal(masked, expected)


def test_clear_special_isotopes_removes_200_and_400_tags() -> None:
    # Isotopes >= 200 are internal markers; clearing them removes the offset.
    mol = mol_from_smiles("IC([200I])([400I])")
    cleared = clear_special_isotopes(mol)
    assert_mols_equal(cleared, "IC(I)(I)")


@pytest.mark.parametrize(
    ("smi_a", "smi_b", "on", "bond_order", "expected"),
    [
        ("CC[*:1]", "CO[*:1]", 1, None, "CCOC"),  # single bond (default)
        ("CC[*:1]", "OC[*:1]", 1, Chem.BondType.DOUBLE, "CC=CO"),  # explicit double bond
    ],
)
def test_join_mols(
    smi_a: str,
    smi_b: str,
    on: int,
    bond_order: Chem.BondType | None,
    expected: str,
) -> None:
    joined = rejoin.join_mols(
        mol_from_smiles(smi_a), mol_from_smiles(smi_b), on=on, order=bond_order
    )
    assert_mols_equal(joined, expected)


def test_join_multiple_attaches_two_rgroups_to_core() -> None:
    core = mol_from_smiles("C(O[*:2])C[*:1]")
    groups = {
        "R1": Chem.MolFromSmiles("CO[*:1]"),
        "R2": Chem.MolFromSmiles("CC[*:2]"),
    }
    joined = join_multiple(core, groups)
    assert_mols_equal(joined, "C(OCC)COC")
