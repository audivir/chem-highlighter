"""Tests for chem_highlighter.utils."""

from __future__ import annotations

import pytest
from conftest import assert_mols_equal
from rdkit import Chem

from chem_highlighter.utils import (
    add_hydrogens,
    get_atoms,
    get_bonds,
    get_neighbors,
    get_smiles_mol_pair,
    mol_from_smiles,
    mol_to_smiles,
)


@pytest.mark.parametrize(
    ("smiles", "expected_atoms", "expected_bonds"),
    [
        ("CCO", 3, 2),  # ethanol: C-C-O
        ("C1CCCCC1", 6, 6),  # cyclohexane: saturated 6-ring
        ("c1ccccc1", 6, 6),  # benzene: aromatic 6-ring
        ("CC(=O)O", 4, 3),  # acetic acid
    ],
)
def test_mol_from_smiles_valid(smiles: str, expected_atoms: int, expected_bonds: int) -> None:
    mol = mol_from_smiles(smiles)
    assert mol.GetNumAtoms() == expected_atoms
    assert mol.GetNumBonds() == expected_bonds


@pytest.mark.parametrize(
    "invalid_smiles",
    [
        "ZZZ",  # no such element
        "",  # empty string → 0-atom mol
        "invalid",  # gibberish
    ],
)
def test_mol_from_smiles_invalid_raises(invalid_smiles: str) -> None:
    with pytest.raises(ValueError, match="Invalid SMILES"):
        mol_from_smiles(invalid_smiles)


@pytest.mark.parametrize("smiles", ["CCO", "CCC", "c1ccccc1", "CC(=O)O"])
def test_mol_to_smiles(smiles: str) -> None:
    # Converting to mol and back must give an equivalent molecule.
    mol = mol_from_smiles(smiles)
    result = mol_to_smiles(mol)
    assert_mols_equal(result, smiles)


def test_mol_to_smiles_empty_mol_raises() -> None:
    # An empty (0-atom) molecule has no valid SMILES representation.
    empty = Chem.RWMol()
    with pytest.raises(ValueError, match="Empty SMILES"):
        mol_to_smiles(empty)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("CC(C)", "CC(C)"),  # When given a string, the pair stores it verbatim (not canonicalized).
        (  # When given a Mol, the stored SMILES is the canonical form.
            Chem.MolFromSmiles("CCC"),
            "CCC",
        ),
    ],
)
def test_get_smiles_mol_pair(data: str | Chem.Mol, expected: str) -> None:
    pair = get_smiles_mol_pair(data)
    assert pair.smiles == expected
    assert_mols_equal(pair.mol, "CCC")


def test_get_atoms() -> None:
    mol = mol_from_smiles("CCC")
    atoms = get_atoms(mol)
    assert all(isinstance(a, Chem.Atom) for a in atoms)


def test_get_bonds() -> None:
    mol = mol_from_smiles("CCO")
    bonds = get_bonds(mol)
    assert all(isinstance(b, Chem.Bond) for b in bonds)


@pytest.mark.parametrize(
    ("ix", "n_neighbors"),
    [
        (0, 1),  # In CCO, atom 0 (first C) is bonded only to atom 1 (second C).
        (1, 2),  # In CCO, atom 1 (second C) is bonded to atom 0 (C) and atom 2 (O).
    ],
)
def test_get_neighbors(ix: int, n_neighbors: int) -> None:
    mol = mol_from_smiles("CCO")
    terminal_c = mol.GetAtomWithIdx(ix)
    assert len(get_neighbors(terminal_c)) == n_neighbors


def test_add_hydrogens() -> None:
    mol = mol_from_smiles("CCO")
    heavy_count = mol.GetNumAtoms()
    [with_hs] = add_hydrogens([mol])
    assert with_hs.GetNumAtoms() > heavy_count
    assert any(a.GetSymbol() == "H" for a in get_atoms(with_hs))
