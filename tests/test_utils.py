"""Tests for chem_highlighter.utils."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import assert_mols_equal, from_fixture_molblock
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.Draw import rdDepictor
from rdkit.Geometry import Point3D

from chem_highlighter.utils import (
    add_hydrogens,
    are_atoms_equal,
    are_bonds_equal,
    flatten_conformer_z,
    get_ansi_color,
    get_atom_position,
    get_atoms,
    get_bonds,
    get_mol_center,
    get_neighbors,
    get_smiles_mol_pair,
    is_same_conformer,
    mol_from_smiles,
    mol_to_smiles,
    move_molecule,
    raise_if_3d_molecule,
    recenter_mol,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


@pytest.mark.parametrize("smiles", ["CCO", "CCC", "C1=CC=CC=C1", "c1ccccc1", "CC(=O)O"])
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


def test_get_atom_position() -> None:
    mol = mol_from_smiles("CCC")
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = get_atom_position(conf, 0)
    assert pos.z == 0


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


def test_is_same_conformer() -> None:
    atol = 1e-5

    with pytest.raises(ValueError, match="Invalid molblocks"):
        is_same_conformer("not a molblock", "also not a molblock", atol=1e-5)

    ethanol = mol_from_smiles("CCO")
    rdDepictor.Compute2DCoords(ethanol)
    assert is_same_conformer(ethanol, Chem.Mol(ethanol), atol=atol)

    propanol = mol_from_smiles("CCCO")
    rdDepictor.Compute2DCoords(propanol)
    assert is_same_conformer(propanol, Chem.Mol(propanol), atol=atol)

    with pytest.raises(ValueError, match="Non-identical molecules"):
        is_same_conformer(ethanol, propanol, atol=atol)


def test_is_conformer_positions() -> None:
    atol = 1e-5
    ethanol = mol_from_smiles("CCO")
    rdDepictor.Compute2DCoords(ethanol)

    ethanol_copy = Chem.Mol(ethanol)
    ethanol_copy_conf = ethanol_copy.GetConformer()
    move_molecule(ethanol_copy_conf, np.array([0.5 * atol, 0.0, 0.0]))
    assert is_same_conformer(ethanol, ethanol_copy, atol=atol)

    # cumulative offset from the original positions is now 2 * atol
    move_molecule(ethanol_copy_conf, np.array([1.5 * atol, 0.0, 0.0]))
    assert not is_same_conformer(ethanol, ethanol_copy, atol=atol)


def test_is_same_conformer_bond_mismatch() -> None:
    mol_a = mol_from_smiles("c1ccccc1")
    rdDepictor.Compute2DCoords(mol_a)
    mol_b = Chem.Mol(mol_a)
    mol_b.GetBondWithIdx(0).SetBondType(Chem.BondType.DOUBLE)
    assert Chem.MolToSmiles(mol_a) == Chem.MolToSmiles(mol_b)
    assert not is_same_conformer(mol_a, mol_b, atol=1e-5)


def test_move_molecule() -> None:
    mol = mol_from_smiles("CCO")
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    original_positions = conf.GetPositions()
    offset = np.array([0.0, 3.0, 0.0])

    move_molecule(conf, offset)

    np.testing.assert_allclose(conf.GetPositions(), original_positions + offset)


def test_get_mol_center() -> None:
    q = from_fixture_molblock("ethanol.mol")

    np.testing.assert_allclose(get_mol_center(q), np.array([0.0, 0.0, 0.0]), atol=1e-5)


@pytest.mark.parametrize("new_center", [np.array([5.0, -3.0, 0.0]), np.array([0.0, 0.0, 0.0])])
@pytest.mark.parametrize(
    "prev_center",
    [np.array([5.0, -3.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 0.0]), None],
)
def test_recenter_mol(
    new_center: NDArray[np.float64],
    prev_center: NDArray[np.float64] | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    atol = 1e-5

    mol = mol_from_smiles("CCO")
    rdDepictor.Compute2DCoords(mol)

    with caplog.at_level(logging.WARNING):
        recenter_mol(mol, new_center, prev_center, atol=atol)

    np.testing.assert_allclose(get_mol_center(mol), new_center, atol=atol)

    if prev_center is not None and not np.allclose(new_center, prev_center, atol=atol):
        assert "shifted" in caplog.text
    else:
        assert "shifted" not in caplog.text


def test_flatten_conformer_z() -> None:
    mol = mol_from_smiles("CCO")
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, Point3D(pos.x, pos.y, 0.0005))

    flatten_conformer_z(mol, conf, atol=1e-5)

    assert conf.GetAtomPosition(0).z == 0.0


def test_raise_if_3d_molecule() -> None:
    atol = 1e-5
    mol = Chem.MolFromSmiles("c1ccccc1")
    rdDepictor.Compute2DCoords(mol)
    raise_if_3d_molecule(mol.GetConformer(), atol=atol)

    rdDistGeom.EmbedMolecule(mol)
    with pytest.raises(ValueError, match="Molecule is a 3D molecule"):
        raise_if_3d_molecule(mol.GetConformer(), atol=atol)


def test_are_atoms_equal() -> None:
    ethanol = mol_from_smiles("CCO")
    carbon, _, oxygen = get_atoms(ethanol)
    assert are_atoms_equal(carbon, ethanol.GetAtomWithIdx(1))
    assert not are_atoms_equal(carbon, oxygen)  # different atomic number

    charged_carbon = mol_from_smiles("[CH3+]").GetAtomWithIdx(0)
    assert not are_atoms_equal(carbon, charged_carbon)  # different formal charge

    aromatic_carbon = mol_from_smiles("c1ccccc1").GetAtomWithIdx(0)
    assert not are_atoms_equal(carbon, aromatic_carbon)  # different aromaticity


def test_are_bonds_equal() -> None:
    single_bond = mol_from_smiles("CCO").GetBondWithIdx(0)
    other_single_bond = mol_from_smiles("CCC").GetBondWithIdx(0)
    assert are_bonds_equal(single_bond, other_single_bond)

    double_bond = mol_from_smiles("C=CO").GetBondWithIdx(0)
    assert not are_bonds_equal(single_bond, double_bond)  # different bond type

    aromatic_bond = mol_from_smiles("c1ccccc1").GetBondWithIdx(0)
    assert not are_bonds_equal(single_bond, aromatic_bond)  # different aromaticity


def test_get_ansi_color() -> None:
    palette = ["#ff0000", "#00ff00"]
    assert get_ansi_color(palette, 0) == "\033[38;2;255;0;0m"
    assert get_ansi_color(palette, 1) == "\033[38;2;0;255;0m"
