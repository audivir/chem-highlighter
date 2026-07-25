"""Configuration and fixtures for tests."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from chem_highlighter.utils import mol_from_smiles

FIXTURES = Path(__file__).parent / "fixtures"


def from_fixture_molblock(fixture_file: str) -> Chem.Mol:
    return Chem.MolFromMolBlock((FIXTURES / fixture_file).read_text(), removeHs=False)


def assert_mols_equal(result: Chem.Mol | str, expected: Chem.Mol | str) -> None:
    if isinstance(result, str):
        result = mol_from_smiles(result)
    if isinstance(expected, str):
        expected = mol_from_smiles(expected)
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(expected), (
        f"{Chem.MolToSmiles(result)} != {Chem.MolToSmiles(expected)}"
    )
