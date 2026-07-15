"""Configuration and fixtures for tests."""

from __future__ import annotations

from rdkit import Chem

from chem_highlighter.utils import mol_from_smiles


def assert_mols_equal(result: Chem.Mol | str, expected: Chem.Mol | str) -> None:
    if isinstance(result, str):
        result = mol_from_smiles(result)
    if isinstance(expected, str):
        expected = mol_from_smiles(expected)
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(expected), (
        f"{Chem.MolToSmiles(result)} != {Chem.MolToSmiles(expected)}"
    )
