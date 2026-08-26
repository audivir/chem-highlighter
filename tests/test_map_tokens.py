"""Tests for map_tokens module."""

# ruff: noqa: S105
from __future__ import annotations

import re

import pytest
from rdkit import Chem

from chem_highlighter.backend.map_tokens import map_smiles_tokens


def canonical_smiles(smiles: str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    return mol, Chem.MolToSmiles(mol, canonical=True, allBondsExplicit=True)


@pytest.mark.parametrize(
    ("smiles", "expected_token"),
    [
        ("c1cc[nH]c1", "[nH]"),
        ("[13CH4]", "[13CH4]"),
        ("C[NH3+]", "[NH3+]"),
        ("C[O-]", "[O-]"),
    ],
)
def test_map_smiles_tokens_bracket_atom(smiles: str, expected_token: str) -> None:
    mol, smi = canonical_smiles(smiles)
    token_maps = map_smiles_tokens(smi, mol)
    bracket_tokens = [m.token for m in token_maps if m.token.startswith("[")]
    assert bracket_tokens == [expected_token]


@pytest.mark.parametrize("halogen", ["Cl", "Br"])
def test_map_smiles_tokens_unbracketed_halogen(halogen: str) -> None:
    mol, smiles = canonical_smiles(f"c1ccccc1{halogen}")
    token_maps = map_smiles_tokens(smiles, mol)
    halogen_tokens = [m for m in token_maps if m.token == halogen]
    assert len(halogen_tokens) == 1
    assert halogen_tokens[0].type == "atom"


def test_map_smiles_tokens_pre_merged_bracket_token() -> None:
    mol, smiles = canonical_smiles("c1cc[nH]c1")

    def tokenize(s: str) -> list[str]:
        return re.findall(r"\[[^\]]*\]|.", s)

    token_maps = map_smiles_tokens(smiles, mol, tokenize_fn=tokenize)
    bracket_maps = [m for m in token_maps if m.token == "[nH]"]
    assert len(bracket_maps) == 1
    assert bracket_maps[0].type == "atom"
    assert bracket_maps[0].belongs_to == "atom"


def test_map_smiles_tokens_two_digit_ring_closure() -> None:
    mol, smiles = canonical_smiles("C")
    token_maps = map_smiles_tokens(smiles + "%12", mol)
    assert token_maps[-1].token == "%12"
    assert token_maps[-1].type == "ring"
