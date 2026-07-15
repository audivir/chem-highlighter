"""Highlight differences between two SMILES strings."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Literal, TypeAlias

from chem_highlighter.utils import GREEN_COLOR, RED_COLOR, RESET_COLOR

if TYPE_CHECKING:
    from collections.abc import Sequence


Diff_T: TypeAlias = Literal["equal", "delete", "insert"]


def get_diff(s1: str, s2: str) -> tuple[list[tuple[Diff_T, str]], list[tuple[Diff_T, str]]]:
    """Create diff information of two strings."""
    matcher = difflib.SequenceMatcher(None, s1, s2)
    out_s1: list[tuple[Diff_T, str]] = []
    out_s2: list[tuple[Diff_T, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out_s1.append(("equal", s1[i1:i2]))
            out_s2.append(("equal", s2[j1:j2]))
        if tag in {"delete", "replace"}:
            out_s1.append(("delete", s1[i1:i2]))
        if tag in {"insert", "replace"}:
            out_s2.append(("insert", s2[j1:j2]))

    return out_s1, out_s2


def get_smiles_diff(
    query: str, new_smiles: str, n_augmentations: int = 100
) -> tuple[list[tuple[Diff_T, str]], list[tuple[Diff_T, str]]]:
    """Create smallest diff information of two SMILES strings."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(new_smiles)
    new_randomized = [Chem.MolToSmiles(mol, doRandom=True) for _ in range(n_augmentations)]
    new_randomized.append(Chem.MolToSmiles(mol))
    new_randomized_set = set(new_randomized)

    def ratio(smiles: str) -> float:
        return difflib.SequenceMatcher(None, query, smiles).ratio()

    new_best = max(new_randomized_set, key=ratio)

    return get_diff(query, new_best)


def colorize_smiles_diff(
    query: str, new_smiles: str, n_augmentations: int = 100
) -> tuple[str, str]:
    """Create colorized diff of two SMILES strings."""

    def _colorize(diff_texts: Sequence[tuple[Diff_T, str]]) -> str:
        out_parts: list[str] = []
        for diff, raw_text in diff_texts:
            if diff != "equal":
                color_code = RED_COLOR if diff == "delete" else GREEN_COLOR
                text = f"{color_code}{raw_text}{RESET_COLOR}"
            else:
                text = raw_text
            out_parts.append(text)
        return "".join(out_parts)

    query_parts, new_smiles_parts = get_smiles_diff(query, new_smiles, n_augmentations)
    return _colorize(query_parts), _colorize(new_smiles_parts)
