"""Tests for chem_highlighter.diff."""

from __future__ import annotations

import pytest

from chem_highlighter.diff import Diff_T, colorize_smiles_diff, get_diff, get_smiles_diff


def test_get_diff() -> None:
    expected = [("equal", "CC"), ("delete", "O")], [("insert", "O"), ("equal", "CC")]
    assert get_diff("CCO", "OCC") == expected


@pytest.mark.parametrize(
    ("query", "new_smiles", "expected"),
    [
        ("CCO", "OCC", ([("equal", "CCO")], [("equal", "CCO")])),
        (
            "CC(C)O",
            "OCCC",
            (
                [("delete", "C"), ("equal", "C(C)"), ("equal", "O")],
                [("equal", "C(C)"), ("insert", "C"), ("equal", "O")],
            ),
        ),
        (
            "OCCC",
            "CC(C)O",
            (
                [("equal", "OC"), ("equal", "C"), ("equal", "C")],
                [("equal", "OC"), ("insert", "("), ("equal", "C"), ("insert", ")"), ("equal", "C")],
            ),
        ),
    ],
)
def test_get_smiles_diff(
    query: str,
    new_smiles: str,
    expected: tuple[list[tuple[Diff_T, str]], list[tuple[Diff_T, str]]],
) -> None:
    assert get_smiles_diff(query, new_smiles) == expected


@pytest.mark.parametrize(
    ("query", "new_smiles", "expected"),
    [
        ("CCO", "OCC", ("CCO", "CCO")),
        (
            "CC(C)O",
            "OCCC",
            (
                "\033[91mC\033[0mC(C)O",
                "C(C)\x1b[92mC\x1b[0mO",
            ),
        ),
        (
            "OCCC",
            "CC(C)O",
            (
                "OCCC",
                "OC\033[92m(\033[0mC\033[92m)\033[0mC",
            ),
        ),
    ],
)
def test_colorize_smiles_diff(query: str, new_smiles: str, expected: tuple[str, str]) -> None:
    assert colorize_smiles_diff(query, new_smiles) == expected
