"""Tests for the decomposer core."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rdkit import Chem
from utils import assert_mols_equal

from chem_highlighter.decomposer.core import (
    Core,
    DecomposedMol,
    check_legends,
    create_qcore,
    decompose,
    set_source_idx,
)
from chem_highlighter.utils import add_hydrogens, get_atoms, mol_to_smiles

if TYPE_CHECKING:
    from collections.abc import Mapping


def assert_group_equals(
    group: Mapping[str, Chem.Mol], expected: Mapping[str, str | Chem.Mol]
) -> None:
    assert set(group) == set(expected)
    for (k1, v1), (k2, v2) in zip(group.items(), expected.items(), strict=True):
        assert k1 == k2
        assert_mols_equal(v1, v2)


def test_decompose_returns_qcore_and_both_molecules() -> None:

    qcore, result = decompose("OCCCO", ["COCCCOS", "NOCCCOP"])
    assert_mols_equal(qcore, "OCCCO")
    assert set(result) == {0, 1}
    assert_mols_equal(result[0][0], add_hydrogens(["COCCCOS"])[0])
    assert_mols_equal(result[1][0], add_hydrogens(["NOCCCOP"])[0])


@pytest.mark.parametrize(
    ("mol_idx", "expected_groups"),
    [
        (0, {"Core": "C(CO[*:1])CO[*:2]", "R1": "C[*:1]", "R2": "S[*:2]"}),
        (1, {"Core": "C(CO[*:1])CO[*:2]", "R1": "N[*:1]", "R2": "P[*:2]"}),
    ],
)
def test_decompose_rgroups_correct(mol_idx: int, expected_groups: dict[str, str]) -> None:

    _, result = decompose("OCCCO", ["COCCCOS", "NOCCCOP"])
    assert_group_equals(result[mol_idx][1], expected_groups)


def test_decompose_non_matching_molecules_are_excluded() -> None:
    # verify that molecules without the target substructure are excluded.
    _, result = decompose("OCCCO", ["COCCCOS", "CCC", "NOCCCOP"])
    assert set(result) == {0, 2}
    assert_group_equals(result[0][1], {"Core": "C(CO[*:1])CO[*:2]", "R1": "C[*:1]", "R2": "S[*:2]"})
    assert_group_equals(result[2][1], {"Core": "C(CO[*:1])CO[*:2]", "R1": "N[*:1]", "R2": "P[*:2]"})


def test_decompose_single_rgroup_molecule() -> None:
    # verify that substituents vary on both ends of the OCCO core.
    _, result = decompose("OCCO", ["COCCOF", "NOCCOP"])
    r1_smis = {mol_to_smiles(result[ix][1]["R1"]) for ix in result}
    r2_smis = {mol_to_smiles(result[ix][1]["R2"]) for ix in result}
    # verify that both positions vary across the molecules.
    assert len(r1_smis) == 2
    assert len(r2_smis) == 2


@pytest.mark.parametrize(
    ("core_smiles", "probe_smiles", "should_match"),
    [
        ("OCCCO", "COCCCOS", True),  # verify that the probe contains the core.
        ("OCCCO", "CCC", False),  # verify that the probe is too short to match.
        ("c1ccccc1", "Cc1ccccc1", True),  # verify that toluene contains benzene.
        ("c1ccccc1", "CCCC", False),  # verify that butane has no ring.
    ],
)
def test_create_qcore_substructure_matching(
    core_smiles: str, probe_smiles: str, should_match: bool
) -> None:
    qcore = create_qcore(core_smiles)
    assert qcore.GetNumAtoms() > 0
    probe = Chem.MolFromSmiles(probe_smiles)
    assert bool(probe.HasSubstructMatch(qcore)) == should_match


def test_set_source_idx_assigns_each_atom_its_own_index() -> None:
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCO")]
    set_source_idx(mols)
    for mol in mols:
        for atom in get_atoms(mol):
            assert atom.GetIntProp("SourceAtomIdx") == atom.GetIdx()


def test_decomposed_mol_labels_contains_only_r_keys() -> None:
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS"])
    assert "Core" not in decomposed[0].labels
    assert set(decomposed[0].labels) == {"R1", "R2"}


def test_decomposed_mol_core_property_returns_core_molecule() -> None:
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS"])
    assert_mols_equal(decomposed[0].core, "C(CO[*:1])CO[*:2]")


def test_decomposed_mol_complete_is_the_full_molecule_with_hydrogens() -> None:
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS"])
    assert_mols_equal(decomposed[0].complete, add_hydrogens(["COCCCOS"])[0])


def test_decomposed_list_iteration_gives_correct_rgroups() -> None:
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS", "NOCCCOP"])
    mols = list(decomposed)
    assert all(isinstance(m, DecomposedMol) for m in mols)
    assert_mols_equal(mols[0].groups["R1"], "C[*:1]")
    assert_mols_equal(mols[0].groups["R2"], "S[*:2]")
    assert_mols_equal(mols[1].groups["R1"], "N[*:1]")
    assert_mols_equal(mols[1].groups["R2"], "P[*:2]")


def test_decomposed_list_indexing_gives_correct_rgroups() -> None:
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS", "NOCCCOP"])
    assert_mols_equal(decomposed[0].groups["R2"], "S[*:2]")
    assert_mols_equal(decomposed[1].groups["R2"], "P[*:2]")


def test_core_decomposed_list_full_check() -> None:
    decomposed, non_matched = Core("OCCCO").decompose(
        ["COCCCOS", "NOCCCOP", "CCC"], ["A", "B", "C"]
    )
    assert non_matched == [2]
    assert_mols_equal(decomposed.core, "C(CO[*:1])CO[*:2]")
    assert_mols_equal(decomposed.qcore, "OCCCO")
    assert len(decomposed) == 2
    assert decomposed[0].legend == "A"

    rdata = decomposed.build_rdata()
    assert set(rdata) == {"R1", "R2"}
    for mol, expected in zip(rdata["R1"], ["C[*:1]", "N[*:1]"], strict=True):
        assert_mols_equal(mol, expected)
    for mol, expected in zip(rdata["R2"], ["S[*:2]", "P[*:2]"], strict=True):
        assert_mols_equal(mol, expected)


def test_core_decompose_all_nonmatching_raises() -> None:
    with pytest.raises(ValueError, match="No decomposed molecules found"):
        Core("OCCCO").decompose(["CCC", "CCCC"])


def test_core_decompose_partial_match_preserves_original_indices() -> None:
    # verify that non-matching molecules are excluded
    # and matching molecules preserve original indices.
    decomposed, non_matched = Core("OCCCO").decompose(["COCCCOS", "CCC", "NOCCCOP"])
    assert non_matched == [1]
    assert len(decomposed) == 2
    assert_mols_equal(decomposed[0].groups["R1"], "C[*:1]")
    assert_mols_equal(decomposed[0].groups["R2"], "S[*:2]")
    assert_mols_equal(decomposed[1].groups["R1"], "N[*:1]")
    assert_mols_equal(decomposed[1].groups["R2"], "P[*:2]")


def test_core_decompose_legends_are_preserved_for_matched_molecules() -> None:
    decomposed, _ = Core("OCCCO").decompose(
        ["COCCCOS", "NOCCCOP", "CCC"],
        legends=["methyl-thiol", "amino-phospho", "ignored"],
    )
    assert decomposed[0].legend == "methyl-thiol"
    assert decomposed[1].legend == "amino-phospho"


def test_core_decompose_skip_cores_with_unknown_core_has_no_effect() -> None:
    # verify that skip_cores entries not present in results do not filter any molecules.
    decomposed, _ = Core("OCCCO").decompose(
        ["COCCCOS", "NOCCCOP"], skip_cores=["THIS_WILL_NEVER_MATCH"]
    )
    assert len(decomposed) == 2
    assert_mols_equal(decomposed[0].groups["R1"], "C[*:1]")
    assert_mols_equal(decomposed[1].groups["R1"], "N[*:1]")


def test_core_decompose_three_molecules_rgroup_substituents() -> None:
    # verify substituent atom sets to avoid dependence on RDKit R-group assignment order.
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS", "NOCCCOP", "COCCCON"])
    rdata = decomposed.build_rdata()
    assert set(rdata) == {"R1", "R2"}
    assert len(rdata["R1"]) == 3

    def substituent_atoms(mol: Chem.Mol) -> set[str]:
        return {a.GetSymbol() for a in get_atoms(mol) if a.GetAtomicNum() not in (0, 1)}

    for mol_idx, expected_pair in [
        (0, {"C", "S"}),
        (1, {"N", "P"}),
        (2, {"C", "N"}),
    ]:
        actual = substituent_atoms(rdata["R1"][mol_idx]) | substituent_atoms(rdata["R2"][mol_idx])
        assert actual == expected_pair, f"molecule {mol_idx}: {actual} != {expected_pair}"


@pytest.mark.parametrize(
    ("n", "legends", "expected"),
    [
        (3, ["A", "B", "C"], ["A", "B", "C"]),  # verify that provided legends are preserved.
        (3, ["A", "", "C"], ["A", None, "C"]),  # verify that empty strings are converted to None.
        (
            4,
            None,
            [None, None, None, None],
        ),  # verify that missing legends result in a list of None values.
    ],
)
def test_check_legends(n: int, legends: list[str] | None, expected: list[str | None]) -> None:
    assert check_legends(n, legends) == expected
