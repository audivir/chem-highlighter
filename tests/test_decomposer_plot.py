"""Tests for chem_highlighter.decomposer.plot."""

from __future__ import annotations

import pytest
from conftest import assert_mols_equal
from matplotlib.figure import Figure
from rdkit import Chem

from chem_highlighter.backend.rdkit import RDKitDocument
from chem_highlighter.decomposer.core import Core, DecomposedMol
from chem_highlighter.decomposer.plot import (
    _change_map_atoms,
    _fix_core_equivs,
    _fix_rdata_equivs,
    _switch_equiv,
    colorbar,
    fix_decomposed,
    plot_decomposed,
    replace_atom,
)
from chem_highlighter.utils import get_atoms, mol_from_smiles, mol_to_smiles


def test_change_map_atoms_updates_the_targeted_number() -> None:
    mol = mol_from_smiles("C[*:1]")
    result = _change_map_atoms(mol, from_=1, to=3)
    smi = Chem.MolToSmiles(result)
    assert "[*:1]" not in smi
    assert "[*:3]" in smi


def test_change_map_atoms_leaves_other_map_numbers_unchanged() -> None:
    mol = mol_from_smiles("C[*:3]")
    result = _change_map_atoms(mol, from_=1, to=5)  # no atom has map 1
    smi = Chem.MolToSmiles(result)
    assert "[*:3]" in smi
    assert "[*:5]" not in smi


def test_change_map_atoms_does_not_mutate_input() -> None:
    mol = mol_from_smiles("C[*:1]")
    _change_map_atoms(mol, from_=1, to=9)
    # The original must be untouched.
    assert "[*:1]" in Chem.MolToSmiles(mol)


@pytest.mark.parametrize(
    ("smi_a", "map_a", "smi_b", "map_b", "expected_a", "expected_b"),
    [
        ("C[*:1]", 1, "N[*:2]", 2, "N[*:1]", "C[*:2]"),
        ("CC[*:1]", 1, "OC[*:2]", 2, "OC[*:1]", "CC[*:2]"),
    ],
)
def test_switch_equiv_swaps_substituents_and_map_numbers(
    smi_a: str,
    map_a: int,
    smi_b: str,
    map_b: int,
    expected_a: str,
    expected_b: str,
) -> None:
    # After switching, each mol carries the other's atom structure
    # but retains its own map number label.
    a_new, b_new = _switch_equiv(
        (mol_from_smiles(smi_a), map_a),
        (mol_from_smiles(smi_b), map_b),
    )
    assert_mols_equal(a_new, expected_a)
    assert_mols_equal(b_new, expected_b)


def test_fix_rdata_equivs_swaps_position_where_b_is_more_common() -> None:
    # Pool counts: [H][*:1]=2, C[*:1]=1, C[*:2]=1, [H][*:2]=2
    # At position 2: count(R2=[H][*:2])=2 > count(R1=C[*:1])=1 → swap.
    rdata: dict[str, list[Chem.Mol]] = {
        "R1": [
            mol_from_smiles("[H][*:1]"),
            mol_from_smiles("[H][*:1]"),
            mol_from_smiles("C[*:1]"),  # ← swapped out
        ],
        "R2": [
            mol_from_smiles("C[*:2]"),
            mol_from_smiles("[H][*:2]"),
            mol_from_smiles("[H][*:2]"),  # ← swapped in
        ],
    }
    fixed = _fix_rdata_equivs(rdata, equivalents=[(1, 2)])

    assert_mols_equal(fixed["R1"][0], "[H][*:1]")
    assert_mols_equal(fixed["R1"][1], "[H][*:1]")
    assert_mols_equal(fixed["R1"][2], "[H][*:1]")  # was C[*:1]

    assert_mols_equal(fixed["R2"][0], "C[*:2]")
    assert_mols_equal(fixed["R2"][1], "[H][*:2]")
    assert_mols_equal(fixed["R2"][2], "C[*:2]")  # was [H][*:2]


def test_fix_rdata_equivs_no_swap_when_counts_are_equal() -> None:
    # Pool counts: C[*:1]=3, N[*:2]=3.  No count is strictly greater → no swaps.
    rdata: dict[str, list[Chem.Mol]] = {
        "R1": [mol_from_smiles("C[*:1]")] * 3,
        "R2": [mol_from_smiles("N[*:2]")] * 3,
    }
    fixed = _fix_rdata_equivs(rdata, equivalents=[(1, 2)])
    for mol in fixed["R1"]:
        assert_mols_equal(mol, "C[*:1]")
    for mol in fixed["R2"]:
        assert_mols_equal(mol, "N[*:2]")


def test_fix_rdata_equivs_empty_equivalents_leaves_data_unchanged() -> None:
    rdata: dict[str, list[Chem.Mol]] = {
        "R1": [mol_from_smiles("C[*:1]"), mol_from_smiles("N[*:1]")],
        "R2": [mol_from_smiles("S[*:2]"), mol_from_smiles("P[*:2]")],
    }
    fixed = _fix_rdata_equivs(rdata, equivalents=[])
    assert_mols_equal(fixed["R1"][0], "C[*:1]")
    assert_mols_equal(fixed["R1"][1], "N[*:1]")
    assert_mols_equal(fixed["R2"][0], "S[*:2]")
    assert_mols_equal(fixed["R2"][1], "P[*:2]")


def test_fix_decomposed_none_equivalents_is_passthrough() -> None:
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS", "NOCCCOP"])
    fixed, rdata = fix_decomposed(decomposed, equivalents=None)

    # Same object returned unchanged.
    assert fixed is decomposed

    assert set(rdata) == {"R1", "R2"}
    assert_mols_equal(rdata["R1"][0], "C[*:1]")
    assert_mols_equal(rdata["R1"][1], "N[*:1]")
    assert_mols_equal(rdata["R2"][0], "S[*:2]")
    assert_mols_equal(rdata["R2"][1], "P[*:2]")


def test_fix_decomposed_with_equivalents_returns_valid_molecules() -> None:
    # OCCCO's two O-ends are NOT symmetric, so no actual swaps occur here,
    # but the function must still return well-formed molecules.
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS", "NOCCCOP"])
    _, rdata = fix_decomposed(decomposed, equivalents=[(1, 2)])

    assert set(rdata) == {"R1", "R2"}
    for key in ("R1", "R2"):
        for mol in rdata[key]:
            assert mol is not None
            assert mol.GetNumAtoms() > 0
            assert mol_to_smiles(mol)


def test_fix_decomposed_with_equivalents_preserves_per_molecule_substituents() -> None:
    # The SET of substituent atoms per molecule must be identical regardless of
    # whether equivalents are applied (only the R1/R2 label assignment may change).
    decomposed, _ = Core("OCCCO").decompose(["COCCCOS", "NOCCCOP"])
    _, rdata_plain = fix_decomposed(decomposed, equivalents=None)
    _, rdata_equiv = fix_decomposed(decomposed, equivalents=[(1, 2)])

    def substituent_atoms(mol: Chem.Mol) -> set[str]:
        return {a.GetSymbol() for a in get_atoms(mol) if a.GetAtomicNum() not in (0, 1)}

    for ix in range(len(decomposed)):
        pair_plain = substituent_atoms(rdata_plain["R1"][ix]) | substituent_atoms(
            rdata_plain["R2"][ix]
        )
        pair_equiv = substituent_atoms(rdata_equiv["R1"][ix]) | substituent_atoms(
            rdata_equiv["R2"][ix]
        )
        assert pair_plain == pair_equiv, (
            f"molecule {ix}: substituents changed: {pair_plain} vs {pair_equiv}"
        )


def test_fix_core_equivs() -> None:
    dummy = mol_from_smiles("C")

    # Build the "matching" mol so the SMILES comparison does not trigger a swap.
    h_smi = mol_to_smiles(mol_from_smiles("[H][*:1]"))
    r1_match = mol_from_smiles(h_smi.replace("[H]", "I"))
    decomposed_match = DecomposedMol(
        qcore=dummy, complete=dummy, groups={"R1": r1_match, "Core": dummy}
    )
    tracker_no_swap = {
        "R1": (mol_from_smiles("[H][*:1]"), 0.9),
        "R2": (mol_from_smiles("N[*:2]"), 0.4),
    }
    result_no_swap = _fix_core_equivs(decomposed_match, tracker_no_swap, [(1, 2)])
    assert result_no_swap["R1"][1] == 0.9  # ratio unchanged

    # A mismatch ("C[*:1]" != "I[*:1]") triggers a swap.
    r1_mismatch = mol_from_smiles("C[*:1]")
    decomposed_mismatch = DecomposedMol(
        qcore=dummy, complete=dummy, groups={"R1": r1_mismatch, "Core": dummy}
    )
    tracker_swap = {
        "R1": (mol_from_smiles("[H][*:1]"), 0.8),
        "R2": (mol_from_smiles("N[*:2]"), 0.5),
    }
    result_swap = _fix_core_equivs(decomposed_mismatch, tracker_swap, [(1, 2)])
    assert result_swap["R1"][1] == 0.5  # got b_ratio
    assert result_swap["R2"][1] == 0.8  # got a_ratio
    assert_mols_equal(result_swap["R1"][0], "N[*:1]")
    assert_mols_equal(result_swap["R2"][0], "[H][*:2]")


def test_replace_atom() -> None:
    mol = Chem.MolFromSmiles("Ic1ccccc1")
    doc = RDKitDocument.from_mol(mol)
    svg = doc.to_svg()
    decomposed = DecomposedMol(qcore=mol, complete=mol, groups={"Core": mol})
    result = replace_atom(svg, decomposed, symbol="I")
    assert "display: none" in result
    assert "stroke-dasharray" in result


def test_plot_decomposed() -> None:
    decomposed, _ = Core("c1ccccc1").decompose(["Cc1ccccc1", "CCc1ccccc1"])
    # fig=None → internal Figure(), title → colorbar label, ylim → ax limits
    final, _, svg, _ = plot_decomposed(decomposed, title="test", ylim=(0.0, 1.0))
    assert "<svg" in svg
    assert final.labels


def test_plot_decomposed_with_equivalents() -> None:
    decomposed, _ = Core("C([*:1])C([*:2])").decompose(["C(C)C(N)", "C(F)C(Br)", "C(O)C(Cl)"])
    fig = Figure()
    _, _, svg, _ = plot_decomposed(decomposed, equivalents=[(1, 2)], fig=fig)
    assert "<svg" in svg


def test_colorbar() -> None:
    fig = Figure()
    ax = colorbar(fig)
    assert ax is not None
