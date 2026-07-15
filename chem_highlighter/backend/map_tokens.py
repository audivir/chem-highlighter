"""Map SMILES tokens to its corresponding RDKit indices."""

# ruff: noqa: S105
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from rdkit import Chem


class SmilesTokenMap(NamedTuple):
    """Store RDKit data for SMILES token."""

    token: str
    type: Literal["bond", "impl_bond", "atom", "ring", "par", "info"]
    belongs_to: Literal["atom", "bond"]
    ix: int


def map_smiles_tokens(  # noqa: PLR0915
    smiles: str, mol: Chem.Mol, tokenize_fn: Callable[[str], list[str]] | None = None
) -> list[SmilesTokenMap]:
    """Maps each tokenacter in a SMILES string to its structural role and RDKit index.

    Assumes `smiles` was generated via `Chem.MolToSmiles(..., allBondsExplicit=True)`.
    """
    all_props = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)
    atom_order: list[int] = list(all_props["_smilesAtomOutputOrder"])
    bond_order: list[int] = list(all_props.get("_smilesBondOutputOrder", []))

    token_maps: list[SmilesTokenMap] = []

    atom_ix = 0
    bond_ix = 0
    ix = 0

    current_atom_index = -1
    branch_stack: list[int] = []

    tokens = tokenize_fn(smiles) if tokenize_fn else list(smiles)
    while ix < len(tokens):
        token = tokens[ix]

        # bracketed atoms (e.g., [nH], [13C])
        if token == "[":
            end_ix = tokens.index("]", ix)
            token_chars = "".join(tokens[ix : end_ix + 1])

            orig_atom_ix = atom_order[atom_ix]
            current_atom_index = orig_atom_ix

            token_maps.append(SmilesTokenMap(token_chars, "atom", "atom", orig_atom_ix))

            atom_ix += 1
            ix = end_ix + 1

        elif token.startswith("[") and token.endswith("]"):
            orig_atom_ix = atom_order[atom_ix]
            current_atom_index = orig_atom_ix

            token_maps.append(SmilesTokenMap(token, "atom", "atom", orig_atom_ix))

            atom_ix += 1
            ix += 1

        # branches
        elif token == "(":
            # Branch opens: Belongs to the FIRST atom INSIDE the branch.
            # atom_ix hasn't incremented yet, so atom_order[atom_ix] is the branch head.
            branch_head_atom = atom_order[atom_ix]
            token_maps.append(SmilesTokenMap(token, "par", "atom", branch_head_atom))

            # We still need to track the parent so the main chain knows where to resume
            branch_stack.append(current_atom_index)
            ix += 1

        elif token == ")":
            # Branch closes: Belongs to the LAST atom inside the branch (the current one).
            token_maps.append(SmilesTokenMap(token, "par", "atom", current_atom_index))

            # Pop the stack to revert the main chain back to the parent atom
            current_atom_index = branch_stack.pop() if branch_stack else current_atom_index
            ix += 1

        # ring closures "%", "1", "2"
        elif token == "%":
            token_chars = "".join(tokens[ix : ix + 3])
            token_maps.append(SmilesTokenMap(token_chars, "ring", "atom", current_atom_index))
            ix += 3

        elif token.isdigit() or (token.startswith("%") and token[1:].isdigit()):
            token_maps.append(SmilesTokenMap(token, "ring", "atom", current_atom_index))
            ix += 1

        # bonds
        elif token in {"-", "=", "#", "/", "\\", ":", "~"}:
            orig_bond_ix = bond_order[bond_ix] if bond_ix < len(bond_order) else -1
            bond_type: Literal["impl_bond", "bond"] = "impl_bond" if token in {"-", ":"} else "bond"
            token_maps.append(SmilesTokenMap(token, bond_type, "bond", orig_bond_ix))

            bond_ix += 1
            ix += 1

        # unbracketed organic subset atoms (including Cl, Br)
        else:
            token_chars = token
            n_tokens = 1
            if ix + 1 < len(tokens) and "".join(tokens[ix : ix + 2]) in {"Cl", "Br"}:
                token_chars = "".join(tokens[ix : ix + 2])
                n_tokens = 2

            orig_atom_ix = atom_order[atom_ix]
            current_atom_index = orig_atom_ix

            token_maps.append(SmilesTokenMap(token_chars, "atom", "atom", orig_atom_ix))

            atom_ix += 1
            ix += n_tokens

    return token_maps
