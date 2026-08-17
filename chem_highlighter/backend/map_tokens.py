"""Map SMILES tokens to RDKit indices."""

# ruff: noqa: S105
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from rdkit import Chem


class SmilesTokenMap(NamedTuple):
    """Container for RDKit data associated with SMILES token."""

    token: str
    type: Literal["bond", "impl_bond", "atom", "ring", "par", "info"]
    belongs_to: Literal["atom", "bond"]
    ix: int


def map_smiles_tokens(  # noqa: PLR0915
    smiles: str, mol: Chem.Mol, tokenize_fn: Callable[[str], list[str]] | None = None
) -> list[SmilesTokenMap]:
    """Map SMILES tokens to structural roles and RDKit indices.

    Requires `smiles` generated with `allBondsExplicit=True`.
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

        # handle bracketed atoms (e.g., [nH], [13C]).
        if token == "[":
            end_ix = tokens.index("]", ix)
            token_chars = "".join(tokens[ix : end_ix + 1])

            orig_atom_ix = atom_order[atom_ix]
            current_atom_index = orig_atom_ix

            # TODO(tihoph): ensure multi-character tokens (e.g., [nH], Cl)
            # are handled as single units.
            token_maps.append(SmilesTokenMap(token_chars, "atom", "atom", orig_atom_ix))

            atom_ix += 1
            ix = end_ix + 1

        elif token.startswith("[") and token.endswith("]"):
            orig_atom_ix = atom_order[atom_ix]
            current_atom_index = orig_atom_ix

            token_maps.append(SmilesTokenMap(token, "atom", "atom", orig_atom_ix))

            atom_ix += 1
            ix += 1

        # handle branches.
        elif token == "(":
            # map branch opening to first atom inside branch.
            branch_head_atom = atom_order[atom_ix]
            token_maps.append(SmilesTokenMap(token, "par", "atom", branch_head_atom))

            # track parent atom to resume main chain traversal.
            branch_stack.append(current_atom_index)
            ix += 1

        elif token == ")":
            # map branch closing to last atom inside branch.
            token_maps.append(SmilesTokenMap(token, "par", "atom", current_atom_index))

            # pop stack to revert main chain to parent atom.
            current_atom_index = branch_stack.pop() if branch_stack else current_atom_index
            ix += 1

        # handle ring closure tokens.
        elif token == "%":
            # handle tokenizer-dependent tokenization.
            token_chars = "".join(tokens[ix : ix + 3])
            token_maps.append(SmilesTokenMap(token_chars, "ring", "atom", current_atom_index))
            ix += 3

        elif token.isdigit() or (token.startswith("%") and token[1:].isdigit()):
            token_maps.append(SmilesTokenMap(token, "ring", "atom", current_atom_index))
            ix += 1

        # handle bond tokens.
        elif token in {"-", "=", "#", "/", "\\", ":", "~"}:
            orig_bond_ix = bond_order[bond_ix] if bond_ix < len(bond_order) else -1
            bond_type: Literal["impl_bond", "bond"] = "impl_bond" if token in {"-", ":"} else "bond"
            token_maps.append(SmilesTokenMap(token, bond_type, "bond", orig_bond_ix))

            bond_ix += 1
            ix += 1

        # handle unbracketed organic subset atoms.
        else:
            token_chars = token
            n_tokens = 1
            # handle tokenizer-dependent tokenization.
            if ix + 1 < len(tokens) and "".join(tokens[ix : ix + 2]) in {"Cl", "Br"}:
                token_chars = "".join(tokens[ix : ix + 2])
                n_tokens = 2

            orig_atom_ix = atom_order[atom_ix]
            current_atom_index = orig_atom_ix

            token_maps.append(SmilesTokenMap(token_chars, "atom", "atom", orig_atom_ix))

            atom_ix += 1
            ix += n_tokens

    return token_maps
