"""Keep and restore a state of RDKit atoms."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import msgspec
from typing_extensions import Self

from chem_highlighter.utils import get_atoms

if TYPE_CHECKING:
    from rdkit import Chem


class AtomState(msgspec.Struct):
    """Store the state of an atom.

    The classes' `freeze`/`unfreeze`-methods are meant to be used together as a pair.
    """

    num_explicit_hs: int
    no_implicit: bool
    formal_charge: int
    num_radical_electrons: int

    @classmethod
    def from_atom(cls, atom: Chem.Atom) -> Self:
        """Create an atom state from an RDKit atom."""
        return cls(
            atom.GetNumExplicitHs(),
            atom.GetNoImplicit(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
        )

    def to_atom(self, atom: Chem.Atom) -> None:
        """Set the atom state to an RDKit atom."""
        atom.SetNumExplicitHs(self.num_explicit_hs)
        atom.SetNoImplicit(self.no_implicit)
        atom.SetFormalCharge(self.formal_charge)
        atom.SetNumRadicalElectrons(self.num_radical_electrons)

    @classmethod
    def freeze(cls, mol: Chem.Mol) -> str:
        """Store each atom's state in a 'state_{tag}' property.

        Mutates `mol` in place: every atom is given a new property named
        after the returned tag. The matching `unfreeze` call must always be
        made to remove it again.

        Returns:
            The property's tag.
        """
        tag = f"state_{uuid.uuid4()}"
        for atom in get_atoms(mol):
            state = cls.from_atom(atom)
            atom.SetProp(tag, msgspec.json.encode(state).decode())
        return tag

    @classmethod
    def unfreeze(cls: type[Self], mol: Chem.Mol, tag: str) -> Chem.Mol:  # type: ignore[redundant-self]
        """Reset to the previous states and remove untagged molecules.

        Removes the `tag` property (added by `freeze`) from every atom that
        carries it and restores its stored state; atoms without the tag are
        assumed to have been added after `freeze` and are removed. This
        clears the property only from `mol`'s own atoms. Use this function's
        output instead of the original `mol` passed to `freeze`.

        Returns:
            The reset molecule.
        """
        from rdkit import Chem

        remove_ix: list[int] = []
        states: dict[int, Self] = {}
        decoder = msgspec.json.Decoder(cls)
        for atom in get_atoms(mol):
            if not atom.HasProp(tag):
                remove_ix.append(atom.GetIdx())
            else:
                state_json = atom.GetProp(tag)
                atom.ClearProp(tag)
                states[atom.GetIdx()] = decoder.decode(state_json)
        editable = Chem.EditableMol(mol)
        for ix in sorted(remove_ix, reverse=True):
            editable.RemoveAtom(ix)
        mol = editable.GetMol()
        for ix, state in states.items():
            state.to_atom(mol.GetAtomWithIdx(ix))
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        return mol
