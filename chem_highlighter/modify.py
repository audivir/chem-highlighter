"""Modify a molecule, namely rotate, mirror, or flip bonds."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, NamedTuple

import msgspec
from rdkit.Chem.rdMolTransforms import TransformConformer
from typing_extensions import Self

from chem_highlighter.utils import get_atoms

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from rdkit import Chem


class Position3D(NamedTuple):
    """A 3D position."""

    x: float
    y: float
    z: float


class AtomState(msgspec.Struct):
    """Store the state of an atom."""

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


def make_transform(
    angle_deg: float = 0.0,
    *,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> NDArray[np.float64]:
    """Create a 2D transform matrix.

    Args:
        angle_deg: Counterclockwise rotation angle in degrees.
        flip_horizontal: Mirror about the vertical axis.
        flip_vertical: Mirror about the horizontal axis.
    """
    import numpy as np

    theta = np.radians(angle_deg % 360.0)
    c = np.cos(theta)
    s = np.sin(theta)

    matrix = np.array(
        [
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    Rx180 = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    Ry180 = np.array(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    if flip_horizontal:
        matrix = matrix @ Ry180

    if flip_vertical:
        matrix = matrix @ Rx180

    # if flip_horizontal:
    #     # 180° rotation about Z (equivalent to a horizontal flip in 2D)
    #     matrix[0, :2] *= -1.0
    #     matrix[1, :2] *= -1.0

    # if flip_vertical:
    #     # 180° rotation about X (equivalent to a vertical flip in 2D)
    #     matrix[1, :2] *= -1.0
    #     matrix[2, 2] = -1.0

    return matrix


def apply_transform(
    mol: Chem.Mol, angle_deg: float, *, flip_horizontal: bool = False, flip_vertical: bool = False
) -> Chem.Mol:
    """Rotate and/or mirror a (flat, z=0) conformer within its own plane.

    Args:
        angle_deg: Counterclockwise rotation angle in degrees.
        flip_horizontal: Mirror about the vertical axis.
        flip_vertical: Mirror about the horizontal axis.
    """
    import numpy as np
    from rdkit import Chem

    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    matrix = make_transform(angle_deg, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
    TransformConformer(conf, matrix)

    positions = np.array(conf.GetPositions())
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    new_positions = [Position3D(*p) for p in positions - center]
    for i, new_pos in enumerate(new_positions):
        conf.SetAtomPosition(i, new_pos)

    return mol


def flip_bond(
    mol: Chem.Mol,
    bond_ix: int,
    anchor_atom_ix: int,
) -> Chem.Mol:
    """Flip a rotatable bond by 180°.

    Args:
        mol: Molecule whose conformer will be modified in-place.
        bond_ix: Bond index to rotate.
        anchor_atom_ix: Atom on the bond that should remain fixed.

    Returns:
        Mol with flipped bond.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    bond = mol.GetBondWithIdx(bond_ix)

    if bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Only single bonds can be flipped")

    if bond.GetIsAromatic():
        raise ValueError("Cannot rotate aromatic bonds")

    begin = bond.GetBeginAtomIdx()
    end = bond.GetEndAtomIdx()

    if anchor_atom_ix == begin:
        idx_b = begin
        idx_c = end
    elif anchor_atom_ix == end:
        idx_b = end
        idx_c = begin
    else:
        raise ValueError("anchor_atom_ix is not part of the specified bond")

    tag = AtomState.freeze(mol)

    mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=[idx_b, idx_c])

    atom_b = mol.GetAtomWithIdx(idx_b)
    atom_c = mol.GetAtomWithIdx(idx_c)

    neighbors_b = [n.GetIdx() for n in atom_b.GetNeighbors() if n.GetIdx() != idx_c]
    neighbors_c = [n.GetIdx() for n in atom_c.GetNeighbors() if n.GetIdx() != idx_b]

    if not neighbors_b:
        raise ValueError("Anchor atom has no suitable neighboring atom")
    if not neighbors_c:
        raise ValueError("Rotating atom has no suitable neighboring atom")

    idx_a = neighbors_b[0]
    idx_d = neighbors_c[0]

    conf = mol.GetConformer()

    angle = rdMolTransforms.GetDihedralDeg(conf, idx_a, idx_b, idx_c, idx_d)

    rdMolTransforms.SetDihedralDeg(
        conf,
        idx_a,
        idx_b,
        idx_c,
        idx_d,
        angle + 180.0,
    )

    return AtomState.unfreeze(mol, tag)
