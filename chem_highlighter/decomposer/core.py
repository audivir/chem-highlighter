"""Core decomposition data structures and algorithms."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chem_highlighter.utils import add_hydrogens, get_atoms, get_smiles_mol_pair, mol_to_smiles

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from rdkit import Chem


def create_qcore(core: str | Chem.Mol) -> Chem.Mol:
    """Create a query core."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    core = get_smiles_mol_pair(core).mol

    ps = Chem.AdjustQueryParameters.NoAdjustments()
    ps.makeDummiesQueries = True  # type: ignore[assignment]

    if not core.GetNumConformers():  # pragma: no cover
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(core)

    return Chem.AdjustQueryProperties(core, ps)


def set_source_idx(mols: Sequence[Chem.Mol]) -> None:
    """Set the source index for all atoms in each molecule."""
    for mol in mols:
        for atom in get_atoms(mol):
            atom.SetIntProp("SourceAtomIdx", atom.GetIdx())


def decompose(
    core: str | Chem.Mol, data: Sequence[str | Chem.Mol]
) -> tuple[Chem.Mol, dict[int, tuple[Chem.Mol, dict[str, Chem.Mol]]]]:
    """Decompose a set of molecules around a common core structure."""
    from rdkit import Chem
    from rdkit.Chem import rdRGroupDecomposition

    mols = [get_smiles_mol_pair(d).mol for d in data]

    qcore = create_qcore(core)

    non_match_idx = [ix for ix, x in enumerate(mols) if not x.HasSubstructMatch(qcore)]
    matches = set(range(len(mols))) - set(non_match_idx)

    if not matches:  # pragma: no cover
        return qcore, {}

    mols = [mols[ix] for ix in matches]
    mols = add_hydrogens(mols)
    set_source_idx(mols)

    groups: list[dict[str, Chem.Mol]]
    groups, rdkit_non_matched = rdRGroupDecomposition.RGroupDecompose([qcore], mols)

    if rdkit_non_matched:  # pragma: no cover
        raise ValueError("RGroupDecomposition failed for some of the molecules")

    return qcore, {
        ix: (m, {k: Chem.Mol(v) for k, v in g.items()})
        for ix, m, g in zip(matches, mols, groups, strict=True)
    }


def check_legends(n_mols: int, legends: Sequence[str] | None = None) -> list[str | None]:
    """Find matching molecules."""
    return [legend or None for legend in legends] if legends else [None] * n_mols


@dataclass
class DecomposedMol:
    """A decomposed molecule."""

    qcore: Chem.Mol
    complete: Chem.Mol
    groups: dict[str, Chem.Mol]
    legend: str | None = None

    @property
    def labels(self) -> list[str]:
        """Keys without Core key."""
        return [k for k in self.groups if k.startswith("R")]

    @property
    def core(self) -> Chem.Mol:
        """The core."""
        return self.groups["Core"]


class DecomposedList:
    """Validates an Iterable of DecomposedMol objects."""

    def __init__(self, decomposed_mols: Iterable[DecomposedMol]) -> None:  # pragma: no cover
        """Initialize the decomposed list."""
        self._decomposed = list(decomposed_mols)

        if not self._decomposed:
            raise ValueError("No decomposed molecules found.")

        if not all(isinstance(x, DecomposedMol) for x in self._decomposed):
            types = Counter(type(x) for x in self._decomposed)
            raise ValueError(f"All elements must be of type DecomposedMol: {types}")

        qcore_smi = mol_to_smiles(self._decomposed[0].qcore)
        if not all(mol_to_smiles(x.qcore) == qcore_smi for x in self._decomposed):
            qcores = Counter(mol_to_smiles(x.qcore) for x in self._decomposed)
            raise ValueError(f"All molecules must have the same core: {qcores}")

        core_smi = mol_to_smiles(self._decomposed[0].core)
        if not all(mol_to_smiles(x.core) == core_smi for x in self._decomposed):
            cores = Counter(mol_to_smiles(x.core) for x in self._decomposed)
            raise ValueError(f"All molecules must have the same core: {cores}")

        labels = self._decomposed[0].labels
        if not all(x.labels == labels for x in self._decomposed):
            labels_counter = Counter(x.labels for x in self._decomposed)
            raise ValueError(f"All molecules must have the same labels: {labels_counter}")

        ## All have to be different
        complete_smis = [mol_to_smiles(x.complete) for x in self._decomposed]
        if len(complete_smis) != len(set(complete_smis)):
            dups = [k for k, v in Counter(complete_smis).items() if v > 1]
            raise ValueError(f"All molecules must be different: {dups}")

        self.core = self._decomposed[0].core
        self.qcore = self._decomposed[0].qcore
        self.labels = labels

    def __getitem__(self, key: int) -> DecomposedMol:
        return self._decomposed[key]

    def __len__(self) -> int:
        return len(self._decomposed)

    def __iter__(self) -> Iterator[DecomposedMol]:
        return iter(self._decomposed)

    def build_rdata(self) -> dict[str, list[Chem.Mol]]:
        """Build the rdata."""
        rdata: dict[str, list[Chem.Mol]] = {key: [] for key in self.labels}
        for x in self:
            for key in self.labels:
                rdata[key].append(x.groups[key])
        return rdata


class Core:
    """Core class."""

    def __init__(self, data: str | Chem.Mol) -> None:
        """Initialize the core."""
        self.mol = get_smiles_mol_pair(data).mol

    def decompose(
        self,
        data: Sequence[str | Chem.Mol],
        legends: Sequence[str] | None = None,
        skip_cores: Iterable[str] | None = None,
    ) -> tuple[DecomposedList, list[int]]:
        """Decompose a list of SMILES into R-groups.

        Returns the DecomposedList and the non-matching indices.
        """
        final_legends = check_legends(len(data), legends)

        qcore, decomposed_data = decompose(self.mol, data)

        decomposed_mols: list[DecomposedMol] = []
        for ix, (mol, groups) in sorted(decomposed_data.items()):
            decomposed_mols.append(DecomposedMol(qcore, mol, groups, final_legends[ix]))

        skip_cores = set(skip_cores) if skip_cores is not None else set()
        decomposed_mols = [x for x in decomposed_mols if mol_to_smiles(x.core) not in skip_cores]

        non_match_idx = set(range(len(data))) - set(decomposed_data)
        return DecomposedList(decomposed_mols), sorted(non_match_idx)
