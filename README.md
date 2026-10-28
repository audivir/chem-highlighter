# chem-highlighter

Highlighting API for chemical molecules. Given a molecule and a set of atoms/bonds/rings to
color, produces highlighted SVG/PNG/console output, or plain format conversion (SDF, Mol, RXN,
CDX, CDXML, SMILES, InChI, InChIKey, SVG, EPS, PNG). Backend-agnostic: `RDKitMolecule` (built in,
uses RDKit for everything) implements the `HighlightBackendMolecule` ABC; other backends can
implement the same ABC against a different underlying engine, so code written against
`HighlightBackendMolecule` is not tied to RDKit specifically.

`Document` (also in `chem_highlighter/hml.py`) wraps a *list* of molecule-backend instances
parsed from a single input, for formats that can legitimately hold more than one structure: SDF
(every record, not just the first), RXN (reactants/agents/products), CDXML (any number of
fragments), and dot-separated SMILES. `HighlightBackendMolecule.from_bytes` keeps its existing
single-molecule restrictions unchanged (e.g. still rejects multi-fragment CDXML); `Document`
is the opt-in multi-molecule path built on top of it.

## Prerequisites

Python >= 3.10, < 3.15. RDKit, matplotlib, numpy, polars, scipy, msgspec are pulled in as
dependencies.

## Installation

```bash
pip install chem-highlighter
```

For a local development install:

```bash
pip install -e .
# or: uv sync
```

## Usage

```python
import msgspec
from chem_highlighter import RDKitMolecule, HML

doc = RDKitMolecule.from_string("c1ccccc1O", "SMILES")
doc.cleanup()

hml = HML(highlighted_atoms={6: 0}, palette=["#ff0000"])
doc.highlight_from_json(msgspec.json.encode(hml).decode())

svg = doc.to_svg()
png = doc.to_png()
```

`HighlightBackendMolecule` (`chem_highlighter/hml.py`) is the actual interface: construction
(`from_bytes`/`from_string`/`from_mol`/`from_molblock`), export (`export`/`export_string`/
`to_molblock`/`to_svg`/`to_png`/`to_console`), and editing (`cleanup`, `kekulize`,
`align_to_reference`, `hide_hydrogens`, `highlight_from_json`) — each editing method is one-shot
per document (calling it twice, or in the wrong order relative to another, raises `ValueError`);
see the class docstrings for the exact rules.

## Aromatic molblocks

`get_high_precision_v3000` writes aromatic bonds as bond type 4 unless `kekulize=True` is passed.
ChemDraw draws such rings with a circle, which looks different from the alternating bonds of a
kekulized ring. An aromatic molblock does not state which ring atom carries the hydrogen of an
aromatic NH, for example the `[nH]` of tetrazole, indole, or pyridone, so ChemDraw would draw an
invalid valence without the hydrogen.

The function therefore adds the V3000 atom property `HCOUNT=<n>` to every aromatic atom that is
not carbon and has hydrogens. ChemDraw reads it and draws the NH. RDKit ignores `HCOUNT` and
cannot kekulize such a molblock, so `Chem.MolFromMolBlock` returns `None`. Use
`chem_highlighter.utils.mol_from_molblock` instead. It applies `HCOUNT` as explicit hydrogens
before sanitizing and restores the aromatic molecule:

```python
from rdkit import Chem

from chem_highlighter.utils import get_high_precision_v3000, mol_from_molblock

mol = Chem.MolFromSmiles("c1nn[nH]n1")
molblock = get_high_precision_v3000(mol)
print(Chem.MolToSmiles(mol_from_molblock(molblock)))  # c1nn[nH]n1
```

`HCOUNT` is a convention of these tools and not part of the molfile standard, so other readers
ignore it. Pass `kekulize=True` for molblocks that other software has to read.

## Modules

- `hml` — the `HighlightBackendMolecule` ABC, the multi-molecule `Document` class, and
  `HML`/`HMol` highlight-payload types.
- `backend/rdkit.py` — `RDKitMolecule`, the RDKit-backed implementation.
- `align` — align one molecule to another via bond flips + rotation (used by
  `align_to_reference`).
- `decomposer` — R-group decomposition, core/residue splitting, and plotting decomposed sets.
- `diff` — highlight the difference between two SMILES strings.
- `modify` — rotate/mirror/flip-bond primitives on RDKit molecules.
- `state` — save and restore RDKit atom state (used internally by `modify`/`align`).
- `table` — render a Polars DataFrame as an AG Grid HTML table.
- `utils` — shared helpers: conformer comparison, high-precision V3000 export and import (with
  `HCOUNT`), PNG render options, color/console formatting.

## Environment variables

PNG rendering (`RDKitMolecule.export`/`to_png`, `utils.get_png_render_options`) reads:

- `CHEM_HIGHLIGHTER_PNG_WIDTH`, `CHEM_HIGHLIGHTER_PNG_HEIGHT` — bounding box in pixels; the
  molecule is scaled to fit and centered. If only one is set, the other mirrors it. Unset: keeps
  the default RDKit canvas sizing (unbounded).
- `CHEM_HIGHLIGHTER_PNG_TRANSPARENT` — `true` for a transparent background instead of white.

Named to match the PNG env vars of other backend implementations, so a caller using more than one
backend configures PNG output once.

## Testing

```bash
pytest
mypy .
ruff check .
```

Coverage is configured for 100% (the `[tool.coverage.report]` section of `pyproject.toml`,
`fail_under = 100`). `vulture` is configured to flag dead code (`[tool.vulture]`).

Some tests (the image/PNG-size tests in `tests/test_rdkit.py`, the shared assertions in
`tests/backend_test.py`) rely on native RDKit rendering; no native/OS-specific setup needed
beyond the pip install above.

## License

MIT, see `LICENSE`.
