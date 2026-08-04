# chem-highlighter

Highlighting API for chemical molecules. Given a molecule and a set of atoms/bonds/rings to
color, produces highlighted SVG/PNG/console output, or plain format conversion (SDF, Mol, RXN,
CDX, CDXML, SMILES, InChI, InChIKey, SVG, EPS, PNG). Backend-agnostic: `RDKitMolecule` (built in,
uses RDKit for everything) implements the `HighlightBackendMolecule` ABC; other backends can
implement the same ABC against a different underlying engine, so code written against
`HighlightBackendMolecule` isn't tied to RDKit specifically.

`Document` (also in `chem_highlighter/hml.py`) wraps a *list* of molecule-backend instances
parsed from a single input, for formats that can legitimately hold more than one structure: SDF
(every record, not just the first), RXN (reactants/agents/products), CDXML (any number of
fragments), and dot-separated SMILES. `HighlightBackendMolecule.from_bytes` keeps its existing
single-molecule restrictions unchanged (e.g. still rejects multi-fragment CDXML); `Document`
is the opt-in multi-molecule path built on top of it.

## Install

```bash
pip install -e .
# or: uv sync
```

Requires Python >= 3.10. RDKit, matplotlib, numpy, polars, scipy, msgspec are pulled in as
dependencies.

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
- `utils` — shared helpers: conformer comparison, high-precision V3000 export, PNG render
  options, color/console formatting.

## Environment variables

PNG rendering (`RDKitMolecule.export`/`to_png`, `utils.get_png_render_options`) reads:

- `CHEM_HIGHLIGHTER_PNG_WIDTH`, `CHEM_HIGHLIGHTER_PNG_HEIGHT` — bounding box in pixels; the
  molecule is scaled to fit and centered. If only one is set, the other mirrors it. Unset: keeps
  RDKit's own default (unbounded) canvas sizing.
- `CHEM_HIGHLIGHTER_PNG_TRANSPARENT` — `true` for a transparent background instead of white.

Named to match other backend implementations' own PNG env vars, so a caller using more than one
backend configures PNG output once.

## Testing

```bash
pytest
mypy .
ruff check .
```

Coverage is configured for 100% (`pyproject.toml`'s `[tool.coverage.report]`, `fail_under = 100`).
`vulture` is configured to flag dead code (`[tool.vulture]`).

Some tests (`tests/test_rdkit.py`'s image/PNG-size tests, `tests/backend_test.py`'s shared
assertions) rely on RDKit's own rendering; no native/OS-specific setup needed beyond the pip
install above.
