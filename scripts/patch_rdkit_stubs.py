"""Fix a syntax error in the installed rdkit stub package before running mypy.

rdkit's auto-generated `rdkit-stubs/Chem/rdRGroupDecomposition.pyi` declares a class
attribute literally named `None`, which is invalid as an annotation target and makes mypy
abort with a syntax error before checking anything else. `uv sync` reinstalls the broken
file on every fresh environment, so this patches it in place, renaming the attribute to
`None_`. Idempotent: does nothing if already patched or the upstream file has moved on.
"""

from __future__ import annotations

import re
from pathlib import Path

import rdkit


def main() -> int:
    """Renames the invalid `None` class attribute in the installed rdkit stub, if present."""
    stub_path = (
        Path(rdkit.__file__).resolve().parent.parent
        / "rdkit-stubs"
        / "Chem"
        / "rdRGroupDecomposition.pyi"
    )
    if not stub_path.exists():
        return 0

    content = stub_path.read_text()
    patched = re.sub(r"^(\s*)None:", r"\1None_:", content, flags=re.MULTILINE)
    patched = patched.replace("'None',", "'None_',")
    if patched != content:
        stub_path.write_text(patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
