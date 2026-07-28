"""Configuration and fixtures for tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem

if TYPE_CHECKING:
    from chem_highlighter.hml import HighlightBackendDocument


FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES_ORDER = [FIXTURES]


def from_fixture_molblock(fixture_file: str) -> Chem.Mol:
    return Chem.MolFromMolBlock(read_fixture(fixture_file), removeHs=False)


def read_fixture(fixture_file: str) -> str:
    for directory in FIXTURES_ORDER:
        path = directory / fixture_file
        if path.exists():
            return path.read_text()
    raise ValueError(f"Fixture {fixture_file} not found")


def mol_from_explicit_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string without collapsing its explicit hydrogen atoms to implicit ones."""
    params = Chem.SmilesParserParams()
    params.removeHs = False  # type: ignore[assignment]
    return Chem.MolFromSmiles(smiles, params)


def assert_mols_equal(result: Chem.Mol | str, expected: Chem.Mol | str) -> None:
    if isinstance(result, str):
        result = mol_from_explicit_smiles(result)
    if isinstance(expected, str):
        expected = mol_from_explicit_smiles(expected)
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(expected), (
        f"{Chem.MolToSmiles(result)} != {Chem.MolToSmiles(expected)}"
    )


def extract_bond_codes(molblock: str) -> list[int]:
    """Extract the bond type codes from a V3000 molblock."""
    codes = []
    in_bond_block = False
    for line in molblock.splitlines():
        stripped = line.strip()
        if stripped == "M  V30 BEGIN BOND":
            in_bond_block = True
        elif stripped == "M  V30 END BOND":
            in_bond_block = False
        elif in_bond_block:
            codes.append(int(line.split()[3]))
    return codes


def assert_benzene_kekulized(doc: HighlightBackendDocument, kekulized: bool) -> None:
    assert doc.get_edit_state() == (kekulized, False, False)

    bond_type_codes = extract_bond_codes(doc.to_molblock())
    kekule_bonds = [[1 if i % 2 == modulo else 2 for i in range(6)] for modulo in (0, 1)]

    if mol := getattr(doc, "mol", None):  # noqa: SIM102
        if isinstance(mol, Chem.Mol):
            ring_bonds = [mol.GetBondBetweenAtoms(i, (i + 1) % 6) for i in range(6)]
            assert all(bond.GetIsAromatic() for bond in ring_bonds)

    if kekulized:
        assert bond_type_codes in kekule_bonds, "exported Mol block still has aromatic bond codes"

    else:
        assert bond_type_codes == [4] * 6, "exported Mol block should use the aromatic bond code"


def visualize_conformers(a: Chem.Mol, b: Chem.Mol) -> None:
    import tempfile
    import webbrowser
    from pathlib import Path

    from chem_highlighter import RDKitDocument

    svg_a = RDKitDocument.from_mol(a).to_svg()
    svg_b = RDKitDocument.from_mol(b).to_svg()

    conf_a = a.GetConformer()
    conf_b = b.GetConformer()

    center_a = (conf_a.GetPositions().min(axis=0) + conf_a.GetPositions().min(axis=0)) / 2.0
    center_b = (conf_b.GetPositions().min(axis=0) + conf_b.GetPositions().min(axis=0)) / 2.0

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Conformer Comparison</title>

        <link
            rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"
        >

        <style>
            body {{
                max-width: 1100px;
                margin: 2rem auto;
                padding: 0 1rem;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 1.5rem;
            }}

            .mol {{
                padding: 1.5rem;
                border-radius: 1rem;
                background: var(--card-background-color);
                box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                overflow: hidden;
            }}

            .mol h2 {{
                text-align: center;
                margin-bottom: 1rem;
            }}

            .mol svg {{
                width: 100%;
                height: auto;
                display: block;
            }}

            /* RDKit uses black lines/text; invert in dark mode */
            @media (prefers-color-scheme: dark) {{
                .mol svg {{
                    filter: invert(1);
                }}
            }}
        </style>
    </head>

    <body>
        <main>
            <h1>Conformer Comparison</h1>

            <div class="grid">
                <article class="mol">
                    <h2>Conformer A ({center_a})</h2>
                    {svg_a}
                </article>

                <article class="mol">
                    <h2>Conformer B ({center_b})</h2>
                    {svg_b}
                </article>
            </div>
        </main>
    </body>
    </html>
    """

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        encoding="utf-8",
        delete=False,
    ) as f:
        f.write(html)
        path = Path(f.name)

    webbrowser.open(path.as_uri())

    breakpoint()  # noqa: T100

    path.unlink(missing_ok=True)
