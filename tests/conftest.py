"""Configuration and fixtures for tests."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from chem_highlighter.utils import mol_from_smiles

FIXTURES = Path(__file__).parent / "fixtures"


def from_fixture_molblock(fixture_file: str) -> Chem.Mol:
    return Chem.MolFromMolBlock((FIXTURES / fixture_file).read_text(), removeHs=False)


def assert_mols_equal(result: Chem.Mol | str, expected: Chem.Mol | str) -> None:
    if isinstance(result, str):
        result = mol_from_smiles(result)
    if isinstance(expected, str):
        expected = mol_from_smiles(expected)
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(expected), (
        f"{Chem.MolToSmiles(result)} != {Chem.MolToSmiles(expected)}"
    )


def visualize_conformers(a: Chem.Mol, b: Chem.Mol) -> None:
    import tempfile
    import webbrowser
    from pathlib import Path

    from chem_highlighter import RDKitDocument

    svg_a = RDKitDocument.from_mol(a).to_svg()
    svg_b = RDKitDocument.from_mol(b).to_svg()

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
                    <h2>Conformer A</h2>
                    {svg_a}
                </article>

                <article class="mol">
                    <h2>Conformer B</h2>
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
