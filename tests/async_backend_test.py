"""Backend-independent asynchronous test functions mirroring `backend_test.py`."""

from __future__ import annotations  # noqa: I001,RUF100

from typing import TYPE_CHECKING, Literal

import msgspec
import pytest
from utils import extract_bond_codes, from_fixture_molblock, mol_from_explicit_smiles, read_fixture

from chem_highlighter.hml import (
    HML,
    AsyncHighlightBackendMoleculeT_co,
    InputFormat,
    InputFormatNotSupported,
    OutputFormat,
    OutputFormatNotSupported,
)
from chem_highlighter.utils import is_same_conformer, mol_to_smiles

if TYPE_CHECKING:
    from collections.abc import Sequence


async def create_doc(
    smiles: str, backend: type[AsyncHighlightBackendMoleculeT_co]
) -> AsyncHighlightBackendMoleculeT_co:
    return await backend.from_string(smiles, "SMILES")


async def assert_doc_init(backend: type[AsyncHighlightBackendMoleculeT_co]) -> None:
    doc = await create_doc("c1ccccc1", backend)
    assert await doc.export_string("SMILES") == "C1=CC=CC=C1"  # kekulize on default
    assert doc.get_hml_json() is None  # non-highlighted
    assert doc.get_edit_state() == (True, False, False)  # set correct state


async def assert_export_mdl(
    fmt: Literal["SDF", "Mol"], use_v2000: bool, backend: type[AsyncHighlightBackendMoleculeT_co]
) -> None:
    doc = await create_doc("c1ccccc1", backend)
    expected = "C1=CC=CC=C1"
    output = await doc.export_string(fmt, use_v2000)
    if use_v2000:
        assert "V2000" in output
    else:
        assert "V3000" in output
    endswith_dollars = output.strip().endswith("$$$$")
    assert endswith_dollars == (fmt == "SDF")
    roundtripped = await backend.from_string(output, fmt)
    assert await roundtripped.export_string("SMILES") == expected


async def assert_export(
    fmt: Literal["SMILES", "InChI", "InChIKey"],
    backend: type[AsyncHighlightBackendMoleculeT_co],
) -> None:
    output = await (await create_doc("c1ccccc1", backend)).export(fmt)
    if fmt == "SMILES":
        assert output == b"C1=CC=CC=C1"
    elif fmt == "InChI":
        assert output == b"InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    elif fmt == "InChIKey":
        assert output == b"UHOVQNZJYSORNB-UHFFFAOYSA-N"
    else:  # pragma: no cover
        raise ValueError("Unexpected format")


async def assert_from_bytes(
    fmt: Literal["SDF", "Mol", "SMILES", "InChI"],
    backend: type[AsyncHighlightBackendMoleculeT_co],
) -> None:
    exported = await (await backend.from_string("c1ccccc1", "SMILES")).export(fmt)
    doc = await backend.from_bytes(exported, fmt)

    assert await doc.export_string("SMILES") == "C1=CC=CC=C1"

    with pytest.raises((ValueError, RuntimeError), match=f"Invalid {fmt} input"):
        await backend.from_bytes(b"", fmt)

    with pytest.raises((ValueError, RuntimeError), match=f"Invalid {fmt} input"):
        await backend.from_bytes(b"invalid input", fmt)


async def assert_from_bytes_unsupported_format(
    fmt: InputFormat,
    backend: type[AsyncHighlightBackendMoleculeT_co],
    sync_backend_name: str,
) -> None:
    # error message references the synchronous backend rather than the async wrapper.
    with pytest.raises(
        InputFormatNotSupported, match=f"{sync_backend_name} does not support importing from {fmt}"
    ):
        await backend.from_bytes(b"", fmt)


async def assert_export_unsupported_format(
    fmt: OutputFormat,
    backend: type[AsyncHighlightBackendMoleculeT_co],
    sync_backend_name: str,
) -> None:
    with pytest.raises(
        OutputFormatNotSupported, match=f"{sync_backend_name} does not support exporting to {fmt}"
    ):
        await (await create_doc("c1ccccc1", backend)).export(fmt)


async def assert_align_to_reference(
    r_file: str,
    q_file_suffixes: Sequence[str],
    atol: float,
    backend: type[AsyncHighlightBackendMoleculeT_co],
) -> None:
    r = from_fixture_molblock(r_file)
    for q_file_suffix in q_file_suffixes:
        q = from_fixture_molblock(r_file.removesuffix(".mol") + f"{q_file_suffix}.mol")
        for doc_mol, ref_mol in ((q, r), (r, q)):
            doc = await backend.from_mol(doc_mol)
            ref = await backend.from_mol(ref_mol)
            await doc.align_to_reference(await ref.to_molblock())
            assert is_same_conformer(await doc.to_molblock(), await ref.to_molblock(), atol=atol)


async def assert_cleanup(
    expected: str, atol: float, backend: type[AsyncHighlightBackendMoleculeT_co]
) -> None:
    doc = await backend.from_molblock(read_fixture("pyrimidine_dirty.mol"))
    assert not is_same_conformer(await doc.to_molblock(), expected, atol=atol)

    await doc.cleanup()

    assert is_same_conformer(await doc.to_molblock(), expected, atol=atol)


async def assert_kekulize(backend: type[AsyncHighlightBackendMoleculeT_co]) -> None:
    async def assert_benzene_kekulized(
        doc: AsyncHighlightBackendMoleculeT_co, kekulized: bool
    ) -> None:
        assert doc.get_edit_state() == (kekulized, False, False)
        bond_type_codes = extract_bond_codes(await doc.to_molblock())
        kekule_bonds = [[1 if i % 2 == modulo else 2 for i in range(6)] for modulo in (0, 1)]
        if kekulized:
            assert bond_type_codes in kekule_bonds, (
                "exported Mol block still has aromatic bond codes"
            )
        else:
            assert bond_type_codes == [4] * 6, (
                "exported Mol block should use the aromatic bond code"
            )

    doc = await create_doc("c1ccccc1", backend)
    await assert_benzene_kekulized(doc, True)
    await doc.kekulize(False)
    await assert_benzene_kekulized(doc, False)
    await doc.kekulize(True)
    await assert_benzene_kekulized(doc, True)


async def assert_hide_hydrogens_table(
    base: str, hide: str, backend: type[AsyncHighlightBackendMoleculeT_co]
) -> None:
    doc = await create_doc(base, backend)
    await doc.hide_hydrogens()
    await doc.kekulize(False)
    output = await doc.export_string("SMILES")
    canonical_output = mol_to_smiles(mol_from_explicit_smiles(output))
    assert canonical_output == hide


async def assert_highlight_from_json(
    hml_json: str, backend: type[AsyncHighlightBackendMoleculeT_co]
) -> None:
    doc = await create_doc("[H]CCO[2H]", backend)
    before = await doc.to_svg()

    await doc.highlight_from_json(hml_json, False)
    svg = await doc.to_svg()

    assert "<svg" in svg
    assert svg != before, "highlighting should change the rendered SVG"
    assert "ff0000" in svg.lower(), "the highlight color should actually appear in the SVG"

    doc = await create_doc("c1ccccc1", backend)
    hml = HML(
        highlighted_atoms={0: 0},
        highlighted_rings={0: 1},
        rings=[[0, 1, 2, 3, 4, 5]],
        palette=["#ff0000", "#00ff00"],
    )
    await doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    ring_svg = await doc.to_svg()
    assert "<svg" in ring_svg
    assert "ff0000" in ring_svg.lower(), "the highlight color should actually appear in the SVG"
    assert "00ff00" in ring_svg.lower(), (
        "the ring highlight color should actually appear in the SVG"
    )


async def assert_add_label(backend: type[AsyncHighlightBackendMoleculeT_co]) -> None:
    doc = await create_doc("c1ccccc1O", backend)
    before = await doc.to_svg()
    assert doc.get_label() is None

    await doc.add_label("Compound 1")

    assert doc.get_label() == "Compound 1"
    svg = await doc.to_svg()
    assert "<svg" in svg
    assert svg != before, "adding a label should change the rendered SVG"

    with pytest.raises(ValueError, match="Already labeled"):
        await doc.add_label("Compound 2")


async def assert_to_console(backend: type[AsyncHighlightBackendMoleculeT_co]) -> None:
    doc = await backend.from_string("C=COCc1ccc(C)cc1", "SMILES")
    await doc.kekulize(False)
    hml = HML(
        highlighted_atoms={8: 0},
        highlighted_bonds={0: 1},
        palette=["#ff0000", "#00ff00"],
    )
    await doc.highlight_from_json(msgspec.json.encode(hml).decode(), False)
    console = await doc.to_console()
    assert console == "C\033[38;2;0;255;0m=\033[0mCOCc1ccc\033[38;2;255;0;0m(C)\033[0mcc1"
