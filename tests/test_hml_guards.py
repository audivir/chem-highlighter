"""Tests repeated call guards for chem_highlighter.hml."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chem_highlighter.align import get_alignment_ops_from_molblock

if TYPE_CHECKING:
    from chem_highlighter.hml import HighlightBackendDocument


def test_align_to_reference_raises_when_already_aligned(doc: HighlightBackendDocument) -> None:
    doc.align_to_reference(doc.to_molblock())
    with pytest.raises(ValueError, match="Already aligned"):
        doc.align_to_reference(doc.to_molblock())


def test_hide_hydrogens_raises_when_already_set(doc: HighlightBackendDocument) -> None:
    doc.hide_hydrogens()
    with pytest.raises(ValueError, match="Hydrogen display already set"):
        doc.hide_hydrogens()


def test_hide_hydrogens_raises_after_highlighting(
    doc: HighlightBackendDocument, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, False)
    with pytest.raises(
        ValueError, match="Setting hydrogen display after highlighting not supported"
    ):
        doc.hide_hydrogens()


def test_highlight_from_json_raises_when_already_highlighted(
    doc: HighlightBackendDocument, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, False)
    with pytest.raises(ValueError, match="Already highlighted"):
        doc.highlight_from_json(hml_json, False)


def test_highlight_from_json_raises_after_hydrogens_hidden(
    doc: HighlightBackendDocument, hml_json: str
) -> None:
    doc.hide_hydrogens()

    with pytest.raises(
        ValueError, match="Highlighting after setting hydrogen display not supported"
    ):
        doc.highlight_from_json(hml_json, False)


def test_hide_hydrogens_raises_after_highlight_from_json_with_hide_hydrogens(
    doc: HighlightBackendDocument, hml_json: str
) -> None:
    doc.highlight_from_json(hml_json, False)
    with pytest.raises(
        ValueError, match="Setting hydrogen display after highlighting not supported"
    ):
        doc.hide_hydrogens()


def test_callback_methods_bypass_the_guard(doc: HighlightBackendDocument, hml_json: str) -> None:
    doc.cleanup_callback()
    doc.cleanup_callback()

    flips, global_flip, angle = get_alignment_ops_from_molblock(
        doc.to_molblock(), doc.to_molblock(), atol=1e-5
    )
    doc.align_to_reference_callback(flips, global_flip, angle)
    doc.align_to_reference_callback(flips, global_flip, angle)

    doc.hide_hydrogens_callback()
    doc.hide_hydrogens_callback()

    doc.highlight_from_json_callback(hml_json, False)
    doc.highlight_from_json_callback(hml_json, False)
