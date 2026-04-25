"""Tests for Genome / Segment."""

from __future__ import annotations

import pytest

from prompt_genome import Genome, Segment
from prompt_genome.genome import KIND_ORDER


def test_segment_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Segment(kind="not-a-real-kind", text="oops")


def test_segment_is_hashable_on_kind_and_text():
    a = Segment("system", "hello")
    b = Segment("system", "hello")
    c = Segment("system", "world")
    # Same (kind, text) hashes equally; different text is distinct.
    assert hash(a) == hash(b)
    assert {a, b, c} == {a, c}


def test_render_uses_canonical_kind_order_regardless_of_input_order():
    # Insert in non-canonical order; render must reorder by KIND_ORDER.
    segs = [
        Segment("freeform", "free"),
        Segment("constraints", "no jokes"),
        Segment("system", "you are helpful"),
        Segment("instruction", "summarize the input"),
    ]
    g = Genome(segments=segs)
    rendered = g.render()
    # The position of each header in the output should follow KIND_ORDER.
    positions = []
    for kind in KIND_ORDER:
        header = f"[{kind}]"
        if header in rendered:
            positions.append(rendered.index(header))
    assert positions == sorted(positions)


def test_render_preserves_relative_order_within_same_kind():
    g = Genome(
        segments=[
            Segment("instruction", "first"),
            Segment("system", "sys"),
            Segment("instruction", "second"),
            Segment("instruction", "third"),
        ]
    )
    out = g.render()
    # All instructions stay in input order even though system comes first overall.
    assert out.index("first") < out.index("second") < out.index("third")
    assert out.index("sys") < out.index("first")


def test_to_dict_and_from_dict_roundtrip():
    g = Genome(
        segments=[
            Segment("system", "be terse"),
            Segment("instruction", "respond"),
        ],
        meta={"label": "v1"},
    )
    restored = Genome.from_dict(g.to_dict())
    assert restored.segments == g.segments
    assert restored.meta == g.meta
    assert restored.render() == g.render()
