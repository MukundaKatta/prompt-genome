"""Tests for mutate / crossover."""

from __future__ import annotations

import random

import pytest

from prompt_genome import Genome, Segment, crossover, mutate


def _seed_genome() -> Genome:
    return Genome(
        segments=[
            Segment("system", "be helpful"),
            Segment("instruction", "summarize"),
            Segment("constraints", "no opinions"),
            Segment("format", "bullet points"),
            Segment("freeform", "go"),
        ]
    )


def _pool() -> list[Segment]:
    # Provide same-kind alternates so replace_text has work to do, plus
    # a couple of cross-kind options for swap_with_pool / insert_from_pool.
    return [
        Segment("system", "be terse"),
        Segment("system", "be precise"),
        Segment("instruction", "rewrite"),
        Segment("instruction", "translate"),
        Segment("constraints", "no slang"),
        Segment("format", "json"),
        Segment("freeform", "stop"),
        Segment("examples", "ex1"),
    ]


def test_mutate_is_deterministic_for_same_seed():
    pool = _pool()
    a = mutate(_seed_genome(), rng=random.Random(42), segment_pool=pool, rate=0.5)
    b = mutate(_seed_genome(), rng=random.Random(42), segment_pool=pool, rate=0.5)
    assert a.segments == b.segments


def test_mutate_does_not_mutate_input():
    g = _seed_genome()
    snapshot = list(g.segments)
    _ = mutate(g, rng=random.Random(0), segment_pool=_pool(), rate=1.0)
    assert g.segments == snapshot


def test_mutate_returns_new_object():
    g = _seed_genome()
    out = mutate(g, rng=random.Random(0), segment_pool=_pool(), rate=0.0)
    # rate=0 → no per-segment mutation, but we still expect a fresh Genome.
    assert out is not g
    assert out.segments is not g.segments
    assert out.segments == g.segments


def test_mutate_rejects_bad_rate():
    with pytest.raises(ValueError):
        mutate(_seed_genome(), rng=random.Random(0), segment_pool=_pool(), rate=1.5)


def test_crossover_uniform_preserves_segment_counts_when_equal_length():
    a = _seed_genome()
    b = Genome(
        segments=[
            Segment("system", "x1"),
            Segment("instruction", "x2"),
            Segment("constraints", "x3"),
            Segment("format", "x4"),
            Segment("freeform", "x5"),
        ]
    )
    c1, c2 = crossover(a, b, rng=random.Random(7), mode="uniform")
    assert len(c1.segments) == len(a.segments) == len(b.segments)
    assert len(c2.segments) == len(a.segments)
    # Children are the union of parent segments at each position.
    combined_in = set(a.segments) | set(b.segments)
    combined_out = set(c1.segments) | set(c2.segments)
    assert combined_out.issubset(combined_in)


def test_crossover_single_point_splits_at_a_cut_index():
    a = _seed_genome()
    b = Genome(
        segments=[
            Segment("system", "x1"),
            Segment("instruction", "x2"),
            Segment("constraints", "x3"),
            Segment("format", "x4"),
            Segment("freeform", "x5"),
        ]
    )
    c1, c2 = crossover(a, b, rng=random.Random(3), mode="single_point")
    n = len(a.segments)
    # There must exist a cut k such that c1 = a[:k] + b[k:] and c2 = b[:k] + a[k:].
    cuts = []
    for k in range(0, n + 1):
        if c1.segments == a.segments[:k] + b.segments[k:] and c2.segments == b.segments[:k] + a.segments[k:]:
            cuts.append(k)
    assert cuts, f"no valid cut index reproduces children\nc1={c1.segments}\nc2={c2.segments}"


def test_crossover_rejects_unknown_mode():
    with pytest.raises(ValueError):
        crossover(_seed_genome(), _seed_genome(), rng=random.Random(0), mode="bogus")
