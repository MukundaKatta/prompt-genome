"""Tests for tournament_select and elitist."""

from __future__ import annotations

import random

import pytest

from prompt_genome import Genome, Segment, elitist, tournament_select


def _scored() -> list[tuple[Genome, float]]:
    out = []
    for i, txt in enumerate(["a", "b", "c", "d", "e"]):
        g = Genome(segments=[Segment("system", txt)])
        out.append((g, float(i)))  # score == index, so "e" is best
    return out


def test_tournament_returns_k_items():
    rng = random.Random(0)
    picks = tournament_select(_scored(), k=4, tournament_size=2, rng=rng)
    assert len(picks) == 4
    assert all(isinstance(p, Genome) for p in picks)


def test_tournament_is_deterministic_under_seeded_rng():
    a = tournament_select(_scored(), k=10, tournament_size=3, rng=random.Random(99))
    b = tournament_select(_scored(), k=10, tournament_size=3, rng=random.Random(99))
    # Same seed → identical sequence of picks (by identity-of-segments).
    assert [g.segments for g in a] == [g.segments for g in b]


def test_tournament_size_one_is_uniform_random_pick():
    rng = random.Random(0)
    picks = tournament_select(_scored(), k=50, tournament_size=1, rng=rng)
    # All 5 candidates should turn up at least once over 50 draws.
    seen = {p.segments[0].text for p in picks}
    assert seen == {"a", "b", "c", "d", "e"}


def test_tournament_rejects_bad_args():
    with pytest.raises(ValueError):
        tournament_select(_scored(), k=-1, tournament_size=2, rng=random.Random(0))
    with pytest.raises(ValueError):
        tournament_select(_scored(), k=2, tournament_size=0, rng=random.Random(0))


def test_elitist_returns_top_n_by_score():
    top = elitist(_scored(), top_n=2)
    # The best two are scored 4.0 ("e") and 3.0 ("d").
    assert [g.segments[0].text for g in top] == ["e", "d"]


def test_elitist_top_n_zero_returns_empty():
    assert elitist(_scored(), top_n=0) == []


def test_elitist_is_stable_on_ties():
    # Build two genomes with identical scores; first one in input must stay first.
    g_first = Genome(segments=[Segment("system", "first")])
    g_second = Genome(segments=[Segment("system", "second")])
    scored = [(g_first, 1.0), (g_second, 1.0)]
    out = elitist(scored, top_n=2)
    assert out[0] is g_first
    assert out[1] is g_second
