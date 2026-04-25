"""Selection strategies for picking parents and survivors."""

from __future__ import annotations

import random
from typing import List

from .genome import Genome


def tournament_select(pop: List[Genome], rng: random.Random, k: int = 3) -> Genome:
    """Tournament selection: pick k random genomes, return the highest scoring."""
    if not pop:
        raise ValueError("tournament_select called with empty population")
    contenders = rng.sample(pop, k=min(k, len(pop)))
    return max(contenders, key=lambda g: g.score)


def elitist_select(pop: List[Genome], n: int) -> List[Genome]:
    """Return the top-n genomes by score (descending)."""
    return sorted(pop, key=lambda g: g.score, reverse=True)[:n]


def rank_select(pop: List[Genome], rng: random.Random) -> Genome:
    """Linear rank selection: probability proportional to rank, not raw score.

    Less greedy than fitness-proportional roulette; helps maintain diversity
    when one or two genomes dominate the early generations.
    """
    if not pop:
        raise ValueError("rank_select called with empty population")
    ranked = sorted(pop, key=lambda g: g.score)  # worst first, best last
    n = len(ranked)
    weights = list(range(1, n + 1))  # 1..n
    total = sum(weights)
    pick = rng.uniform(0, total)
    upto = 0
    for genome, w in zip(ranked, weights):
        upto += w
        if upto >= pick:
            return genome
    return ranked[-1]
