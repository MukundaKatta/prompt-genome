"""Selection strategies for the evolutionary loop."""

from __future__ import annotations

import random

from .genome import Genome


def tournament_select(
    scored: list[tuple[Genome, float]],
    k: int,
    *,
    tournament_size: int = 3,
    rng: random.Random,
) -> list[Genome]:
    """Pick `k` genomes by repeated tournament.

    For each pick we sample `tournament_size` competitors with replacement
    and return the highest-scoring among them. Replacement is used so a
    small population doesn't starve selection pressure.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not scored:
        if k == 0:
            return []
        raise ValueError("scored is empty; cannot run a tournament")
    if tournament_size < 1:
        raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")

    chosen: list[Genome] = []
    for _ in range(k):
        contenders = [rng.choice(scored) for _ in range(tournament_size)]
        # Highest score wins; rng.choice already broke any earlier ties
        # by sampling, so plain max() is fine and stable.
        winner = max(contenders, key=lambda gs: gs[1])
        chosen.append(winner[0])
    return chosen


def elitist(scored: list[tuple[Genome, float]], top_n: int) -> list[Genome]:
    """Return the `top_n` highest-scoring genomes (descending by score).

    Stable: when scores tie, original order is preserved.
    """
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    if top_n == 0:
        return []
    # `sorted` is stable; negate score to sort descending while preserving
    # input order on ties.
    ranked = sorted(scored, key=lambda gs: -gs[1])
    return [g for g, _ in ranked[:top_n]]


__all__ = ["tournament_select", "elitist"]
