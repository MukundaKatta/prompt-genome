"""The evolutionary loop: Optimizer and Result."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .genome import Genome, Segment
from .operators import crossover, mutate
from .selection import elitist, tournament_select


EvalFn = Callable[[str], float]


@dataclass
class Result:
    """Outcome of an `Optimizer.evolve` run."""

    history: list[float] = field(default_factory=list)
    best: Genome = field(default_factory=Genome)
    best_score: float = float("-inf")


class Optimizer:
    """Evolve a population of prompt genomes against an eval function.

    Determinism: every random operation flows through a single
    `random.Random(seed)`; given the same seed, eval_fn, pool, and initial
    population, two runs produce identical Results.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        segment_pool: list[Segment],
        *,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        elitism: int = 2,
        tournament_size: int = 3,
        seed: int = 0,
    ) -> None:
        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}")
        if elitism < 0 or elitism > population_size:
            raise ValueError(
                f"elitism must be in [0, population_size]; got {elitism} with population_size={population_size}"
            )
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
        if tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")

        self.eval_fn = eval_fn
        self.segment_pool = list(segment_pool)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.seed = seed

    def _score(self, pop: list[Genome]) -> list[tuple[Genome, float]]:
        return [(g, float(self.eval_fn(g.render()))) for g in pop]

    def _normalize_population(self, pop: list[Genome]) -> list[Genome]:
        """Pad/truncate the seed population to match `population_size`.

        - If smaller, we cycle through the provided genomes (stable, no rng).
        - If larger, we keep the first `population_size`.
        Inputs are cloned so callers' lists are never mutated downstream.
        """
        if not pop:
            raise ValueError("initial_population must contain at least one genome")
        if len(pop) >= self.population_size:
            return [g.clone() for g in pop[: self.population_size]]
        out = [g.clone() for g in pop]
        i = 0
        while len(out) < self.population_size:
            out.append(pop[i % len(pop)].clone())
            i += 1
        return out

    def evolve(self, initial_population: list[Genome], generations: int) -> Result:
        """Run `generations` rounds of selection + crossover + mutation.

        Elitism: the top `self.elitism` parents survive each generation
        unchanged, which guarantees best_score is monotonically non-decreasing.
        """
        if generations < 0:
            raise ValueError(f"generations must be >= 0, got {generations}")

        rng = random.Random(self.seed)
        population = self._normalize_population(initial_population)

        scored = self._score(population)
        best_genome, best_score = max(scored, key=lambda gs: gs[1])
        # `clone` so later in-place edits to `best_genome` (none today, but
        # cheap insurance) can't poison the recorded result.
        best_genome = best_genome.clone()

        history: list[float] = [best_score]

        for _ in range(generations):
            # 1. Elitism: carry the strongest parents forward unchanged.
            survivors = elitist(scored, self.elitism)

            # 2. Fill the rest via tournament-selected parents + crossover + mutation.
            n_offspring = self.population_size - len(survivors)
            offspring: list[Genome] = []
            while len(offspring) < n_offspring:
                parents = tournament_select(
                    scored, k=2, tournament_size=self.tournament_size, rng=rng
                )
                c1, c2 = crossover(parents[0], parents[1], rng=rng, mode="single_point")
                c1 = mutate(c1, rng=rng, segment_pool=self.segment_pool, rate=self.mutation_rate)
                c2 = mutate(c2, rng=rng, segment_pool=self.segment_pool, rate=self.mutation_rate)
                offspring.append(c1)
                if len(offspring) < n_offspring:
                    offspring.append(c2)

            population = survivors + offspring
            scored = self._score(population)

            gen_best, gen_best_score = max(scored, key=lambda gs: gs[1])
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_genome = gen_best.clone()
            history.append(best_score)

        return Result(history=history, best=best_genome, best_score=best_score)


__all__ = ["Optimizer", "Result"]
