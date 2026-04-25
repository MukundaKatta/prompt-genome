"""Top level evolutionary loop.

Usage:

    evolver = Evolver(seeds=[...], fitness_fn=score, population_size=12)
    best = evolver.run()
    print(best.text, best.score)
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import operators as ops
from .genome import Genome
from .selection import elitist_select, tournament_select


FitnessFn = Callable[[str], float]


@dataclass
class EvolverStats:
    generations_run: int = 0
    evaluations: int = 0
    best_score: float = float("-inf")
    wall_seconds: float = 0.0


class Evolver:
    def __init__(
        self,
        seeds: List[str],
        fitness_fn: FitnessFn,
        population_size: int = 10,
        generations: int = 6,
        mutation_rate: float = 0.4,
        crossover_rate: float = 0.6,
        elitism: int = 1,
        tournament_k: int = 3,
        plateau_patience: Optional[int] = 3,
        plateau_eps: float = 1e-4,
        seed: Optional[int] = None,
        log_dir: Optional[str] = "runs",
    ) -> None:
        if not seeds:
            raise ValueError("Evolver needs at least one seed prompt")
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if elitism < 0 or elitism >= population_size:
            raise ValueError("elitism must be in [0, population_size)")

        self.seeds = list(seeds)
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.plateau_patience = plateau_patience
        self.plateau_eps = plateau_eps
        self.rng = random.Random(seed)
        self.log_dir = log_dir
        self.stats = EvolverStats()
        self._log_path: Optional[str] = None

    # --- public API ---------------------------------------------------------

    def run(self) -> Genome:
        """Run the evolutionary loop and return the best-scoring genome."""
        start = time.time()
        self._open_log()
        population = self._initial_population()
        self._score(population)
        self._log_generation(0, population)

        best_history: List[float] = [max(g.score for g in population)]

        for gen in range(1, self.generations + 1):
            children = self._breed(population, gen)
            self._score(children)
            population = self._next_generation(population, children)
            best = max(population, key=lambda g: g.score)
            best_history.append(best.score)
            self._log_generation(gen, population)
            self.stats.generations_run = gen
            self.stats.best_score = best.score
            if self._plateaued(best_history):
                break

        self.stats.wall_seconds = time.time() - start
        return max(population, key=lambda g: g.score)

    # --- internals ----------------------------------------------------------

    def _initial_population(self) -> List[Genome]:
        pop: List[Genome] = []
        for i in range(self.population_size):
            seed_text = self.seeds[i % len(self.seeds)]
            pop.append(Genome.seed(seed_text, slot=i))
        # mild perturbation so we don't have N copies of the same seed
        for i, g in enumerate(pop[len(self.seeds):], start=len(self.seeds)):
            g.text = ops.word_swap(g.text, self.rng)
            g.ops = ["seed", "word_swap"]
        return pop

    def _score(self, genomes: List[Genome]) -> None:
        for g in genomes:
            if g.score == float("-inf"):
                g.score = float(self.fitness_fn(g.text))
                self.stats.evaluations += 1

    def _breed(self, parents_pool: List[Genome], gen: int) -> List[Genome]:
        children: List[Genome] = []
        slot = 0
        donor_pool = [g.text for g in parents_pool]
        # how many we need; elitism slots come from the survivor step
        target = self.population_size - self.elitism
        while len(children) < target:
            mom = tournament_select(parents_pool, self.rng, k=self.tournament_k)
            dad = tournament_select(parents_pool, self.rng, k=self.tournament_k)
            applied: List[str] = []
            if self.rng.random() < self.crossover_rate and mom is not dad:
                child_text = ops.crossover_sentences(mom.text, dad.text, self.rng)
                applied.append("crossover_sentences")
                parents = [mom, dad]
            else:
                child_text = mom.text
                parents = [mom]

            if self.rng.random() < self.mutation_rate:
                op_choice = self.rng.choice(
                    ["word_swap", "instruction_inject", "fragment_splice", "persona_shuffle"]
                )
                if op_choice == "word_swap":
                    child_text = ops.word_swap(child_text, self.rng)
                elif op_choice == "instruction_inject":
                    child_text = ops.instruction_inject(child_text, self.rng)
                elif op_choice == "fragment_splice":
                    child_text = ops.fragment_splice(child_text, self.rng, donor_pool)
                else:
                    child_text = ops.persona_shuffle(child_text, self.rng)
                applied.append(op_choice)

            child = mom.child(child_text, gen=gen, slot=slot, parents=parents, ops=applied)
            children.append(child)
            slot += 1
        return children

    def _next_generation(self, current: List[Genome], children: List[Genome]) -> List[Genome]:
        elites = elitist_select(current, self.elitism) if self.elitism else []
        return elites + children

    def _plateaued(self, history: List[float]) -> bool:
        if not self.plateau_patience or len(history) <= self.plateau_patience:
            return False
        recent = history[-(self.plateau_patience + 1):]
        return (max(recent) - min(recent)) < self.plateau_eps

    def _open_log(self) -> None:
        if not self.log_dir:
            self._log_path = None
            return
        os.makedirs(self.log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = os.path.join(self.log_dir, f"{ts}.jsonl")

    def _log_generation(self, gen: int, pop: List[Genome]) -> None:
        if not self._log_path:
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            for g in pop:
                f.write(json.dumps(g.to_log()) + "\n")
