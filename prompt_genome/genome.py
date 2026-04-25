"""Genome dataclass: a single prompt candidate plus its lineage and score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Genome:
    """A single candidate prompt in the population.

    Attributes:
        text: The prompt string itself.
        gen: Generation number this genome was created in (0 = seed).
        gid: A short unique id, e.g. "g3-7" (generation 3, slot 7).
        score: Fitness score assigned by the user-supplied fitness function.
        parents: Ids of parent genomes (empty for seeds).
        ops: Names of mutation/crossover operators applied to produce this genome.
    """

    text: str
    gen: int = 0
    gid: str = ""
    score: float = float("-inf")
    parents: List[str] = field(default_factory=list)
    ops: List[str] = field(default_factory=list)

    def to_log(self) -> dict:
        """Serializable form used for the per-generation JSONL log."""
        return {
            "gen": self.gen,
            "id": self.gid,
            "score": None if self.score == float("-inf") else self.score,
            "parents": list(self.parents),
            "ops": list(self.ops),
            "text": self.text,
        }

    @classmethod
    def seed(cls, text: str, slot: int) -> "Genome":
        """Build a seed genome (generation 0)."""
        return cls(text=text, gen=0, gid=f"g0-{slot}", parents=[], ops=["seed"])

    def child(self, text: str, gen: int, slot: int, parents: List["Genome"], ops: List[str]) -> "Genome":
        """Create a child genome with proper lineage and a fresh id."""
        return Genome(
            text=text,
            gen=gen,
            gid=f"g{gen}-{slot}",
            score=float("-inf"),
            parents=[p.gid for p in parents],
            ops=list(ops),
        )
