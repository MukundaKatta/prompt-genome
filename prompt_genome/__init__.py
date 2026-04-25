"""prompt-genome: a small evolutionary optimizer for LLM prompts."""

from .genome import Genome
from .evolver import Evolver
from .operators import (
    word_swap,
    instruction_inject,
    fragment_splice,
    persona_shuffle,
    crossover_sentences,
)
from .selection import tournament_select, elitist_select, rank_select

__version__ = "0.1.0"

__all__ = [
    "Genome",
    "Evolver",
    "word_swap",
    "instruction_inject",
    "fragment_splice",
    "persona_shuffle",
    "crossover_sentences",
    "tournament_select",
    "elitist_select",
    "rank_select",
    "__version__",
]
