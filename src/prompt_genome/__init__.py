"""prompt-genome: a tiny, dependency-free evolutionary optimizer for LLM prompts."""

from .genome import Genome, Segment
from .operators import mutate, crossover
from .selection import tournament_select, elitist
from .optimizer import Optimizer, Result

__version__ = "0.1.0"

__all__ = [
    "Genome",
    "Segment",
    "mutate",
    "crossover",
    "tournament_select",
    "elitist",
    "Optimizer",
    "Result",
    "__version__",
]
