"""Evolve a classification prompt with a label-match fitness.

Pretend the model's response is the lowercase label from a small set; we
score how often our prompt nudges it toward the right label by checking
which label tokens the prompt itself emphasizes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_genome import Evolver


SEEDS = [
    "Classify the message as positive, neutral, or negative.",
    "Decide whether the message is positive, neutral, or negative.",
    "Read the message and return one label: positive, neutral, or negative.",
]

LABELS = ["positive", "neutral", "negative"]


def fitness(prompt: str) -> float:
    """Reward prompts that mention all 3 labels and ask for a single choice."""
    text = prompt.lower()
    label_hits = sum(1 for lab in LABELS if lab in text)
    label_score = label_hits / len(LABELS)
    structure_bonus = 0.2 if ("one" in text or "single" in text or "exactly" in text) else 0.0
    length_pen = 0.001 * max(0, len(prompt) - 200)
    return label_score + structure_bonus - length_pen


if __name__ == "__main__":
    evolver = Evolver(
        seeds=SEEDS,
        fitness_fn=fitness,
        population_size=10,
        generations=6,
        mutation_rate=0.45,
        seed=23,
    )
    best = evolver.run()
    print(f"best score: {best.score:.4f}")
    print(f"best prompt:\n{best.text}")
