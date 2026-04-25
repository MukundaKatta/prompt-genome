"""End-to-end demo with a mocked fitness function.

Run:
    python examples/cli_demo.py
Then:
    python -m prompt_genome.cli inspect runs/<latest>.jsonl --top 5
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_genome import Evolver


SEEDS = [
    "Summarize the article in 3 bullet points.",
    "Read the article and produce a concise 3-point summary.",
    "Give me a tight 3-bullet summary of the following article.",
    "Extract the main ideas from the article as 3 short bullets.",
]

# Mock fitness: rewards prompts that mention "bullet" and stay short.
# Replace with a real eval (judge LLM, ROUGE, etc.) in your project.
def fitness(text: str) -> float:
    score = 0.0
    if "bullet" in text.lower():
        score += 0.5
    if "concise" in text.lower() or "tight" in text.lower() or "compact" in text.lower():
        score += 0.2
    if "cite" in text.lower():
        score += 0.1
    # length penalty after 220 chars
    over = max(0, len(text) - 220)
    score -= 0.001 * over
    return score


if __name__ == "__main__":
    evolver = Evolver(
        seeds=SEEDS,
        fitness_fn=fitness,
        population_size=10,
        generations=6,
        mutation_rate=0.5,
        crossover_rate=0.6,
        elitism=2,
        seed=7,
    )
    best = evolver.run()
    print("== best ==")
    print(f"score: {best.score:.4f}")
    print(f"id   : {best.gid}")
    print("text :")
    print(best.text)
    print()
    print("stats:", evolver.stats)
