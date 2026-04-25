"""Evolve a summarization prompt against a tiny dev set.

Plug in your favorite scoring function (ROUGE, embedding similarity,
judge-LLM rating, etc.) where indicated.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_genome import Evolver


SEEDS = [
    "Summarize the passage in 2-3 short sentences.",
    "Produce a concise summary of the passage.",
    "Capture the key ideas of the passage in a brief summary.",
]


# Toy dev set: (input passage, reference summary).
DEV_SET = [
    (
        "The new vector database release adds quantization, multi-tenant "
        "isolation, and faster cold starts for serverless deployments.",
        "vector db update: quantization, multi-tenant, faster cold starts",
    ),
    (
        "Researchers introduced a small distilled model that matches a much "
        "larger model on math benchmarks while running on a single GPU.",
        "distilled small model matches larger one on math, single-GPU",
    ),
]


def keyword_overlap_fitness(prompt: str) -> float:
    """Toy fitness: pretend our 'LLM' just emits the prompt.

    Replace with a real LLM call + scorer in production.
    """
    score = 0.0
    for passage, ref in DEV_SET:
        # Extremely simple stand-in: count overlapping content words between
        # the prompt-augmented passage and the reference summary.
        ref_words = set(w.lower() for w in ref.split())
        prompt_words = set(w.lower() for w in prompt.split())
        passage_words = set(w.lower() for w in passage.split())
        produced = passage_words | prompt_words
        overlap = len(produced & ref_words) / max(1, len(ref_words))
        score += overlap
    return score / len(DEV_SET)


if __name__ == "__main__":
    evolver = Evolver(
        seeds=SEEDS,
        fitness_fn=keyword_overlap_fitness,
        population_size=8,
        generations=5,
        mutation_rate=0.4,
        seed=11,
    )
    best = evolver.run()
    print(f"best score: {best.score:.4f}")
    print(f"best prompt:\n{best.text}")
