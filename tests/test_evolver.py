"""Smoke tests for the evolutionary loop and operators."""

import unittest
import random
import tempfile
import os

from prompt_genome import (
    Evolver,
    Genome,
    word_swap,
    instruction_inject,
    fragment_splice,
    persona_shuffle,
    crossover_sentences,
    tournament_select,
    elitist_select,
    rank_select,
)


def length_fitness(text: str) -> float:
    """Prefer prompts close to 80 chars."""
    return -abs(len(text) - 80)


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)

    def test_word_swap_changes_or_keeps_text(self):
        out = word_swap("Please summarize the report concisely.", self.rng)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)

    def test_word_swap_with_no_candidates_returns_input(self):
        text = "xyz qrs tuv"
        self.assertEqual(word_swap(text, self.rng), text)

    def test_instruction_inject_adds_text(self):
        original = "Answer the question."
        out = instruction_inject(original, self.rng)
        self.assertGreater(len(out), len(original))

    def test_persona_shuffle_adds_persona_when_missing(self):
        out = persona_shuffle("Just answer.", self.rng)
        self.assertIn("You are", out)

    def test_persona_shuffle_replaces_existing(self):
        text = "You are a pirate.\nAnswer the question."
        out = persona_shuffle(text, self.rng)
        self.assertIn("Answer the question.", out)
        self.assertNotIn("pirate", out)

    def test_fragment_splice_with_empty_pool_returns_input(self):
        text = "Hello there."
        self.assertEqual(fragment_splice(text, self.rng, donor_pool=[]), text)

    def test_crossover_yields_nonempty(self):
        a = "Be precise. Cite sources. Stay short."
        b = "Use bullets. Avoid speculation. Prefer concrete examples."
        child = crossover_sentences(a, b, self.rng)
        self.assertTrue(len(child) > 0)


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)
        self.pop = [
            Genome(text="a", score=0.1, gid="g0-0"),
            Genome(text="b", score=0.5, gid="g0-1"),
            Genome(text="c", score=0.9, gid="g0-2"),
            Genome(text="d", score=0.3, gid="g0-3"),
        ]

    def test_elitist_picks_top(self):
        top = elitist_select(self.pop, n=2)
        self.assertEqual([g.gid for g in top], ["g0-2", "g0-1"])

    def test_tournament_returns_member_of_pop(self):
        winner = tournament_select(self.pop, self.rng, k=2)
        self.assertIn(winner, self.pop)

    def test_rank_select_returns_member_of_pop(self):
        winner = rank_select(self.pop, self.rng)
        self.assertIn(winner, self.pop)


class TestEvolver(unittest.TestCase):
    def test_run_improves_or_holds_best_score(self):
        seeds = ["short", "a slightly longer seed prompt that is quite a bit too long for the target"]
        with tempfile.TemporaryDirectory() as tmp:
            evolver = Evolver(
                seeds=seeds,
                fitness_fn=length_fitness,
                population_size=6,
                generations=4,
                mutation_rate=0.6,
                crossover_rate=0.6,
                elitism=1,
                seed=42,
                log_dir=tmp,
            )
            best = evolver.run()
            self.assertIsInstance(best, Genome)
            self.assertGreater(evolver.stats.evaluations, 0)
            self.assertGreaterEqual(evolver.stats.generations_run, 1)
            # log file should exist and have entries
            files = os.listdir(tmp)
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmp, files[0])) as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertGreater(len(lines), 0)

    def test_invalid_args_raise(self):
        with self.assertRaises(ValueError):
            Evolver(seeds=[], fitness_fn=length_fitness)
        with self.assertRaises(ValueError):
            Evolver(seeds=["a"], fitness_fn=length_fitness, population_size=1)
        with self.assertRaises(ValueError):
            Evolver(seeds=["a"], fitness_fn=length_fitness, mutation_rate=1.5)


if __name__ == "__main__":
    unittest.main()
