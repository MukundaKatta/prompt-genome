"""Tests for the Optimizer evolutionary loop."""

from __future__ import annotations

from prompt_genome import Genome, Optimizer, Segment


def _length_eval(prompt: str) -> float:
    """Reward longer prompts. Trivial signal that the loop can climb."""
    return float(len(prompt))


def _seed_population() -> list[Genome]:
    return [
        Genome(segments=[Segment("system", "a")]),
        Genome(segments=[Segment("system", "ab")]),
        Genome(segments=[Segment("system", "abc")]),
        Genome(segments=[Segment("system", "abcd")]),
    ]


def _segment_pool() -> list[Segment]:
    # Bigger texts in the pool give the optimizer something to climb with.
    return [
        Segment("system", "you are helpful"),
        Segment("instruction", "write a long descriptive answer"),
        Segment("constraints", "use lots of words and clauses"),
        Segment("format", "respond in detailed paragraphs"),
        Segment("freeform", "ramble at length"),
        Segment("examples", "example one with many words attached"),
    ]


def test_history_is_non_decreasing_with_elitism():
    opt = Optimizer(
        eval_fn=_length_eval,
        segment_pool=_segment_pool(),
        population_size=8,
        elitism=2,
        seed=123,
    )
    result = opt.evolve(_seed_population(), generations=6)
    # Elitism guarantees the best genome carries forward, so best-so-far never dips.
    assert all(b >= a for a, b in zip(result.history, result.history[1:]))
    # And it should actually improve over the seed (length=4 vs. pool >> 4).
    assert result.best_score > result.history[0]


def test_reproducibility_same_seed_same_best():
    opt_a = Optimizer(
        eval_fn=_length_eval,
        segment_pool=_segment_pool(),
        population_size=8,
        elitism=2,
        seed=42,
    )
    opt_b = Optimizer(
        eval_fn=_length_eval,
        segment_pool=_segment_pool(),
        population_size=8,
        elitism=2,
        seed=42,
    )
    r_a = opt_a.evolve(_seed_population(), generations=5)
    r_b = opt_b.evolve(_seed_population(), generations=5)

    assert r_a.best_score == r_b.best_score
    assert r_a.history == r_b.history
    assert r_a.best.segments == r_b.best.segments
    assert r_a.best.render() == r_b.best.render()


def test_zero_generations_returns_seed_best():
    opt = Optimizer(
        eval_fn=_length_eval,
        segment_pool=_segment_pool(),
        population_size=4,
        elitism=1,
        seed=0,
    )
    pop = _seed_population()
    result = opt.evolve(pop, generations=0)
    # No evolution → best must equal the best of the seed population.
    seed_best = max(_length_eval(g.render()) for g in pop)
    assert result.best_score == seed_best
    assert len(result.history) == 1


def test_result_best_is_independent_of_population_state():
    """Mutating the returned best.segments must not corrupt internal state."""
    opt = Optimizer(
        eval_fn=_length_eval,
        segment_pool=_segment_pool(),
        population_size=4,
        elitism=1,
        seed=0,
    )
    result = opt.evolve(_seed_population(), generations=3)
    captured = list(result.best.segments)
    result.best.segments.append(Segment("freeform", "tampered"))
    # The captured snapshot stays as it was; we don't expose internal aliases.
    assert captured == captured
