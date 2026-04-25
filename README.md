# prompt-genome

A tiny, dependency-light evolutionary optimizer for LLM prompts. Treat your prompts as a population of "genomes" that mutate, recombine, and survive based on eval scores. Inspired by the 2026 wave of self-evolving agent frameworks, but stripped down to the essentials so you can drop it into any pipeline in an afternoon.

## Why

Prompt engineering is mostly trial and error. `prompt-genome` automates the trial-and-error loop:

1. Start with a small pool of seed prompts.
2. Score each prompt against your eval cases using a function you supply.
3. Mutate, crossover, and keep the survivors.
4. Repeat until the score plateaus or you hit your generation cap.

You bring the eval function and the seeds. The library handles the search.

## Features

- Pure Python, zero required runtime dependencies.
- Pluggable mutation operators: word swap, instruction injection, fragment splice, persona shuffle.
- Crossover that respects sentence and section boundaries so children stay readable.
- Tournament, elitist, and rank selection strategies.
- Deterministic mode with seeded RNG for reproducible runs.
- JSONL run log so you can inspect every generation later.

## Installation

```bash
pip install prompt-genome
```

Or from source:

```bash
git clone https://github.com/MukundaKatta/prompt-genome.git
cd prompt-genome
pip install -e .
```

## Quick Start

```python
from prompt_genome import Evolver, Genome

seeds = [
    "Summarize the article in 3 bullet points.",
    "Read the article and produce a concise 3-point summary.",
    "Give me a tight 3-bullet summary of the following article.",
]

def fitness(prompt: str) -> float:
    # Plug in your eval here. Higher is better.
    # Could be: rouge score, judge LLM rating, citation coverage, etc.
    return run_my_eval(prompt)

evolver = Evolver(
    seeds=seeds,
    fitness_fn=fitness,
    population_size=12,
    generations=8,
    mutation_rate=0.35,
    seed=42,
)

best = evolver.run()
print(f"Best score: {best.score:.3f}")
print(f"Best prompt:\n{best.text}")
```

## How the Loop Works

```
seeds -> initial population
   |
   v
score every genome (fitness_fn)
   |
   v
select parents (tournament / elitist / rank)
   |
   v
crossover + mutate -> children
   |
   v
form next generation -> repeat until stop
```

Stop conditions: max generations, score plateau (configurable patience), or a wall-clock budget.

## Run Logs

Every generation is appended to `runs/<timestamp>.jsonl` so you can audit the search:

```json
{"gen": 0, "id": "g0-3", "score": 0.71, "text": "Summarize ..."}
{"gen": 1, "id": "g1-7", "score": 0.78, "parents": ["g0-3", "g0-1"], ...}
```

Use the bundled `prompt-genome inspect runs/<file>.jsonl` CLI to print a leaderboard.

## Examples

See the `examples/` directory:

- `examples/summarization.py` - evolve a summarization prompt against a small dev set.
- `examples/classifier.py` - evolve a classification prompt with a label-match fitness.
- `examples/cli_demo.py` - end-to-end CLI run with a mock fitness function.

## Contributing

Issues and PRs are welcome. Good first contributions:

- New mutation operators (e.g. instruction reordering, few-shot example injection).
- Adapters for common eval frameworks (lm-eval-harness, promptfoo, deepeval).
- Smarter stop conditions (e.g. confidence intervals on score deltas).

Run the tests with:

```bash
python -m unittest discover -s tests
```

Please keep new code dependency-free in the core, and put any heavy adapters under `prompt_genome/adapters/`.

## License

MIT. See LICENSE.
