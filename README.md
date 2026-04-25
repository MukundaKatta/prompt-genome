# prompt-genome

A tiny, dependency-free evolutionary optimizer for LLM prompts. Mutate, crossover, and select prompts based on your eval scores.

## Install

```bash
pip install -e .
```

Pure-stdlib at runtime. Python 3.10+. `pytest` is the only dev dependency.

## Concept

A prompt is modeled as a **Genome** of typed **Segments** (`system`, `instruction`, `examples`, `constraints`, `format`, `freeform`). Mutation rewrites, swaps, deletes, inserts, or reorders segments; crossover recombines two parent genomes. A user-supplied `evaluate(prompt) -> float` drives selection. Tournament selection plus elitism guarantee the best-so-far score is non-decreasing across generations.

## Quick start

```python
from prompt_genome import Genome, Segment, Optimizer

pool = [
    Segment("system", "you are concise"),
    Segment("instruction", "answer in one sentence"),
    Segment("constraints", "no filler words"),
]
seed = [Genome(segments=[Segment("system", "be helpful")])]

# Trivially: favor short rendered prompts.
def evaluate(prompt: str) -> float:
    return -float(len(prompt))

opt = Optimizer(evaluate, pool, population_size=10, seed=0)
result = opt.evolve(seed, generations=10)
print(result.best_score, result.best.render())
```

## CLI

```bash
pgen evolve \
  --population pop.jsonl \
  --eval my_eval.py \
  --pool pool.jsonl \
  --generations 10 \
  --seed 0 \
  --out best.json
```

`pop.jsonl` is one Genome per line (`{"segments":[{"kind":"system","text":"..."}],"meta":{}}`). `pool.jsonl` is one Segment per line (`{"kind":"instruction","text":"..."}`). The output file is JSON with `score`, `history`, `genome`, and `rendered`.

`pgen --help` and `pgen --version` are also available.

## Eval file contract

`--eval` points at a Python file that defines a top-level callable:

```python
def evaluate(prompt: str) -> float:
    # higher is better
    return your_score
```

The file is loaded via `importlib.util.spec_from_file_location`, so any pure-Python implementation works. `prompt` is the rendered Genome (`Genome.render()`).

## Determinism

Every operator and the optimizer take a `random.Random` (or a `seed`). Same seed plus same `evaluate` plus same inputs produce the same result.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## License

MIT. See `LICENSE`.
