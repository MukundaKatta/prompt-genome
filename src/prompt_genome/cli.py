"""Command-line interface for prompt-genome.

Usage:

    pgen evolve --population pop.jsonl --eval my_eval.py --pool pool.jsonl \
        --generations 10 --seed 0 [--out best.json]

The `--eval` file must define a top-level `evaluate(prompt: str) -> float`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from . import __version__
from .genome import Genome, Segment
from .optimizer import Optimizer


def _load_jsonl(path: Path) -> list[dict]:
    """Load a UTF-8 JSONL file, ignoring blank lines."""
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_population(path: Path) -> list[Genome]:
    return [Genome.from_dict(d) for d in _load_jsonl(path)]


def _load_pool(path: Path) -> list[Segment]:
    return [Segment.from_dict(d) for d in _load_jsonl(path)]


def _load_eval_fn(path: Path) -> Callable[[str], float]:
    """Import a Python source file and return its `evaluate` callable."""
    if not path.exists():
        raise FileNotFoundError(f"eval file not found: {path}")
    spec = importlib.util.spec_from_file_location("pgen_user_eval", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # We don't register in sys.modules: this is a one-shot user script and
    # we don't want to pollute the import cache between CLI runs.
    spec.loader.exec_module(module)
    if not hasattr(module, "evaluate"):
        raise AttributeError(
            f"eval file {path} must define a top-level `evaluate(prompt) -> float`"
        )
    return module.evaluate  # type: ignore[no-any-return]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pgen",
        description="A tiny, dependency-free evolutionary optimizer for LLM prompts.",
    )
    p.add_argument("--version", action="version", version=f"pgen {__version__}")

    sub = p.add_subparsers(dest="command")

    ev = sub.add_parser("evolve", help="Run the evolutionary loop.")
    ev.add_argument("--population", required=True, type=Path, help="JSONL of seed genomes.")
    ev.add_argument("--eval", required=True, type=Path, help="Python file defining evaluate(prompt) -> float.")
    ev.add_argument("--pool", required=True, type=Path, help="JSONL of segments to mutate from.")
    ev.add_argument("--generations", type=int, default=10, help="Number of generations.")
    ev.add_argument("--seed", type=int, default=0, help="RNG seed for determinism.")
    ev.add_argument("--population-size", type=int, default=20, help="Population per generation.")
    ev.add_argument("--mutation-rate", type=float, default=0.3, help="Per-segment mutation rate.")
    ev.add_argument("--elitism", type=int, default=2, help="Top-N survivors carried forward.")
    ev.add_argument("--tournament-size", type=int, default=3, help="Tournament selection size.")
    ev.add_argument("--out", type=Path, default=Path("best.json"), help="Where to write the best genome.")
    return p


def _run_evolve(args: argparse.Namespace) -> int:
    pop = _load_population(args.population)
    pool = _load_pool(args.pool)
    eval_fn = _load_eval_fn(args.eval)

    opt = Optimizer(
        eval_fn=eval_fn,
        segment_pool=pool,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        tournament_size=args.tournament_size,
        seed=args.seed,
    )
    result = opt.evolve(pop, generations=args.generations)

    payload = {
        "score": result.best_score,
        "history": list(result.history),
        "genome": result.best.to_dict(),
        "rendered": result.best.render(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"best score: {result.best_score}")
    print(f"wrote: {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "evolve":
        return _run_evolve(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
