"""Tiny CLI for inspecting run logs.

Usage:
    python -m prompt_genome.cli inspect runs/20260424-120000.jsonl
    python -m prompt_genome.cli inspect runs/<file>.jsonl --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Iterable, List


def _load(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _leaderboard(rows: List[dict], top: int) -> List[dict]:
    scored = [r for r in rows if isinstance(r.get("score"), (int, float))]
    return sorted(scored, key=lambda r: r["score"], reverse=True)[:top]


def _per_gen_best(rows: List[dict]) -> List[tuple]:
    by_gen = defaultdict(list)
    for r in rows:
        if isinstance(r.get("score"), (int, float)):
            by_gen[r["gen"]].append(r["score"])
    return sorted(
        ((gen, max(scores), sum(scores) / len(scores)) for gen, scores in by_gen.items()),
        key=lambda t: t[0],
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    rows = _load(args.path)
    if not rows:
        print(f"no rows in {args.path}", file=sys.stderr)
        return 1

    print(f"== Run summary: {args.path}")
    print(f"   total entries: {len(rows)}")
    print()
    print("== Per-generation (best, mean):")
    for gen, best, mean in _per_gen_best(rows):
        print(f"   gen {gen:>2}: best={best:.4f}  mean={mean:.4f}")
    print()
    print(f"== Top {args.top} prompts:")
    for r in _leaderboard(rows, args.top):
        text = r["text"].replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"   [{r['score']:.4f}] {r['id']}: {text}")
    return 0


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="prompt-genome")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="show leaderboard for a run log")
    p_inspect.add_argument("path", help="path to a runs/*.jsonl file")
    p_inspect.add_argument("--top", type=int, default=5, help="how many top prompts to show")
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
