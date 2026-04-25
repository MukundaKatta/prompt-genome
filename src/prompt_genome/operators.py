"""Mutation and crossover operators.

All operators take an explicit `random.Random` so behavior is deterministic
given a seeded rng. Operators always return new Genome instances; inputs
are never mutated.
"""

from __future__ import annotations

import random
from typing import Sequence

from .genome import Genome, Segment, KIND_ORDER


# Names of the per-segment mutation primitives. Kept here so the choice is
# stable across runs (sorted by name → reproducible given a seeded rng).
_MUTATION_OPS: tuple[str, ...] = (
    "delete_segment",
    "insert_from_pool",
    "reorder_kind_group",
    "replace_text",
    "swap_with_pool",
)


def _replace_text(seg: Segment, *, rng: random.Random, pool: Sequence[Segment]) -> Segment:
    """Replace the text of `seg` with text from a pool segment of the same kind.

    Falls back to identity if no same-kind pool entry exists.
    """
    same_kind = [p for p in pool if p.kind == seg.kind and p.text != seg.text]
    if not same_kind:
        return seg
    chosen = rng.choice(same_kind)
    return Segment(kind=seg.kind, text=chosen.text)


def _swap_with_pool(seg: Segment, *, rng: random.Random, pool: Sequence[Segment]) -> Segment:
    """Replace `seg` with any pool segment (kind may change).

    Falls back to identity if pool is empty.
    """
    if not pool:
        return seg
    return rng.choice(pool)


def mutate(
    genome: Genome,
    *,
    rng: random.Random,
    segment_pool: list[Segment],
    rate: float = 0.3,
) -> Genome:
    """Return a mutated copy of `genome`.

    For each segment we draw a uniform; if it falls below `rate`, we apply
    one randomly-chosen mutation primitive. The original genome is never
    mutated.

    Primitives:
      - replace_text       → swap text with a same-kind pool entry
      - swap_with_pool     → replace the segment with any pool entry
      - delete_segment     → drop the segment
      - insert_from_pool   → insert a pool segment after the current one
      - reorder_kind_group → shuffle relative order within the segment's kind
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    # Walk the original list and emit the mutated result. We iterate by
    # index so insertions/deletions can be expressed naturally.
    new_segments: list[Segment] = []
    for seg in genome.segments:
        if rng.random() >= rate:
            new_segments.append(seg)
            continue

        op = rng.choice(_MUTATION_OPS)
        if op == "replace_text":
            new_segments.append(_replace_text(seg, rng=rng, pool=segment_pool))
        elif op == "swap_with_pool":
            new_segments.append(_swap_with_pool(seg, rng=rng, pool=segment_pool))
        elif op == "delete_segment":
            # Skip emitting; this drops the segment.
            continue
        elif op == "insert_from_pool":
            new_segments.append(seg)
            if segment_pool:
                new_segments.append(rng.choice(segment_pool))
        elif op == "reorder_kind_group":
            # Defer: we shuffle within-kind groups after the first pass so
            # the operator's effect is well-defined regardless of position.
            new_segments.append(seg)
            new_segments = _reorder_kind_group(new_segments, kind=seg.kind, rng=rng)
        else:  # pragma: no cover - defensive
            new_segments.append(seg)

    return Genome(segments=new_segments, meta=dict(genome.meta))


def _reorder_kind_group(
    segments: list[Segment], *, kind: str, rng: random.Random
) -> list[Segment]:
    """Shuffle (in-list) the relative order of segments matching `kind`.

    Other kinds keep their positions; only the slots occupied by `kind`
    are permuted among themselves.
    """
    indices = [i for i, s in enumerate(segments) if s.kind == kind]
    if len(indices) < 2:
        return segments
    picked = [segments[i] for i in indices]
    rng.shuffle(picked)
    out = list(segments)
    for slot, seg in zip(indices, picked):
        out[slot] = seg
    return out


def crossover(
    a: Genome,
    b: Genome,
    *,
    rng: random.Random,
    mode: str = "single_point",
) -> tuple[Genome, Genome]:
    """Recombine two genomes into two children.

    Modes:
      - "single_point": pick a cut index in [0, min(len(a), len(b))]; child1
        takes a[:cut] + b[cut:], child2 takes b[:cut] + a[cut:]. Total
        segment counts may differ when parents are different lengths.
      - "uniform": for each position up to min length, flip a coin to swap;
        any tail (longer parent) is appended to the corresponding child so
        no segments are lost.
    """
    if mode not in {"single_point", "uniform"}:
        raise ValueError(f"crossover mode must be 'single_point' or 'uniform', got {mode!r}")

    a_segs = list(a.segments)
    b_segs = list(b.segments)

    if mode == "single_point":
        # Cut bounds inclusive of 0 and min length, so identity / full-swap
        # are both reachable.
        upper = min(len(a_segs), len(b_segs))
        cut = rng.randint(0, upper)
        c1 = a_segs[:cut] + b_segs[cut:]
        c2 = b_segs[:cut] + a_segs[cut:]
        return (
            Genome(segments=c1, meta=dict(a.meta)),
            Genome(segments=c2, meta=dict(b.meta)),
        )

    # uniform
    pair_len = min(len(a_segs), len(b_segs))
    c1: list[Segment] = []
    c2: list[Segment] = []
    for i in range(pair_len):
        if rng.random() < 0.5:
            c1.append(a_segs[i])
            c2.append(b_segs[i])
        else:
            c1.append(b_segs[i])
            c2.append(a_segs[i])
    # Append any tail from the longer parent so segment counts are preserved.
    c1.extend(a_segs[pair_len:])
    c2.extend(b_segs[pair_len:])
    return (
        Genome(segments=c1, meta=dict(a.meta)),
        Genome(segments=c2, meta=dict(b.meta)),
    )


# Re-export to keep public surface tidy.
__all__ = ["mutate", "crossover", "KIND_ORDER"]
