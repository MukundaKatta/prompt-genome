"""Genome and Segment dataclasses.

A Genome is an ordered list of Segments. Each Segment is a (kind, text)
pair where `kind` is one of a small fixed vocabulary. Rendering joins
segments in a canonical kind order, with multiple segments of the same
kind preserving their relative input order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


# Canonical kinds. The render() method emits segments in this order;
# segments sharing a kind keep their original relative order.
KIND_ORDER: tuple[str, ...] = (
    "system",
    "instruction",
    "examples",
    "constraints",
    "format",
    "freeform",
)
VALID_KINDS: frozenset[str] = frozenset(KIND_ORDER)


@dataclass(frozen=True)
class Segment:
    """A single labeled chunk of prompt text.

    Frozen so it is hashable; equality and hashing are over (kind, text).
    """

    kind: str
    text: str

    # Make available without instance lookup.
    VALID_KINDS: ClassVar[frozenset[str]] = VALID_KINDS

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"Segment.kind must be one of {sorted(VALID_KINDS)!r}, got {self.kind!r}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Segment":
        return cls(kind=d["kind"], text=d["text"])


@dataclass
class Genome:
    """An ordered collection of segments that compose a prompt."""

    segments: list[Segment] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render the prompt by emitting segments in canonical kind order.

        Each segment is prefixed with a small header `[<kind>]` and segments
        are joined by blank lines. Multiple segments of the same kind keep
        their relative input order.
        """
        # Bucket per kind, preserving relative order.
        buckets: dict[str, list[Segment]] = {k: [] for k in KIND_ORDER}
        for seg in self.segments:
            # __post_init__ on Segment guarantees kind is valid.
            buckets[seg.kind].append(seg)

        parts: list[str] = []
        for kind in KIND_ORDER:
            for seg in buckets[kind]:
                parts.append(f"[{kind}]\n{seg.text}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genome":
        segs = [Segment.from_dict(s) for s in d.get("segments", [])]
        meta = dict(d.get("meta", {}))
        return cls(segments=segs, meta=meta)

    def clone(self) -> "Genome":
        # Segments are frozen so we can share refs; copy the list + meta.
        return Genome(segments=list(self.segments), meta=dict(self.meta))
