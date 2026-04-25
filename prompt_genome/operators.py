"""Mutation and crossover operators.

All operators take an `rng` (random.Random) so the search is reproducible
when the Evolver is seeded.
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple


# --- mutation operators -----------------------------------------------------

_FILLER_SWAPS = {
    "summarize": ["condense", "boil down", "distill"],
    "explain": ["walk through", "break down", "clarify"],
    "list": ["enumerate", "lay out", "spell out"],
    "concise": ["tight", "brief", "compact"],
    "detailed": ["thorough", "in depth", "comprehensive"],
    "analyze": ["examine", "evaluate", "investigate"],
    "answer": ["respond to", "address", "tackle"],
}


def word_swap(text: str, rng: random.Random, max_swaps: int = 2) -> str:
    """Swap a few common verbs/adjectives with synonyms.

    Keeps semantics roughly equivalent while perturbing surface form.
    """
    tokens = re.split(r"(\s+)", text)  # keep whitespace as separators
    candidates = [
        i for i, tok in enumerate(tokens)
        if tok.lower().strip(".,;:!?") in _FILLER_SWAPS
    ]
    if not candidates:
        return text
    picks = rng.sample(candidates, k=min(max_swaps, len(candidates)))
    for i in picks:
        raw = tokens[i]
        bare = raw.lower().strip(".,;:!?")
        repl_choices = _FILLER_SWAPS[bare]
        new_word = rng.choice(repl_choices)
        # preserve trailing punctuation and original capitalization of first letter
        trail = ""
        for ch in reversed(raw):
            if ch in ".,;:!?":
                trail = ch + trail
            else:
                break
        if raw[:1].isupper():
            new_word = new_word[:1].upper() + new_word[1:]
        tokens[i] = new_word + trail
    return "".join(tokens)


_INSTRUCTION_BANK = [
    "Be precise and avoid speculation.",
    "Cite the source spans you used.",
    "If unsure, say so explicitly.",
    "Use plain language, no jargon.",
    "Keep the answer self contained.",
    "Prefer concrete examples over abstractions.",
]


def instruction_inject(text: str, rng: random.Random) -> str:
    """Append (or prepend) a steering instruction from a small bank."""
    instr = rng.choice(_INSTRUCTION_BANK)
    if rng.random() < 0.5:
        return f"{text.rstrip()} {instr}"
    return f"{instr} {text.lstrip()}"


def fragment_splice(text: str, rng: random.Random, donor_pool: List[str]) -> str:
    """Splice a sentence-sized fragment from another prompt into this one."""
    if not donor_pool:
        return text
    donor = rng.choice(donor_pool)
    donor_sents = _split_sentences(donor)
    if not donor_sents:
        return text
    fragment = rng.choice(donor_sents).strip()
    sents = _split_sentences(text)
    if not sents:
        return f"{text.rstrip()} {fragment}"
    insert_at = rng.randrange(len(sents) + 1)
    sents.insert(insert_at, fragment)
    return " ".join(s.strip() for s in sents if s.strip())


_PERSONAS = [
    "You are a careful technical writer.",
    "You are a senior code reviewer.",
    "You are a patient teacher.",
    "You are a research analyst.",
    "You are a domain expert focused on accuracy.",
]


def persona_shuffle(text: str, rng: random.Random) -> str:
    """Replace any leading 'You are ...' line with a fresh persona, or add one."""
    persona = rng.choice(_PERSONAS)
    lines = text.splitlines()
    if lines and lines[0].lower().lstrip().startswith("you are"):
        lines[0] = persona
        return "\n".join(lines)
    return f"{persona}\n{text}"


# --- crossover --------------------------------------------------------------

def crossover_sentences(parent_a: str, parent_b: str, rng: random.Random) -> str:
    """Sentence-boundary single-point crossover.

    Takes the first half (by sentence count) of A and the second half of B.
    Produces a child that is grammatical at the seams.
    """
    a_sents = _split_sentences(parent_a)
    b_sents = _split_sentences(parent_b)
    if not a_sents and not b_sents:
        return parent_a
    if not a_sents:
        return parent_b
    if not b_sents:
        return parent_a
    cut_a = max(1, len(a_sents) // 2)
    cut_b = max(1, len(b_sents) // 2)
    # add small jitter so identical-length parents don't always split the same way
    cut_a = max(1, min(len(a_sents), cut_a + rng.choice([-1, 0, 1])))
    cut_b = max(1, min(len(b_sents), cut_b + rng.choice([-1, 0, 1])))
    head = a_sents[:cut_a]
    tail = b_sents[cut_b:]
    if not tail:
        tail = b_sents[-1:]
    return " ".join(s.strip() for s in head + tail if s.strip())


# --- helpers ----------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def _split_sentences(text: str) -> List[str]:
    """Split a chunk of text into sentence-ish pieces.

    Naive but good enough for prompts; we fall back to line splits if needed.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    if len(parts) == 1:
        # try line-based split as a fallback
        line_parts = [p for p in text.splitlines() if p.strip()]
        if len(line_parts) > 1:
            return line_parts
    return parts
