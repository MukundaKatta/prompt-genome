"""Smoke test for `pgen evolve` end-to-end via `main()`."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_genome.cli import main


_EVAL_SRC = '''
def evaluate(prompt: str) -> float:
    # Reward longer prompts; same as the unit test fitness.
    return float(len(prompt))
'''


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_cli_evolve_writes_best_json(tmp_path):
    pop_path = tmp_path / "pop.jsonl"
    pool_path = tmp_path / "pool.jsonl"
    eval_path = tmp_path / "eval_fn.py"
    out_path = tmp_path / "best.json"

    _write_jsonl(
        pop_path,
        [
            {"segments": [{"kind": "system", "text": "a"}], "meta": {}},
            {"segments": [{"kind": "system", "text": "ab"}], "meta": {}},
            {"segments": [{"kind": "instruction", "text": "do it"}], "meta": {}},
        ],
    )
    _write_jsonl(
        pool_path,
        [
            {"kind": "system", "text": "be a verbose and detailed assistant"},
            {"kind": "instruction", "text": "write a long answer with many words"},
            {"kind": "constraints", "text": "include numerous helpful clauses"},
            {"kind": "format", "text": "respond in elaborate paragraphs"},
        ],
    )
    eval_path.write_text(_EVAL_SRC, encoding="utf-8")

    rc = main(
        [
            "evolve",
            "--population", str(pop_path),
            "--eval", str(eval_path),
            "--pool", str(pool_path),
            "--generations", "3",
            "--seed", "0",
            "--population-size", "6",
            "--elitism", "2",
            "--out", str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "score" in payload
    assert isinstance(payload["score"], (int, float))
    assert "history" in payload and isinstance(payload["history"], list) and payload["history"]
    assert "genome" in payload and "segments" in payload["genome"]
    assert "rendered" in payload and isinstance(payload["rendered"], str)


def test_cli_help_exits_cleanly(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "pgen" in out


def test_cli_version_exits_cleanly(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "pgen" in out
