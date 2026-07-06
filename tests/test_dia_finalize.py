"""Tests for scripts/dia_finalize.py (training demo finalization helpers)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ai_impact_accounting import track
from ai_impact_accounting.producer.cli import cmd_validate


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dia_finalize import exit_from_finalize, finalize_run, hub_push_enabled  # noqa: E402


def _finished_track() -> track:
    with track(base_model="distilbert-base-uncased", relation="finetune", region="test") as t:
        pass
    return t


@pytest.mark.parametrize(
    ("repo", "env", "expected"),
    [
        ("org/model", {}, True),
        ("org/model", {"DIA_LOCAL": "1"}, False),
        ("org/model", {"DIA_LOCAL": "true"}, False),
        ("org/model", {"DIA_NO_PUSH": "yes"}, False),
        ("", {}, False),
        ("  ", {}, False),
    ],
)
def test_hub_push_enabled(monkeypatch, repo, env, expected):
    for key in ("DIA_LOCAL", "DIA_NO_PUSH"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert hub_push_enabled(repo) is expected


def test_finalize_run_local_only_writes_card(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DIA_LOCAL", "1")
    out_dir = tmp_path / "out-bert"
    out_dir.mkdir()
    saved = {"called": False}

    def save_fn() -> None:
        saved["called"] = True
        (out_dir / "model.bin").write_text("weights")

    code = finalize_run(
        _finished_track(),
        out_dir=str(out_dir),
        repo="org/my-model",
        token="hf_test",
        base_model="distilbert-base-uncased",
        save_fn=save_fn,
        push_fn=lambda: (_ for _ in ()).throw(AssertionError("push should be skipped")),
    )

    assert code == 0
    assert saved["called"]
    card = out_dir / "README.md"
    assert card.is_file()
    assert cmd_validate(str(card)) == 0
    assert "Skipping Hub push (DIA_LOCAL set" in capsys.readouterr().out


def test_finalize_run_interrupted_returns_130(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DIA_LOCAL", "1")
    out_dir = tmp_path / "out-bert"
    out_dir.mkdir()

    code = finalize_run(
        _finished_track(),
        out_dir=str(out_dir),
        repo="org/my-model",
        token=None,
        base_model="distilbert-base-uncased",
        save_fn=lambda: None,
        interrupted=True,
    )

    assert code == 130
    assert "Interrupted — finalizing partial run locally." in capsys.readouterr().out


def test_finalize_run_no_token_skips_push(tmp_path, capsys):
    out_dir = tmp_path / "out-bert"
    out_dir.mkdir()

    code = finalize_run(
        _finished_track(),
        out_dir=str(out_dir),
        repo="org/my-model",
        token=None,
        base_model="distilbert-base-uncased",
        save_fn=lambda: None,
        push_fn=lambda: (_ for _ in ()).throw(AssertionError("push should be skipped")),
    )

    assert code == 0
    assert "No HF token — skipping Hub push." in capsys.readouterr().out


def test_finalize_run_push_failure_returns_1(tmp_path, capsys):
    out_dir = tmp_path / "out-bert"
    out_dir.mkdir()

    def boom() -> None:
        raise RuntimeError("network down")

    code = finalize_run(
        _finished_track(),
        out_dir=str(out_dir),
        repo="org/my-model",
        token="hf_test",
        base_model="distilbert-base-uncased",
        save_fn=lambda: None,
        push_fn=boom,
    )

    assert code == 1
    assert "WARNING: Hub push failed (network down)" in capsys.readouterr().out
    assert (out_dir / "README.md").is_file()


def test_finalize_run_successful_push(tmp_path, capsys):
    out_dir = tmp_path / "out-bert"
    out_dir.mkdir()
    pushed = {"called": False}

    def push_fn() -> None:
        pushed["called"] = True

    code = finalize_run(
        _finished_track(),
        out_dir=str(out_dir),
        repo="org/my-model",
        token="hf_test",
        base_model="distilbert-base-uncased",
        save_fn=lambda: None,
        push_fn=push_fn,
    )

    assert code == 0
    assert pushed["called"]
    assert "https://huggingface.co/org/my-model" in capsys.readouterr().out


def test_exit_from_finalize_exits_on_nonzero():
    with pytest.raises(SystemExit) as exc:
        exit_from_finalize(1)
    assert exc.value.code == 1


def test_exit_from_finalize_no_exit_on_zero():
    exit_from_finalize(0)
