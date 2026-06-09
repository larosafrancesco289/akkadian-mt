from __future__ import annotations

import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "akkadian_mt.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help() -> None:
    proc = run_cli("--help")
    assert proc.returncode == 0
    assert proc.stdout.startswith("usage: akkadian-mt")


def test_cli_train_help() -> None:
    proc = run_cli("train", "seq2seq", "--help")
    assert proc.returncode == 0
    assert "usage: akkadian-mt train seq2seq" in proc.stdout


def test_cli_eval_help() -> None:
    proc = run_cli("eval", "val", "--help")
    assert proc.returncode == 0
    assert "usage: akkadian-mt eval val" in proc.stdout


def test_cli_data_build_external_mix_help() -> None:
    proc = run_cli("data", "build-external-mix", "--help")
    assert proc.returncode == 0
    assert "usage: akkadian-mt data build-external-mix" in proc.stdout


def test_cli_report_plot_loss_curves_help() -> None:
    proc = run_cli("report", "plot-loss-curves", "--help")
    assert proc.returncode == 0
    assert "usage: akkadian-mt report plot-loss-curves" in proc.stdout


def test_build_parser_prog() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import akkadian_mt.cli as cli; print(cli.build_parser().prog)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "akkadian-mt"
