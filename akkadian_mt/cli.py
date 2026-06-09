"""Canonical CLI for akkadian-mt.

This command is the authoritative entrypoint for the paper's reproduction
pipeline: training, evaluation, and report figures. Shared implementation lives
in `akkadian_mt/`; experiment orchestration scripts live under
`experiments/scripts/`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable


def _forward(module_path: str) -> Callable[[argparse.Namespace, list[str]], int]:
    """Forward unparsed CLI args to a module's ``main()`` via ``sys.argv``.

    The training/evaluation modules own their own argument parsers, so the CLI
    only routes the subcommand and hands the remaining tokens straight through.
    """

    def _handler(_: argparse.Namespace, unknown: list[str]) -> int:
        module = importlib.import_module(module_path)
        if not hasattr(module, "main"):
            raise RuntimeError(f"Module {module_path!r} has no main()")
        old_argv = sys.argv[:]
        sys.argv = [f"{module_path.rsplit('.', 1)[-1]}.py", *unknown]
        try:
            result = module.main()
            return int(result) if isinstance(result, int) else 0
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            return code if isinstance(code, int) else 1
        finally:
            sys.argv = old_argv

    return _handler


def _handle_plot_loss_curves(ns: argparse.Namespace, _: list[str]) -> int:
    from akkadian_mt.report import build_loss_curve_plots

    written = build_loss_curve_plots(
        manifest_path=Path(ns.manifest),
        output_dir=Path(ns.output_dir),
        priorities=set(ns.priority) if ns.priority else None,
        groups=set(ns.group) if ns.group else None,
        smoothing_window=ns.smoothing_window,
    )
    for path in written:
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="akkadian-mt",
        description="Reproduction CLI for byte-level Akkadian-to-English MT.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train = subparsers.add_parser("train", help="Training commands")
    train_sub = train.add_subparsers(dest="train_cmd", required=True)
    train_seq2seq = train_sub.add_parser(
        "seq2seq", help="Fine-tune a T5-family seq2seq model from a YAML config"
    )
    train_seq2seq.set_defaults(handler=_forward("akkadian_mt.train"))

    # eval
    eval_p = subparsers.add_parser("eval", help="Evaluation commands")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_val = eval_sub.add_parser("val", help="Validation-split evaluation")
    eval_val.set_defaults(handler=_forward("akkadian_mt.evaluate"))
    eval_lb = eval_sub.add_parser("lb", help="Independent / held-out test-set evaluation")
    eval_lb.set_defaults(handler=_forward("akkadian_mt.evaluation.evaluate_lb"))

    # report helpers
    report = subparsers.add_parser("report", help="Report figure helpers")
    cw_sub = report.add_subparsers(dest="report_cmd", required=True)
    cw_loss = cw_sub.add_parser(
        "plot-loss-curves", help="Build grouped training-loss plots from rerun histories"
    )
    cw_loss.add_argument(
        "--manifest",
        default="experiments/results/loss_curve_runs.csv",
        help="CSV manifest listing the runs and grouping for loss-curve reruns",
    )
    cw_loss.add_argument(
        "--output-dir",
        default="experiments/report/figures/loss_curves",
        help="Directory where the loss-curve figures are written",
    )
    cw_loss.add_argument(
        "--priority", action="append", help="Optional manifest priority filter (repeatable)"
    )
    cw_loss.add_argument(
        "--group", action="append", help="Optional manifest group filter (repeatable)"
    )
    cw_loss.add_argument("--smoothing-window", type=int, default=1, help="Loss smoothing window")
    cw_loss.set_defaults(handler=_handle_plot_loss_curves)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns, unknown = parser.parse_known_args(argv)
    handler = getattr(ns, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(ns, unknown)


if __name__ == "__main__":
    sys.exit(main())
