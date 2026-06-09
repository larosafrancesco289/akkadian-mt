from __future__ import annotations

import csv
from pathlib import Path

from akkadian_mt.commands.coursework import build_loss_curve_plots


def _write_history(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "metric", "value"])
        writer.writeheader()
        for step, loss in enumerate(losses, start=1):
            writer.writerow(
                {
                    "step": step,
                    "epoch": 1,
                    "metric": "train/loss",
                    "value": loss,
                }
            )
            writer.writerow(
                {
                    "step": step,
                    "epoch": 1,
                    "metric": "val/combined",
                    "value": 30.0 + step,
                }
            )


def test_build_loss_curve_plots_writes_grouped_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "coursework" / "results" / "loss_curve_runs.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    history_a = tmp_path / "outputs" / "coursework" / "run_a" / "training_history.csv"
    history_b = tmp_path / "outputs" / "coursework" / "run_b" / "training_history.csv"
    _write_history(history_a, [4.0, 3.2, 2.8, 2.3])
    _write_history(history_b, [5.0, 4.1, 3.7, 3.1])

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "priority",
                "comparison_group",
                "group_title",
                "order",
                "curve_label",
                "config_path",
                "history_path",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run_a",
                "priority": "core",
                "comparison_group": "toy_group",
                "group_title": "Toy Group",
                "order": 1,
                "curve_label": "Run A",
                "config_path": "",
                "history_path": "outputs/coursework/run_a/training_history.csv",
                "reason": "test",
            }
        )
        writer.writerow(
            {
                "run_id": "run_b",
                "priority": "core",
                "comparison_group": "toy_group",
                "group_title": "Toy Group",
                "order": 2,
                "curve_label": "Run B",
                "config_path": "",
                "history_path": "outputs/coursework/run_b/training_history.csv",
                "reason": "test",
            }
        )

    output_dir = tmp_path / "coursework" / "report" / "figures" / "loss_curves"
    written = build_loss_curve_plots(
        root=tmp_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        priorities={"core"},
        smoothing_window=2,
    )

    assert output_dir / "manifest_status.csv" in written
    assert output_dir / "toy_group_training_loss.pdf" in written
    assert output_dir / "toy_group_training_loss.png" in written
    assert (output_dir / "manifest_status.csv").exists()
    assert (output_dir / "toy_group_training_loss.pdf").exists()
    assert (output_dir / "toy_group_training_loss.png").exists()
