"""Report figure helpers: grouped training-loss curves from rerun histories."""

from __future__ import annotations

import csv
from pathlib import Path

from akkadian_mt.config import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_under_root(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def _load_loss_curve_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing loss-curve manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Loss-curve manifest is empty: {manifest_path}")
    return rows


def _resolve_loss_history_path(root: Path, row: dict[str, str]) -> Path:
    history_path = (row.get("history_path") or "").strip()
    if history_path:
        return _resolve_under_root(root, history_path)

    config_path = (row.get("config_path") or "").strip()
    if not config_path:
        raise RuntimeError(f"Manifest row is missing both history_path and config_path: {row}")

    config = load_config(_resolve_under_root(root, config_path))
    output_dir = Path(config.train.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return output_dir / "training_history.csv"


def _load_train_loss_series(history_path: Path) -> list[tuple[int, float]]:
    with history_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        by_step: dict[int, float] = {}
        for row in reader:
            if row.get("metric") != "train/loss":
                continue
            step_text = (row.get("step") or "").strip()
            value_text = (row.get("value") or "").strip()
            if not step_text or not value_text:
                continue
            by_step[int(step_text)] = float(value_text)
    return sorted(by_step.items())


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    smoothed: list[float] = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
        count = min(index + 1, window)
        smoothed.append(running_total / count)
    return smoothed


def build_loss_curve_plots(
    *,
    root: Path | None = None,
    manifest_path: Path,
    output_dir: Path,
    priorities: set[str] | None = None,
    groups: set[str] | None = None,
    smoothing_window: int = 20,
) -> list[Path]:
    root = root or _repo_root()
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_loss_curve_manifest(manifest_path)
    if priorities is not None:
        rows = [row for row in rows if (row.get("priority") or "").strip() in priorities]
    if groups is not None:
        rows = [row for row in rows if (row.get("comparison_group") or "").strip() in groups]
    if not rows:
        raise RuntimeError("No loss-curve manifest rows matched the requested filters")

    group_rows: dict[str, list[tuple[dict[str, str], list[tuple[int, float]]]]] = {}
    inventory: list[dict[str, str | int | float]] = []
    for row in rows:
        history_path = _resolve_loss_history_path(root, row)
        group_name = (row.get("comparison_group") or "").strip()
        status = "ok"
        point_count = 0
        final_step = ""
        final_loss = ""
        series: list[tuple[int, float]] = []

        if not history_path.exists():
            status = "missing_history"
        else:
            series = _load_train_loss_series(history_path)
            if not series:
                status = "missing_train_loss"
            else:
                point_count = len(series)
                final_step = series[-1][0]
                final_loss = series[-1][1]
                group_rows.setdefault(group_name, []).append((row, series))

        inventory.append(
            {
                "run_id": (row.get("run_id") or "").strip(),
                "priority": (row.get("priority") or "").strip(),
                "comparison_group": group_name,
                "curve_label": (row.get("curve_label") or "").strip(),
                "status": status,
                "history_path": str(history_path.relative_to(root)),
                "num_points": point_count,
                "final_step": final_step,
                "final_loss": final_loss,
            }
        )

    inventory_path = output_dir / "manifest_status.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "priority",
                "comparison_group",
                "curve_label",
                "status",
                "history_path",
                "num_points",
                "final_step",
                "final_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(inventory)

    if not group_rows:
        raise RuntimeError(
            "No training history files with train/loss rows were found for the requested runs"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written_paths = [inventory_path]
    for group_name, entries in sorted(group_rows.items()):
        ordered_entries = sorted(entries, key=lambda item: int(item[0].get("order") or "0"))
        fig, ax = plt.subplots(figsize=(8.6, 5.2))

        for row, series in ordered_entries:
            steps = [step for step, _ in series]
            losses = [value for _, value in series]
            smooth = _moving_average(losses, smoothing_window)
            label = (row.get("curve_label") or row.get("run_id") or "").strip()
            smooth_line = ax.plot(steps, smooth, linewidth=2.2, label=label)[0]
            ax.plot(
                steps,
                losses,
                linewidth=0.9,
                alpha=0.18,
                color=smooth_line.get_color(),
            )

        title = (ordered_entries[0][0].get("group_title") or group_name.replace("_", " ")).strip()
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Training loss")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.legend(loc="best", fontsize=9, ncol=2 if len(ordered_entries) > 4 else 1)
        fig.tight_layout()

        stem = output_dir / f"{group_name}_training_loss"
        png_path = stem.with_suffix(".png")
        pdf_path = stem.with_suffix(".pdf")
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        written_paths.extend([png_path, pdf_path])

    return written_paths
