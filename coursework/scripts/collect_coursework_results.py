"""Append coursework run metrics into coursework/results/results.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "run_id",
    "model_family",
    "model_size",
    "seed",
    "ablation",
    "config_path",
    "checkpoint_path",
    "summary_path",
    "independent_bleu",
    "independent_chrf",
    "independent_combined",
    "newtest_bleu",
    "newtest_chrf",
    "newtest_combined",
    "status",
    "notes",
]


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Defaults to coursework/results/artifacts/<run_id>/summary.json",
    )
    parser.add_argument(
        "--results-csv",
        default="coursework/results/results.csv",
        help="Results table to append or update.",
    )
    parser.add_argument("--status", default="complete")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    summary_path = Path(
        args.summary_path or f"coursework/results/artifacts/{args.run_id}/summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "run_id": args.run_id,
        "model_family": args.model_family,
        "model_size": args.model_size,
        "seed": args.seed,
        "ablation": args.ablation,
        "config_path": args.config_path,
        "checkpoint_path": args.checkpoint_path or summary.get("artifact_source"),
        "summary_path": str(summary_path),
        "independent_bleu": summary["independent"]["bleu"],
        "independent_chrf": summary["independent"]["chrf"],
        "independent_combined": summary["independent"]["combined"],
        "newtest_bleu": summary["newtest"]["bleu"],
        "newtest_chrf": summary["newtest"]["chrf"],
        "newtest_combined": summary["newtest"]["combined"],
        "status": args.status,
        "notes": args.notes,
    }

    results_path = Path(args.results_csv)
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        results_df = _empty_results_frame()

    results_df = results_df[results_df["run_id"] != args.run_id].copy()
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df = results_df[RESULT_COLUMNS]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
