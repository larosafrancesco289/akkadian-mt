"""Build document-level subsampled training sets for data scaling experiments.

Creates 25%, 50%, 75% subsets of the gold corpus by sampling documents
(grouped by oare_id) so that within-document sentence pairs stay together.
The full 100% baseline already exists as golden_corpus_v2_prefilt.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        default="data/processed/golden_corpus_v2_prefilt.csv",
        help="Full gold corpus CSV",
    )
    parser.add_argument(
        "--holdout-file",
        default="data/processed/coursework_holdout_doc_ids.json",
        help="Holdout doc IDs to exclude before subsampling",
    )
    parser.add_argument(
        "--allow-missing-holdout",
        action="store_true",
        help="Allow subset generation to continue when the holdout JSON is missing",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/coursework_scaling",
        help="Output directory for subsampled CSVs",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="Fractions of documents to sample",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    train_df = pd.read_csv(args.train_file)
    holdout_ids: set[str] = set()
    holdout_path = Path(args.holdout_file)
    if holdout_path.exists():
        holdout_ids = {str(x) for x in json.loads(holdout_path.read_text())}
    elif not args.allow_missing_holdout:
        raise FileNotFoundError(
            f"Missing holdout file: {holdout_path}. "
            "Run coursework/scripts/build_coursework_holdout_union.py first, or pass "
            "--allow-missing-holdout if you explicitly want leaky subsets."
        )

    # Identify document column
    doc_col = "oare_id" if "oare_id" in train_df.columns else "doc_id"
    if doc_col not in train_df.columns:
        raise ValueError(f"No document column ({doc_col}) found in {args.train_file}")

    # Get unique document IDs (excluding holdout)
    all_doc_ids = train_df[doc_col].astype(str).unique()
    eligible_docs = sorted(d for d in all_doc_ids if d not in holdout_ids)

    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(eligible_docs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}

    for frac in args.fractions:
        n_docs = max(1, int(len(shuffled) * frac))
        selected_docs = set(shuffled[:n_docs])

        subset = train_df[train_df[doc_col].astype(str).isin(selected_docs)].copy()
        tag = f"{int(frac * 100)}pct"
        out_path = output_dir / f"train_{tag}.csv"
        subset.to_csv(out_path, index=False)

        manifest[tag] = {
            "fraction": frac,
            "n_docs": n_docs,
            "total_docs": len(eligible_docs),
            "n_rows": len(subset),
            "total_rows": len(train_df),
            "file": str(out_path),
        }
        print(f"{tag}: {n_docs}/{len(eligible_docs)} docs, {len(subset)} rows -> {out_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
