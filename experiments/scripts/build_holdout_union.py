"""Build the holdout union from the two independent holdout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_ids(path: Path, *, allow_missing: bool) -> set[str]:
    if not path.exists():
        if allow_missing:
            return set()
        raise FileNotFoundError(
            f"Missing holdout file: {path}. Commit or copy the processed artifacts first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return {str(item) for item in payload if str(item).strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-file",
        action="append",
        default=[],
        help="Input holdout JSON list. Repeat to add more files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/coursework_holdout_doc_ids.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing input files instead of failing.",
    )
    args = parser.parse_args()

    all_ids: set[str] = set()
    holdout_files = args.holdout_file or [
        "data/processed/holdout_doc_ids.json",
        "data/processed/new_holdout_doc_ids.json",
    ]
    for file_name in holdout_files:
        all_ids.update(_load_ids(Path(file_name), allow_missing=args.allow_missing))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sorted(all_ids), indent=2), encoding="utf-8")

    print(f"Wrote {len(all_ids)} holdout IDs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
