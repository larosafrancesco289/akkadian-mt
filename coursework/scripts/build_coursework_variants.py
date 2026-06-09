"""Materialize the coursework experiment datasets for the CW4 pivot."""

from __future__ import annotations

import argparse
import json

from akkadian_mt.data.coursework_variants import (
    DEFAULT_HOLDOUT_FILE,
    DEFAULT_PRIMARY_TEST_FILE,
    DEFAULT_SECONDARY_TEST_FILE,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VARIANT_DIR,
    build_all_coursework_variants,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--primary-test-file", default=str(DEFAULT_PRIMARY_TEST_FILE))
    parser.add_argument("--secondary-test-file", default=str(DEFAULT_SECONDARY_TEST_FILE))
    parser.add_argument("--holdout-file", default=str(DEFAULT_HOLDOUT_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_VARIANT_DIR))
    args = parser.parse_args()

    paths = build_all_coursework_variants(
        train_file=args.train_file,
        primary_test_file=args.primary_test_file,
        secondary_test_file=args.secondary_test_file,
        holdout_file=args.holdout_file,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                name: {
                    "train_file": str(artifact.train_file),
                    "primary_test_file": str(artifact.primary_test_file),
                    "secondary_test_file": str(artifact.secondary_test_file),
                }
                for name, artifact in paths.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
