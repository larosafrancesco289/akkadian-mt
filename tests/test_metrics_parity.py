from __future__ import annotations

import pytest

pytest.importorskip("sacrebleu")

from akkadian_mt.evaluation.evaluate_lb import (
    compute_combined_score as legacy_compute_combined_score,
)
from akkadian_mt.evaluation.metrics import compute_combined_score


def test_metrics_parity_with_legacy_lb() -> None:
    preds = ["Say to my father.", "He brought the silver."]
    refs = ["Say to my father.", "He brought silver."]
    current = compute_combined_score(preds, refs)
    legacy = legacy_compute_combined_score(preds, refs)
    assert current == legacy
