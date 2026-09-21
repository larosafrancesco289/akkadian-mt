"""Paired bootstrap over independent-test sentences for the combined metric.

For each system pair (seed-matched runs sharing the 951-row independent test
set), resample sentences with replacement, recompute corpus BLEU, chrF++, and
combined = sqrt(BLEU * chrF++) for both systems on each resample, and report
the 95% percentile CI of the difference plus the bootstrap p (fraction of
resamples where the sign of the difference flips against the observed one).

The prediction CSVs carry no reference text (the corpus is not redistributed);
references are read row-aligned from the locally rebuilt independent test set,
see DATA_ACCESS.md.

Usage: uv run python experiments/scripts/paired_bootstrap.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

ART = "experiments/results/matrix_h100/artifacts"
TEST_SET = "data/processed/independent_test_set_clean.csv"
N_BOOT = 2000
SEED = 12345

bleu = BLEU(effective_order=False)
chrf = CHRF(word_order=2)


def load(run: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(f"{ART}/{run}/independent_preds.csv")
    refs = pd.read_csv(TEST_SET)
    assert len(refs) == len(df), f"{run}: {len(df)} predictions vs {len(refs)} test rows"
    if "doc_id" in df.columns:
        assert (refs["doc_id"].values == df["doc_id"].values).all(), f"{run}: row order differs from test set"
    return df["prediction"].fillna("").astype(str).tolist(), refs["translation"].astype(str).tolist()


def combined(hyps: list[str], refs: list[str]) -> float:
    b = bleu.corpus_score(hyps, [refs]).score
    c = chrf.corpus_score(hyps, [refs]).score
    return float(np.sqrt(b * c))


def pair(name: str, run_a: str, run_b: str) -> None:
    hyp_a, ref_a = load(run_a)
    hyp_b, ref_b = load(run_b)
    assert ref_a == ref_b, f"reference mismatch between {run_a} and {run_b}"
    n = len(ref_a)
    obs = combined(hyp_a, ref_a) - combined(hyp_b, ref_b)
    rng = np.random.default_rng(SEED)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        refs = [ref_a[j] for j in idx]
        deltas[i] = combined([hyp_a[j] for j in idx], refs) - combined(
            [hyp_b[j] for j in idx], refs
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float(np.mean(deltas <= 0) if obs > 0 else np.mean(deltas >= 0))
    print(
        f"{name}: delta={obs:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
        f"bootstrap p={p:.4f}  ({run_a} - {run_b})",
        flush=True,
    )


if __name__ == "__main__":
    # Family gap at 580M, both retuned, seed-matched.
    pair("family gap s42 (retuned)", "byt5_base_baseline_lr1e4", "mt5_base_lr1e3")
    pair("family gap s52 (retuned)", "byt5_base_baseline_lr1e4_seed52", "mt5_base_lr1e3_seed52")
    pair("family gap s62 (retuned)", "byt5_base_baseline_lr1e4_seed62", "mt5_base_lr1e3_seed62")
    # Family gap against the shared-recipe 7e-5 reference (env-check rerun, seed 42).
    pair("family gap s42 (7e-5 ref)", "byt5_base_baseline_envr2", "mt5_base_lr1e3")
    # Data gap: 100% vs 25%, same rate (7e-5), same environment, seed 42.
    pair("data 100% vs 25% s42", "byt5_base_baseline_envr2", "byt5_base_scale_25pct_r2")
    # Gloss at 1.2B vs its 5e-5 baseline, seed-matched.
    pair("large gloss s42", "byt5_large_rq3_dictionary_gloss_seed42", "byt5_large_baseline_lr5e5")
    pair("large gloss s52", "byt5_large_rq3_dictionary_gloss_seed52", "byt5_large_baseline_seed52")
    pair("large gloss s62", "byt5_large_rq3_dictionary_gloss_seed62", "byt5_large_baseline_seed62")
