#!/usr/bin/env python
"""Select illustrative translation examples for qualitative analysis.

Loads predictions from key experiments, computes per-sentence BLEU and chrF++,
and selects examples that illustrate the main findings:
  1. Tokenisation gap (mT5 vs ByT5)
  2. Capacity x preprocessing interaction
  3. Genre contrast (formulaic vs complex)
  4. Near-miss / characteristic errors
  5. Data scaling contrast

Outputs:
  - coursework/results/qualitative_examples.csv
  - LaTeX-formatted table to stdout
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import sacrebleu

ARTIFACTS = Path("coursework/results/artifacts")

# Experiments to load
EXPERIMENTS = {
    "mt5_small": "mt5_small_baseline",
    "mt5_base": "mt5_base_baseline",
    "byt5_small": "byt5_small_baseline",
    "byt5_base": "byt5_base_baseline",
    "byt5_small_genre": "byt5_small_cw4_genre_conditioned",
    "byt5_base_genre": "byt5_base_cw4_genre_conditioned",
    "byt5_base_25pct": "byt5_base_scale_25pct",
}


def sentence_bleu(pred: str, ref: str) -> float:
    """Compute sentence-level BLEU."""
    return sacrebleu.sentence_bleu(pred, [ref]).score


def sentence_chrf(pred: str, ref: str) -> float:
    """Compute sentence-level chrF++."""
    return sacrebleu.sentence_chrf(pred, [ref], word_order=2).score


def load_experiment(name: str) -> pd.DataFrame:
    """Load predictions CSV for an experiment."""
    path = ARTIFACTS / name / "independent_preds.csv"
    df = pd.read_csv(path)
    return df


def main() -> None:
    # Load all experiments
    dfs = {}
    for key, exp_name in EXPERIMENTS.items():
        df = load_experiment(exp_name)
        dfs[key] = df

    # Use byt5_base as the anchor: join on (doc_id, sentence_idx)
    anchor = dfs["byt5_base"][
        ["transliteration", "translation", "doc_id", "genre", "sentence_idx"]
    ].copy()

    # Merge predictions from each experiment
    for key, df in dfs.items():
        pred_col = f"pred_{key}"
        merge_df = df[["doc_id", "sentence_idx", "prediction"]].rename(
            columns={"prediction": pred_col}
        )
        anchor = anchor.merge(merge_df, on=["doc_id", "sentence_idx"], how="inner")

    print(f"Joined {len(anchor)} sentences across {len(EXPERIMENTS)} experiments")

    # Compute per-sentence scores for key models
    score_models = [
        "mt5_base",
        "byt5_base",
        "byt5_small",
        "byt5_small_genre",
        "byt5_base_genre",
        "byt5_base_25pct",
    ]
    for model in score_models:
        pred_col = f"pred_{model}"
        bleu_col = f"bleu_{model}"
        chrf_col = f"chrf_{model}"
        anchor[bleu_col] = anchor.apply(
            lambda r: sentence_bleu(str(r[pred_col]), str(r["translation"])), axis=1
        )
        anchor[chrf_col] = anchor.apply(
            lambda r: sentence_chrf(str(r[pred_col]), str(r["translation"])), axis=1
        )

    # Filter to medium-length sentences (readable in a table)
    src_len = anchor["transliteration"].str.len()
    ref_len = anchor["translation"].str.len()
    medium = anchor[(src_len >= 20) & (src_len <= 200) & (ref_len >= 15) & (ref_len <= 200)]
    print(f"Medium-length candidates: {len(medium)}")

    examples = []

    # ── Category 1: Tokenisation gap ──
    # ByT5-base high chrF, mT5-base very low chrF
    cat1 = medium.copy()
    cat1["gap"] = cat1["chrf_byt5_base"] - cat1["chrf_mt5_base"]
    cat1 = cat1[cat1["chrf_byt5_base"] > 40]  # ByT5 does reasonably well
    cat1 = cat1.sort_values("gap", ascending=False)
    print("\n=== TOKENISATION GAP (ByT5-base >> mT5-base) ===")
    for _, row in cat1.head(8).iterrows():
        print(f"\n  SRC:  {row['transliteration'][:120]}")
        print(f"  REF:  {row['translation'][:120]}")
        print(f"  mT5:  {row['pred_mt5_base'][:120]}  [chrF={row['chrf_mt5_base']:.1f}]")
        print(f"  ByT5: {row['pred_byt5_base'][:120]}  [chrF={row['chrf_byt5_base']:.1f}]")
        print(f"  Gap:  {row['gap']:.1f} chrF++ points")
    if len(cat1) > 0:
        best = cat1.iloc[0]
        examples.append(
            {
                "category": "tokenisation_gap",
                "doc_id": best["doc_id"],
                "sentence_idx": best["sentence_idx"],
                "genre": best["genre"],
            }
        )

    # ── Category 2: Capacity x preprocessing interaction ──
    # Genre conditioning helps small but hurts base
    cat2 = medium.copy()
    cat2["small_gain"] = cat2["chrf_byt5_small_genre"] - cat2["chrf_byt5_small"]
    cat2["base_loss"] = cat2["chrf_byt5_base_genre"] - cat2["chrf_byt5_base"]
    cat2["interaction"] = cat2["small_gain"] - cat2["base_loss"]
    # small_gain > 0 and base_loss < 0
    cat2 = cat2[(cat2["small_gain"] > 5) & (cat2["base_loss"] < -5)]
    cat2 = cat2.sort_values("interaction", ascending=False)
    print("\n=== CAPACITY x PREPROCESSING INTERACTION ===")
    for _, row in cat2.head(8).iterrows():
        print(f"\n  SRC:  {row['transliteration'][:120]}")
        print(f"  REF:  {row['translation'][:120]}")
        print(f"  Small baseline:       chrF={row['chrf_byt5_small']:.1f}")
        print(
            f"  Small +genre:         chrF={row['chrf_byt5_small_genre']:.1f}  (Δ={row['small_gain']:+.1f})"
        )
        print(f"  Base baseline:        chrF={row['chrf_byt5_base']:.1f}")
        print(
            f"  Base +genre:          chrF={row['chrf_byt5_base_genre']:.1f}  (Δ={row['base_loss']:+.1f})"
        )
    if len(cat2) > 0:
        best = cat2.iloc[0]
        examples.append(
            {
                "category": "capacity_x_preprocessing",
                "doc_id": best["doc_id"],
                "sentence_idx": best["sentence_idx"],
                "genre": best["genre"],
            }
        )

    # ── Category 3a: Formulaic success (debt note / witness list) ──
    cat3a = medium[medium["genre"].isin(["debt-note", "memo"])].copy()
    cat3a = cat3a.sort_values("chrf_byt5_base", ascending=False)
    print("\n=== FORMULAIC SUCCESS (debt notes / memos) ===")
    for _, row in cat3a.head(8).iterrows():
        print(f"\n  Genre: {row['genre']}")
        print(f"  SRC:  {row['transliteration'][:120]}")
        print(f"  REF:  {row['translation'][:120]}")
        print(f"  ByT5: {row['pred_byt5_base'][:120]}  [chrF={row['chrf_byt5_base']:.1f}]")
    if len(cat3a) > 0:
        best = cat3a.iloc[0]
        examples.append(
            {
                "category": "formulaic_success",
                "doc_id": best["doc_id"],
                "sentence_idx": best["sentence_idx"],
                "genre": best["genre"],
            }
        )

    # ── Category 3b: Complex failure (letter with proper names) ──
    cat3b = medium[medium["genre"] == "letter"].copy()
    cat3b = cat3b[(cat3b["chrf_byt5_base"] > 15) & (cat3b["chrf_byt5_base"] < 40)]
    cat3b = cat3b.sort_values("chrf_byt5_base", ascending=True)
    print("\n=== COMPLEX FAILURE (letters) ===")
    for _, row in cat3b.head(8).iterrows():
        print(f"\n  SRC:  {row['transliteration'][:120]}")
        print(f"  REF:  {row['translation'][:120]}")
        print(f"  ByT5: {row['pred_byt5_base'][:120]}  [chrF={row['chrf_byt5_base']:.1f}]")
    if len(cat3b) > 0:
        best = cat3b.iloc[0]
        examples.append(
            {
                "category": "complex_failure",
                "doc_id": best["doc_id"],
                "sentence_idx": best["sentence_idx"],
                "genre": best["genre"],
            }
        )

    # ── Category 5: Data scaling contrast ──
    cat5 = medium.copy()
    cat5["scale_gap"] = cat5["chrf_byt5_base"] - cat5["chrf_byt5_base_25pct"]
    cat5 = cat5[cat5["chrf_byt5_base"] > 50]  # full model does well
    cat5 = cat5.sort_values("scale_gap", ascending=False)
    print("\n=== DATA SCALING (100% >> 25%) ===")
    for _, row in cat5.head(8).iterrows():
        print(f"\n  SRC:  {row['transliteration'][:120]}")
        print(f"  REF:  {row['translation'][:120]}")
        print(
            f"  25%:  {row['pred_byt5_base_25pct'][:120]}  [chrF={row['chrf_byt5_base_25pct']:.1f}]"
        )
        print(f"  100%: {row['pred_byt5_base'][:120]}  [chrF={row['chrf_byt5_base']:.1f}]")
        print(f"  Gap:  {row['scale_gap']:.1f}")
    if len(cat5) > 0:
        best = cat5.iloc[0]
        examples.append(
            {
                "category": "data_scaling",
                "doc_id": best["doc_id"],
                "sentence_idx": best["sentence_idx"],
                "genre": best["genre"],
            }
        )

    # ── Save selected examples ──
    print(f"\n\nSelected {len(examples)} example categories")

    # Build output with full details
    output_rows = []
    for ex in examples:
        row = anchor[
            (anchor["doc_id"] == ex["doc_id"]) & (anchor["sentence_idx"] == ex["sentence_idx"])
        ].iloc[0]
        out = {
            "category": ex["category"],
            "genre": row["genre"],
            "transliteration": row["transliteration"],
            "reference": row["translation"],
        }
        for key in EXPERIMENTS:
            out[f"pred_{key}"] = row[f"pred_{key}"]
        for model in score_models:
            out[f"chrf_{model}"] = row[f"chrf_{model}"]
            out[f"bleu_{model}"] = row[f"bleu_{model}"]
        output_rows.append(out)

    out_path = Path("coursework/results/qualitative_examples.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved to {out_path}")

    # ── Print LaTeX-ready format ──
    print("\n\n" + "=" * 70)
    print("LATEX-READY EXAMPLES")
    print("=" * 70)
    for _, row in out_df.iterrows():
        print(f"\n% Category: {row['category']} | Genre: {row['genre']}")
        print(f"% Source: {row['transliteration']}")
        print(f"% Reference: {row['reference']}")
        for key in EXPERIMENTS:
            col = f"pred_{key}"
            if col in row:
                print(f"% {key}: {row[col]}")
        chrf_cols = [c for c in row.index if c.startswith("chrf_")]
        for c in chrf_cols:
            print(f"% {c}: {row[c]:.1f}")


if __name__ == "__main__":
    main()
