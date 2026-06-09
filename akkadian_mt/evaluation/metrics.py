"""Evaluation metrics: BLEU, chrF++, combined score, and MBR decoding via sacrebleu."""

from __future__ import annotations

import math

import sacrebleu


def mbr_pick(candidates: list[str], pool_cap: int = 32) -> str:
    """Pick the candidate with highest average chrF++ against all others (MBR).

    Args:
        candidates: List of candidate translations (may contain duplicates/empty).
        pool_cap: Maximum number of unique candidates to consider.

    Returns:
        Best candidate string (empty string if no valid candidates).
    """
    unique: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]

    unique = unique[:pool_cap]
    n = len(unique)
    best_i, best_score = 0, -1e9
    for i, hyp in enumerate(unique):
        score = sum(
            sacrebleu.sentence_chrf(hyp, [unique[j]], word_order=2).score
            for j in range(n)
            if j != i
        ) / max(1, n - 1)
        if score > best_score:
            best_score, best_i = score, i
    return unique[best_i]


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score (micro-averaged).

    Args:
        predictions: List of predicted translations.
        references: List of reference translations.

    Returns:
        BLEU score (0-100 scale).
    """
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score


def compute_chrf(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level chrF++ score (micro-averaged).

    Args:
        predictions: List of predicted translations.
        references: List of reference translations.

    Returns:
        chrF++ score (0-100 scale).
    """
    result = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return result.score


def compute_combined_score(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute the Kaggle competition metric: geometric mean of BLEU and chrF++.

    The official metric is sqrt(BLEU * chrF++), with each score's sufficient
    statistics aggregated across the entire corpus (micro-average).

    Reference: https://www.kaggle.com/code/metric/dpi-bleu-chrf

    Args:
        predictions: List of predicted translations.
        references: List of reference translations.

    Returns:
        Dictionary with 'bleu', 'chrf', and 'combined' scores.
    """
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    combined = math.sqrt(bleu * chrf)
    return {"bleu": bleu, "chrf": chrf, "combined": combined}
