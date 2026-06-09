"""Builders for the seven preprocessing-intervention datasets evaluated in the paper."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from akkadian_mt.data.dataset import _exclude_holdout_docs
from akkadian_mt.data.preprocessing import clean_translation, normalize_transliteration

DEFAULT_VARIANT_DIR = Path("data/processed/coursework_variants")
DEFAULT_TRAIN_FILE = Path("data/processed/golden_corpus_v2_prefilt.csv")
DEFAULT_PRIMARY_TEST_FILE = Path("data/processed/independent_test_set_clean.csv")
DEFAULT_SECONDARY_TEST_FILE = Path("data/processed/new_test_set.csv")
DEFAULT_HOLDOUT_FILE = Path("data/processed/coursework_holdout_doc_ids.json")

_GENRE_WORD_RE = re.compile(r"[^a-z0-9]+")
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(?:\.\d+|/\d+)?$")


@dataclass(frozen=True)
class VariantArtifactPaths:
    train_file: Path
    primary_test_file: Path
    secondary_test_file: Path


def canonicalize_genre(value: Any) -> str:
    """Convert a genre label into a stable lowercase tag."""
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return "unknown"
    text = _GENRE_WORD_RE.sub("_", text).strip("_")
    return text or "unknown"


def _normalize_source_text(text: Any) -> str:
    return normalize_transliteration(str(text), do_strip_homophone_subscripts=True)


def _normalize_target_text(text: Any) -> str:
    return clean_translation(str(text))


def _utf8_truncate(text: str, max_bytes: int) -> str:
    """Truncate a string to at most max_bytes when encoded as UTF-8."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore").strip()


def build_genre_conditioned_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend a genre control tag to each source sentence."""
    out = df.copy()
    base = out["transliteration"].astype(str).apply(_normalize_source_text)
    genres = out.get("genre", pd.Series(["unknown"] * len(out), index=out.index)).apply(
        canonicalize_genre
    )
    out["base_transliteration"] = base
    out["genre_tag"] = genres
    out["transliteration"] = [
        f"genre_{genre} source_text {text}".strip()
        for genre, text in zip(genres, base, strict=True)
    ]
    if "translation" in out.columns:
        out["translation"] = out["translation"].astype(str).apply(_normalize_target_text)
    return out


def _fraction_to_tag(value: Fraction) -> str | None:
    whole = value.numerator // value.denominator
    remainder = value - whole
    if whole == 0 and remainder == 0:
        return None

    parts: list[str] = []
    if whole:
        parts.append(str(whole))
    if remainder:
        parts.extend([str(remainder.numerator), str(remainder.denominator)])
    return "_".join(parts)


def normalize_numeric_token(token: str) -> str | None:
    """Return a canonical annotation tag for fractional/decimal numeric tokens."""
    if not _NUMERIC_TOKEN_RE.match(token):
        return None
    if token.isdigit():
        return None
    try:
        if "/" in token:
            fraction = Fraction(token)
        else:
            fraction = Fraction(token).limit_denominator(12)
    except (ValueError, ZeroDivisionError):
        return None

    if not math.isfinite(float(fraction)):
        return None
    if fraction.denominator > 12:
        return None
    return _fraction_to_tag(fraction)


def annotate_numerals_in_text(text: str) -> tuple[str, int]:
    """Append explicit numeral tags after decimal and fractional tokens."""
    parts: list[str] = []
    annotations = 0
    for token in text.split():
        parts.append(token)
        tag = normalize_numeric_token(token)
        if tag is not None:
            parts.append(f"num_value_{tag}")
            annotations += 1
    return " ".join(parts), annotations


def build_numeral_normalized_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate numeral tokens with canonical fractional forms."""
    out = df.copy()
    base = out["transliteration"].astype(str).apply(_normalize_source_text)
    annotated: list[str] = []
    counts: list[int] = []
    for text in base:
        new_text, count = annotate_numerals_in_text(text)
        annotated.append(new_text)
        counts.append(count)
    out["base_transliteration"] = base
    out["numeral_annotation_count"] = counts
    out["transliteration"] = annotated
    if "translation" in out.columns:
        out["translation"] = out["translation"].astype(str).apply(_normalize_target_text)
    return out


def _doc_column(df: pd.DataFrame) -> str | None:
    if "oare_id" in df.columns:
        return "oare_id"
    if "doc_id" in df.columns:
        return "doc_id"
    return None


def _pick_best_neighbor(similarities: np.ndarray, forbid_index: int | None = None) -> int:
    if forbid_index is not None:
        similarities = similarities.copy()
        similarities[forbid_index] = -1.0
    return int(similarities.argmax())


def build_retrieval_augmented_frames(
    *,
    train_df: pd.DataFrame,
    eval_dfs: dict[str, pd.DataFrame],
    holdout_doc_ids: set[str],
    analyzer: str = "char",
    ngram_min: int = 3,
    ngram_max: int = 5,
    retrieved_source_max_bytes: int = 160,
    retrieved_translation_max_bytes: int = 220,
    include_retrieved_translation: bool = True,
) -> dict[str, pd.DataFrame]:
    """Materialize retrieval-augmented train/eval frames."""
    normalized_train = train_df.copy()
    normalized_train["_row_id"] = np.arange(len(normalized_train))
    normalized_train["base_transliteration"] = normalized_train["transliteration"].apply(
        _normalize_source_text
    )
    normalized_train["translation"] = normalized_train["translation"].apply(_normalize_target_text)

    retrieval_bank = _exclude_holdout_docs(normalized_train, holdout_doc_ids).reset_index(drop=True)
    if retrieval_bank.empty:
        raise ValueError("Retrieval bank is empty after excluding holdout docs.")

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
    )
    bank_matrix = vectorizer.fit_transform(retrieval_bank["base_transliteration"].tolist())
    bank_doc_col = _doc_column(retrieval_bank)
    row_id_to_bank_index = {
        int(row_id): bank_idx
        for bank_idx, row_id in enumerate(retrieval_bank["_row_id"].astype(int).tolist())
    }

    def _format_with_retrieval(
        df: pd.DataFrame,
        best_indices: list[int],
        best_scores: list[float],
    ) -> pd.DataFrame:
        out = df.copy()
        base = out["transliteration"].astype(str).apply(_normalize_source_text)
        out["base_transliteration"] = base
        out["translation"] = out["translation"].astype(str).apply(_normalize_target_text)

        retrieved_rows = retrieval_bank.iloc[best_indices].reset_index(drop=True)
        if bank_doc_col:
            out["retrieved_doc_id"] = retrieved_rows[bank_doc_col].tolist()
        else:
            out["retrieved_doc_id"] = [str(idx) for idx in best_indices]
        out["retrieved_score"] = best_scores
        out["retrieved_transliteration"] = retrieved_rows["base_transliteration"].tolist()
        out["retrieved_translation"] = retrieved_rows["translation"].tolist()
        formatted_inputs: list[str] = []
        for src, tgt, current in zip(
            out["retrieved_transliteration"],
            out["retrieved_translation"],
            out["base_transliteration"],
            strict=True,
        ):
            segments = ["retrieved_source", _utf8_truncate(src, retrieved_source_max_bytes)]
            if include_retrieved_translation and retrieved_translation_max_bytes > 0:
                segments.extend(
                    [
                        "retrieved_translation",
                        _utf8_truncate(tgt, retrieved_translation_max_bytes),
                    ]
                )
            segments.extend(["current_source", current])
            formatted_inputs.append(" ".join(segment for segment in segments if segment).strip())
        out["transliteration"] = formatted_inputs
        if "_row_id" in out.columns:
            out = out.drop(columns=["_row_id"])
        return out

    train_best_indices: list[int] = []
    train_best_scores: list[float] = []
    train_base = normalized_train["base_transliteration"].tolist()

    for idx, text in enumerate(train_base):
        row = vectorizer.transform([text])
        sims = (row @ bank_matrix.T).toarray().ravel()
        original_row_id = int(normalized_train.iloc[idx]["_row_id"])
        forbid_index = row_id_to_bank_index.get(original_row_id)
        best_index = _pick_best_neighbor(sims, forbid_index=forbid_index)
        train_best_indices.append(best_index)
        train_best_scores.append(float(sims[best_index]))

    result = {
        "train": _format_with_retrieval(normalized_train, train_best_indices, train_best_scores)
    }

    for split_name, eval_df in eval_dfs.items():
        normalized_eval = eval_df.copy()
        eval_base = normalized_eval["transliteration"].astype(str).apply(_normalize_source_text)
        eval_matrix = vectorizer.transform(eval_base.tolist())
        sims = eval_matrix @ bank_matrix.T
        best_indices = sims.argmax(axis=1).A1.tolist()
        best_scores = [
            float(sims[row_idx, best_idx]) for row_idx, best_idx in enumerate(best_indices)
        ]
        normalized_eval["transliteration"] = eval_base
        result[split_name] = _format_with_retrieval(normalized_eval, best_indices, best_scores)

    return result


def read_holdout_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    holdout_path = Path(path)
    if not holdout_path.exists():
        raise FileNotFoundError(
            "Missing holdout file: "
            f"{holdout_path}. Run experiments/scripts/build_holdout_union.py first."
        )
    payload = json.loads(holdout_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Holdout file must contain a JSON list: {holdout_path}")
    return {str(item) for item in payload if str(item).strip()}


def build_all_variants(
    *,
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    primary_test_file: str | Path = DEFAULT_PRIMARY_TEST_FILE,
    secondary_test_file: str | Path = DEFAULT_SECONDARY_TEST_FILE,
    holdout_file: str | Path = DEFAULT_HOLDOUT_FILE,
    output_dir: str | Path = DEFAULT_VARIANT_DIR,
) -> dict[str, VariantArtifactPaths]:
    """Build all preprocessing-variant datasets and return their file locations."""
    train_path = Path(train_file)
    primary_path = Path(primary_test_file)
    secondary_path = Path(secondary_test_file)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    primary_df = pd.read_csv(primary_path)
    secondary_df = pd.read_csv(secondary_path)
    holdout_ids = read_holdout_ids(holdout_file)

    variants: dict[str, dict[str, pd.DataFrame]] = {
        "genre_conditioned": {
            "train": build_genre_conditioned_frame(train_df),
            "primary": build_genre_conditioned_frame(primary_df),
            "secondary": build_genre_conditioned_frame(secondary_df),
        },
        "numeral_normalized": {
            "train": build_numeral_normalized_frame(train_df),
            "primary": build_numeral_normalized_frame(primary_df),
            "secondary": build_numeral_normalized_frame(secondary_df),
        },
    }
    variants["retrieval_augmented"] = build_retrieval_augmented_frames(
        train_df=train_df,
        eval_dfs={"primary": primary_df, "secondary": secondary_df},
        holdout_doc_ids=holdout_ids,
    )
    variants["retrieval_compact"] = build_retrieval_augmented_frames(
        train_df=train_df,
        eval_dfs={"primary": primary_df, "secondary": secondary_df},
        holdout_doc_ids=holdout_ids,
        retrieved_source_max_bytes=96,
        retrieved_translation_max_bytes=96,
    )
    variants["retrieval_source_only"] = build_retrieval_augmented_frames(
        train_df=train_df,
        eval_dfs={"primary": primary_df, "secondary": secondary_df},
        holdout_doc_ids=holdout_ids,
        retrieved_source_max_bytes=160,
        retrieved_translation_max_bytes=0,
        include_retrieved_translation=False,
    )

    manifests: dict[str, Any] = {}
    artifact_paths: dict[str, VariantArtifactPaths] = {}

    for variant_name, frames in variants.items():
        variant_dir = output_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        train_out = variant_dir / "train.csv"
        primary_out = variant_dir / "independent_test.csv"
        secondary_out = variant_dir / "new_test.csv"
        frames["train"].to_csv(train_out, index=False)
        frames["primary"].to_csv(primary_out, index=False)
        frames["secondary"].to_csv(secondary_out, index=False)

        artifact_paths[variant_name] = VariantArtifactPaths(
            train_file=train_out,
            primary_test_file=primary_out,
            secondary_test_file=secondary_out,
        )
        manifests[variant_name] = {
            "train_rows": int(len(frames["train"])),
            "primary_rows": int(len(frames["primary"])),
            "secondary_rows": int(len(frames["secondary"])),
            "train_source_bytes_p95": int(
                np.percentile(
                    [len(str(text).encode("utf-8")) for text in frames["train"]["transliteration"]],
                    95,
                )
            ),
            "train_source_bytes_max": int(
                max(len(str(text).encode("utf-8")) for text in frames["train"]["transliteration"])
            ),
            "files": {
                "train": str(train_out),
                "primary": str(primary_out),
                "secondary": str(secondary_out),
            },
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return artifact_paths
