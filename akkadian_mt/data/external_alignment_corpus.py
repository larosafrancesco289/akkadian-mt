"""Build a cleaned external alignment corpus and optional mixed training CSV."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from akkadian_mt.data.extract_publication_translations import detect_language
from akkadian_mt.data.preprocessing import META_RE, clean_translation, normalize_transliteration
from akkadian_mt.data.programmatic_corpus import (
    STRICT_FOREIGN_TARGET_LANGS,
    dedupe_parallel_pairs,
    english_hint_hits,
    filter_parallel_text_quality,
    foreign_stopword_hits,
)

ELLIPSIS_RE = re.compile(r"\.\.\.|…|\[\.\.\.")
BRACKET_RE = re.compile(r"[\[\]{}]")
OCR_NOISE_RE = re.compile(r"[መሩሠၣ]")
BAD_TERMINAL_RE = re.compile(r"[;,]\s*$")
PLACEHOLDER_X_RE = re.compile(r"\bx\b")
SUSPICIOUS_SCRIPT_RE = re.compile(r"[\u0700-\u074F\u1200-\u137F\u1000-\u109F]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default="phucthaiv02/akkadian_english_sentences_alignment_2",
        help="Hugging Face dataset repo ID",
    )
    parser.add_argument(
        "--dataset-file",
        action="append",
        default=None,
        help="Dataset file inside the HF repo; repeat to include multiple splits",
    )
    parser.add_argument(
        "--stage2-corpus",
        default="data/processed/programmatic_v1/stage2_finetune_evalsafe.csv",
        help="Primary local corpus used to exclude seen sources",
    )
    parser.add_argument(
        "--exclude-corpus",
        action="append",
        default=[
            "data/processed/new_test_set.csv",
            "data/processed/independent_test_set_clean.csv",
        ],
        help="Additional local corpora whose normalized sources should be excluded",
    )
    parser.add_argument(
        "--base-corpus",
        default="data/processed/golden_corpus_v2_prefilt.csv",
        help="Optional baseline corpus to append the cleaned external rows to",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/external_alignment_phucthaiv_clean_v1.csv",
        help="Cleaned external-only CSV output",
    )
    parser.add_argument(
        "--output-summary",
        default="data/processed/external_alignment_phucthaiv_clean_v1_summary.json",
        help="JSON summary output",
    )
    parser.add_argument(
        "--output-mixed-csv",
        default="data/processed/external_alignment_mix_v1.csv",
        help="Optional mixed base+external CSV output",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Optional source label to stamp onto kept rows",
    )
    parser.add_argument(
        "--oare-prefix",
        default=None,
        help="Optional oare_id prefix; defaults to the resolved source label",
    )
    parser.add_argument(
        "--pdf-include-pattern",
        default=None,
        help="Optional regex; keep only rows whose pdf_name matches",
    )
    parser.add_argument(
        "--pdf-exclude-pattern",
        default=None,
        help="Optional regex; drop rows whose pdf_name matches",
    )
    parser.add_argument(
        "--translation-include-pattern",
        default=None,
        help="Optional regex; keep only rows whose cleaned translation matches",
    )
    parser.add_argument(
        "--translation-exclude-pattern",
        default=None,
        help="Optional regex; drop rows whose cleaned translation matches",
    )
    parser.add_argument("--min-source-words", type=int, default=3)
    parser.add_argument("--min-target-words", type=int, default=5)
    parser.add_argument("--max-target-words", type=int, default=35)
    parser.add_argument("--min-translation-chars", type=int, default=10)
    parser.add_argument("--max-length-ratio", type=float, default=10.0)
    parser.add_argument(
        "--allow-seen-sources",
        action="store_true",
        help="Keep rows whose normalized source already appears in the local corpora",
    )
    return parser


def _strip_control_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def _normalize_source(text: str) -> str:
    return normalize_transliteration(
        _strip_control_chars(text),
        full_normalize=True,
        do_strip_homophone_subscripts=True,
        do_remove_gaps=True,
    )


def _normalize_target(text: str) -> str:
    return clean_translation(_strip_control_chars(text))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "external_alignment"


def _resolve_source_name(args: argparse.Namespace) -> str:
    if args.source_name:
        return _slugify(args.source_name)
    source_name = f"external_alignment_{_slugify(args.dataset_id)}"
    filter_spec = json.dumps(
        {
            "include": args.pdf_include_pattern,
            "exclude": args.pdf_exclude_pattern,
        },
        sort_keys=True,
    )
    if args.pdf_include_pattern or args.pdf_exclude_pattern:
        digest = hashlib.sha1(filter_spec.encode("utf-8")).hexdigest()[:8]
        source_name = f"{source_name}_{digest}"
    return source_name


def _load_seen_sources(paths: list[str]) -> tuple[set[str], set[tuple[str, str]]]:
    seen_sources: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if "transliteration" not in df.columns or "translation" not in df.columns:
            continue
        src = df["transliteration"].astype(str).map(_normalize_source)
        tgt = df["translation"].astype(str).map(_normalize_target)
        seen_sources.update(src[src.ne("")].tolist())
        seen_pairs.update((s, t) for s, t in zip(src, tgt, strict=False) if s and t)
    return seen_sources, seen_pairs


def _apply_english_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["detected_language"] = out["translation"].astype(str).apply(detect_language)
    out["english_hint_hits"] = out["translation"].astype(str).apply(english_hint_hits)
    out["foreign_stopword_hits"] = [
        foreign_stopword_hits(text, language)
        for text, language in zip(
            out["translation"].astype(str),
            out["detected_language"].astype(str),
            strict=False,
        )
    ]
    drop_mask = (
        (out["detected_language"].eq("de") & out["foreign_stopword_hits"].ge(1))
        | (out["detected_language"].eq("de") & out["english_hint_hits"].lt(2))
        | (
            out["detected_language"].isin(STRICT_FOREIGN_TARGET_LANGS)
            & out["foreign_stopword_hits"].ge(1)
        )
        | (
            out["detected_language"].isin(STRICT_FOREIGN_TARGET_LANGS)
            & out["english_hint_hits"].eq(0)
        )
    )
    return out.loc[~drop_mask].reset_index(drop=True)


def build_external_df(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    source_name = _resolve_source_name(args)
    oare_prefix = _slugify(args.oare_prefix) if args.oare_prefix else source_name
    dataset_files = args.dataset_file or ["data/train-00000-of-00001.parquet"]
    raw_parts: list[pd.DataFrame] = []
    for dataset_file in dataset_files:
        parquet_path = hf_hub_download(
            repo_id=args.dataset_id,
            filename=dataset_file,
            repo_type="dataset",
        )
        part = pd.read_parquet(parquet_path)
        part["_dataset_file"] = dataset_file
        raw_parts.append(part)
    raw = pd.concat(raw_parts, ignore_index=True, sort=False)
    raw = raw.rename(
        columns={"transliteration": "transliteration_raw", "translation": "translation_raw"}
    )

    df = raw.copy()
    df["transliteration"] = df["transliteration_raw"].astype(str).map(_normalize_source)
    df["translation"] = df["translation_raw"].astype(str).map(_normalize_target)
    df["control_chars_removed"] = (
        df["transliteration_raw"].astype(str) != df["transliteration"].astype(str)
    ) | (df["translation_raw"].astype(str) != df["translation"].astype(str))
    df = df[
        df["transliteration"].astype(str).str.strip().ne("")
        & df["translation"].astype(str).str.strip().ne("")
    ].copy()

    if args.pdf_include_pattern and "pdf_name" in df.columns:
        df = df[
            df["pdf_name"]
            .fillna("")
            .astype(str)
            .str.contains(args.pdf_include_pattern, case=False, regex=True)
        ].copy()
    if args.pdf_exclude_pattern and "pdf_name" in df.columns:
        df = df[
            ~df["pdf_name"]
            .fillna("")
            .astype(str)
            .str.contains(args.pdf_exclude_pattern, case=False, regex=True)
        ].copy()
    if args.translation_include_pattern:
        df = df[
            df["translation"]
            .fillna("")
            .astype(str)
            .str.contains(args.translation_include_pattern, case=False, regex=True)
        ].copy()
    if args.translation_exclude_pattern:
        df = df[
            ~df["translation"]
            .fillna("")
            .astype(str)
            .str.contains(args.translation_exclude_pattern, case=False, regex=True)
        ].copy()

    df["source_word_count"] = df["transliteration"].astype(str).str.split().str.len()
    df["target_word_count"] = df["translation"].astype(str).str.split().str.len()
    df["has_ellipsis"] = df["translation"].astype(str).str.contains(ELLIPSIS_RE)
    df["has_question"] = df["translation"].astype(str).str.contains(r"\?")
    df["has_brackets"] = df["translation"].astype(str).str.contains(BRACKET_RE)
    df["meta_like"] = df["translation"].astype(str).apply(lambda text: bool(META_RE.search(text)))
    df["starts_lower"] = df["translation"].astype(str).str.contains(r"^[a-z]", regex=True)
    df["bad_terminal"] = df["translation"].astype(str).str.contains(BAD_TERMINAL_RE)
    df["placeholder_x"] = df["translation"].astype(str).str.contains(PLACEHOLDER_X_RE)
    df["ocr_noise"] = df["transliteration"].astype(str).str.contains(OCR_NOISE_RE) | df[
        "translation"
    ].astype(str).str.contains(OCR_NOISE_RE)
    df["suspicious_script"] = df["transliteration"].astype(str).apply(
        lambda text: bool(SUSPICIOUS_SCRIPT_RE.search(text))
    ) | df["translation"].astype(str).apply(lambda text: bool(SUSPICIOUS_SCRIPT_RE.search(text)))

    df = _apply_english_filter(df)

    seen_paths = [args.stage2_corpus, *args.exclude_corpus]
    seen_sources, seen_pairs = _load_seen_sources(seen_paths)
    df["seen_source"] = df["transliteration"].isin(seen_sources)
    df["seen_pair"] = [
        (src, tgt) in seen_pairs
        for src, tgt in zip(df["transliteration"], df["translation"], strict=False)
    ]

    keep_mask = (
        df["source_word_count"].ge(args.min_source_words)
        & df["target_word_count"].ge(args.min_target_words)
        & df["target_word_count"].le(args.max_target_words)
        & ~df["has_ellipsis"]
        & ~df["has_question"]
        & ~df["has_brackets"]
        & ~df["meta_like"]
        & ~df["starts_lower"]
        & ~df["bad_terminal"]
        & ~df["placeholder_x"]
        & ~df["ocr_noise"]
        & ~df["suspicious_script"]
    )
    if not args.allow_seen_sources:
        keep_mask &= ~df["seen_source"]

    df = df.loc[keep_mask].copy()
    df, _dropped = filter_parallel_text_quality(
        df,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=args.min_source_words,
    )

    df["source"] = source_name
    df["granularity"] = "sentence"
    df["tier"] = "external_alignment"
    df["confidence"] = 0.70
    df["quality_flags"] = ""

    df["_target_len"] = df["translation"].astype(str).str.len()
    df = (
        df.sort_values(
            ["seen_pair", "target_word_count", "_target_len", "english_hint_hits"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(subset=["transliteration"], keep="first")
        .drop_duplicates(subset=["transliteration", "translation"], keep="first")
        .drop(columns=["_target_len"])
        .reset_index(drop=True)
    )
    df["oare_id"] = [f"{oare_prefix}::{idx:05d}" for idx in range(len(df))]

    out_cols = [
        "oare_id",
        "transliteration",
        "translation",
        "pdf_name",
        "page",
        "source",
        "granularity",
        "tier",
        "confidence",
        "quality_flags",
        "detected_language",
        "english_hint_hits",
        "foreign_stopword_hits",
        "source_word_count",
        "target_word_count",
        "control_chars_removed",
        "seen_source",
        "seen_pair",
    ]
    df = df[[col for col in out_cols if col in df.columns]].copy()
    df = dedupe_parallel_pairs(df)

    summary = {
        "dataset_id": args.dataset_id,
        "dataset_files": dataset_files,
        "source_name": source_name,
        "oare_prefix": oare_prefix,
        "pdf_include_pattern": args.pdf_include_pattern,
        "pdf_exclude_pattern": args.pdf_exclude_pattern,
        "translation_include_pattern": args.translation_include_pattern,
        "translation_exclude_pattern": args.translation_exclude_pattern,
        "rows_raw": int(len(raw)),
        "rows_cleaned": int(len(df)),
        "unique_sources": int(df["transliteration"].nunique()),
        "seen_source_excluded": not args.allow_seen_sources,
        "rows_seen_pair_still_present": int(df["seen_pair"].sum()) if "seen_pair" in df else 0,
        "rows_control_chars_removed": int(df["control_chars_removed"].sum())
        if "control_chars_removed" in df
        else 0,
        "avg_source_words": float(df["source_word_count"].mean()) if not df.empty else 0.0,
        "avg_target_words": float(df["target_word_count"].mean()) if not df.empty else 0.0,
    }
    return df, summary


def build_mixed_df(
    base_path: Path, external_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    base_df = pd.read_csv(base_path, low_memory=False)
    if "source" not in base_df.columns:
        base_df["source"] = "base_corpus"
    if "granularity" not in base_df.columns:
        base_df["granularity"] = "sentence"
    if "tier" not in base_df.columns:
        base_df["tier"] = "gold"
    if "confidence" not in base_df.columns:
        base_df["confidence"] = 1.0
    if "quality_flags" not in base_df.columns:
        base_df["quality_flags"] = ""

    combined = pd.concat([base_df, external_df], ignore_index=True, sort=False)
    combined["_source_rank"] = 2
    combined.loc[
        combined["source"].isin(
            {"golden_corpus_v2_prefilt", "train_sentence", "published_texts_sentence"}
        ),
        "_source_rank",
    ] = 0
    if "tier" in combined.columns:
        combined.loc[combined["tier"].eq("external_alignment"), "_source_rank"] = 1
    combined = (
        combined.sort_values(["_source_rank"])
        .drop_duplicates(subset=["transliteration", "translation"], keep="first")
        .drop(columns=["_source_rank"])
        .reset_index(drop=True)
    )
    summary = {
        "base_rows": int(len(base_df)),
        "external_rows": int(len(external_df)),
        "mixed_rows": int(len(combined)),
    }
    return combined, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    external_df, summary = build_external_df(args)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    external_df.to_csv(output_csv, index=False)

    if args.base_corpus:
        mixed_df, mix_summary = build_mixed_df(Path(args.base_corpus), external_df)
        mixed_path = Path(args.output_mixed_csv)
        mixed_path.parent.mkdir(parents=True, exist_ok=True)
        mixed_df.to_csv(mixed_path, index=False)
        summary["mixed"] = mix_summary

    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"External corpus: {output_csv}")
    if args.base_corpus:
        print(f"Mixed corpus: {args.output_mixed_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
