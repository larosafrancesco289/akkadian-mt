"""Build programmatic training corpora with tiered confidence/granularity.

This pipeline is intentionally data-quality first:

- Gold: repaired `Sentences_Oare` sentence pairs from train/published_texts
- Aux: document-level train pairs for docs without sentence anchors
- Experimental: publication snippets aligned back to known sentence spans
- Pretrain: domain-filtered Akkademia/ORACC pairs

It emits both full and eval-safe variants so local model selection does not
require the same corpus as final Kaggle training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from akkadian_mt.data.preprocessing import META_RE, clean_translation, normalize_transliteration
from akkadian_mt.data.sentence_extractor import _validate_anchor

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_CANDIDATES = [
    Path("data/processed/new_holdout_doc_ids.json"),
    Path("data/processed/holdout_doc_ids.json"),
]
BAD_PUBLICATION_FLAGS_RE = re.compile(
    r"citation_like|scholarly_like|meta_like|definition_like|philology_like|"
    r"apparatus_like|transliteration_like|contains_newlines"
)
WORD_RE = re.compile(r"[A-Za-z']+")
EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "at",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "it",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "his",
    "her",
    "their",
    "our",
    "your",
    "as",
    "or",
    "from",
}
ENGLISH_TARGET_HINTS = frozenset(
    {
        "the",
        "and",
        "to",
        "from",
        "of",
        "in",
        "for",
        "with",
        "on",
        "at",
        "by",
        "this",
        "that",
        "is",
        "are",
        "was",
        "were",
        "be",
        "not",
        "have",
        "has",
        "had",
        "will",
        "shall",
        "said",
        "say",
        "send",
        "give",
        "take",
        "pay",
        "silver",
        "mina",
        "minas",
        "shekel",
        "shekels",
        "witness",
        "witnesses",
        "son",
        "seal",
        "seals",
        "month",
        "year",
        "eponymate",
        "owes",
        "interest",
        "textile",
        "textiles",
        "your",
        "goods",
        "gold",
        "copper",
        "tin",
        "debt",
        "loan",
        "dispatch",
        "here",
        "died",
        "seized",
        "donkeys",
        "money",
        "after",
        "before",
        "let",
        "message",
    }
)
FOREIGN_TARGET_STOPWORDS = {
    "de": frozenset(
        {
            "aber",
            "als",
            "auch",
            "dann",
            "das",
            "den",
            "der",
            "des",
            "die",
            "doch",
            "dort",
            "ein",
            "eine",
            "einer",
            "einem",
            "einen",
            "er",
            "es",
            "für",
            "gegeben",
            "hat",
            "hier",
            "ich",
            "ihnen",
            "ihre",
            "ist",
            "mich",
            "nicht",
            "sein",
            "so",
            "und",
            "was",
            "wer",
            "wir",
        }
    ),
    "fr": frozenset(
        {"avec", "des", "elle", "est", "et", "les", "pas", "pour", "que", "une", "votre"}
    ),
    "es": frozenset({"con", "del", "ella", "ellos", "esta", "para", "pero", "por", "que", "una"}),
    "it": frozenset({"che", "con", "del", "della", "gli", "non", "per", "sono", "una"}),
    "pt": frozenset({"com", "das", "dos", "não", "para", "que", "seu", "sua", "uma"}),
    "nl": frozenset({"een", "het", "met", "niet", "van", "voor", "zij"}),
    "tr": frozenset({"ama", "bir", "bu", "çok", "için", "ile", "ve"}),
    "ro": frozenset({"care", "din", "este", "pentru", "sunt", "și"}),
    "ca": frozenset({"amb", "aquesta", "els", "les", "per", "que", "una"}),
}
STRICT_FOREIGN_TARGET_LANGS = frozenset({"fr", "tr", "it", "pt", "nl", "es", "ro", "ca"})
ORACC_COMMERCE_TARGET_KEYWORDS = frozenset(
    {
        "silver",
        "mina",
        "minas",
        "shekel",
        "shekels",
        "tin",
        "copper",
        "textile",
        "textiles",
        "merchant",
        "merchants",
        "loan",
        "interest",
        "pay",
        "pays",
        "paid",
        "owes",
        "debt",
        "donkey",
        "donkeys",
        "tablet",
        "month",
        "eponymate",
        "colony",
        "goods",
        "gold",
        "talent",
        "talents",
        "send",
        "sent",
        "brought",
        "price",
        "money",
        "field",
        "fields",
        "horses",
    }
)
ORACC_OFFDOMAIN_TARGET_KEYWORDS = frozenset(
    {
        "king",
        "palace",
        "battle",
        "enemy",
        "enemies",
        "troops",
        "temple",
        "mountain",
        "mountains",
        "heaven",
        "netherworld",
        "lung",
        "liver",
        "omen",
        "omens",
        "god",
        "gods",
        "goddess",
        "goddesses",
        "chariot",
    }
)
ORACC_COMMERCE_SOURCE_KEYWORDS = frozenset(
    {
        "kù.babbar",
        "ma-na",
        "gín",
        "an.na",
        "urudu",
        "ṣí-ib-tám",
        "dam.gàr",
        "túg",
        "túg.hi.a",
        "anše",
        "ṭup-pá-am",
        "ṭup-pì-im",
        "kišib",
        "itu.kam",
        "itu.1.kam",
        "li-mu-um",
        "kà-ri-im",
        "kà-ru-um",
        "kà-ni-iš",
        "kù.gi",
        "hu-bu-ul",
    }
)


@dataclass(frozen=True)
class BundlePaths:
    sentence_gold_full: Path
    sentence_gold_evalsafe: Path
    sentence_gold_quarantine_full: Path
    sentence_gold_quarantine_evalsafe: Path
    document_aux_full: Path
    document_aux_evalsafe: Path
    publication_sentence_full: Path
    publication_sentence_evalsafe: Path
    publication_sentence_legacy_full: Path
    publication_sentence_legacy_evalsafe: Path
    stage1_full: Path
    stage1_evalsafe: Path
    stage2_full: Path
    stage2_evalsafe: Path
    oracc_domain_full: Path
    oracc_domain_evalsafe: Path
    publication_review: Path
    report: Path


def normalize_source_text(text: str) -> str:
    return normalize_transliteration(
        str(text),
        full_normalize=True,
        do_strip_homophone_subscripts=True,
    ).strip()


def normalize_target_text(text: str) -> str:
    return clean_translation(str(text)).strip()


def source_tokens(text: str) -> list[str]:
    return [tok for tok in str(text).lower().split() if tok]


def english_tokens(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in WORD_RE.findall(str(text))
        if len(tok) > 1 and tok.lower() not in EN_STOPWORDS
    ]


def english_hint_hits(text: str) -> int:
    return sum(tok.lower() in ENGLISH_TARGET_HINTS for tok in WORD_RE.findall(str(text)))


def foreign_stopword_hits(text: str, language: str) -> int:
    stopwords = FOREIGN_TARGET_STOPWORDS.get(language)
    if not stopwords:
        return 0
    return sum(tok.lower() in stopwords for tok in WORD_RE.findall(str(text)))


def text_word_count(text: str) -> int:
    return len(str(text).split())


def utf8_len(text: str) -> int:
    return len(str(text).encode("utf-8"))


def _trim_formulaic_header_prefix(words: list[str], translation: str) -> tuple[str, str] | None:
    """Recover short sender/recipient headers whose source span absorbed later content."""
    if not words:
        return None

    target = str(translation)
    if not (target.startswith("From ") or target.startswith("To ") or target.endswith(":")):
        return None

    if words[0] == "um-ma":
        qibi_positions = [idx for idx, word in enumerate(words) if word == "qí-bi-ma"]
        if not qibi_positions:
            return None
        end = qibi_positions[0] + 1
        if len(qibi_positions) > 1 and (qibi_positions[1] - qibi_positions[0]) <= 8:
            end = qibi_positions[1] + 1
        return " ".join(words[:end]), "formulaic_header"

    if words[0] != "a-na":
        return None

    try:
        qibi_idx = words.index("qí-bi-ma")
        umma_idx = words.index("um-ma", qibi_idx + 1)
    except ValueError:
        return None

    end = umma_idx + 1
    for idx in range(umma_idx + 1, min(len(words), umma_idx + 6)):
        end = idx + 1
        if words[idx].endswith("-ma"):
            break
    return " ".join(words[:end]), "formulaic_header"


def _trim_seal_list_prefix(words: list[str], translation: str) -> tuple[str, str] | None:
    """Recover seal-list rows by cutting before numeric/legal follow-on content begins."""
    if not words or not str(translation).startswith(("Seal", "Seals")) or words[0] != "KIŠIB":
        return None

    boundary_tokens = {
        "ma-na",
        "GÚ",
        "GÍN",
        "GÍN.TA",
        "KÙ.BABBAR",
        "AN.NA",
        "URUDU",
        "i-na",
        "a-na",
        "ŠU.NÍGIN",
        "ITU.KAM",
        "ITU.1.KAM",
    }

    idx = 0
    seal_count = 0
    while idx < len(words):
        if words[idx] != "KIŠIB":
            break
        seal_count += 1
        idx += 1
        while idx < len(words):
            word = words[idx]
            if word == "KIŠIB":
                break
            if word in boundary_tokens or bool(re.search(r"\d", word)):
                if seal_count >= 1:
                    return " ".join(words[:idx]), "seal_list"
                return None
            idx += 1

    if seal_count >= 1 and idx < len(words):
        return " ".join(words[:idx]), "seal_list"
    return None


def maybe_recover_overlong_sentence_row(
    row: pd.Series,
    *,
    max_source_words: int,
    max_source_bytes: int,
) -> pd.Series | None:
    """Recover a small set of high-precision overlong rows by trimming obvious tails."""
    words = str(row.get("transliteration", "")).split()
    translation = str(row.get("translation", ""))

    for recover in (_trim_formulaic_header_prefix, _trim_seal_list_prefix):
        candidate = recover(words, translation)
        if candidate is None:
            continue
        recovered_text, strategy = candidate
        recovered_text = normalize_source_text(recovered_text)
        if not recovered_text:
            return None
        if text_word_count(recovered_text) > max_source_words:
            return None
        if utf8_len(recovered_text) > max_source_bytes:
            return None

        recovered = row.copy()
        recovered["transliteration"] = recovered_text
        recovered["source_word_count"] = text_word_count(recovered_text)
        recovered["source_byte_count"] = utf8_len(recovered_text)
        recovered["quality_flags"] = ""
        recovered["tier"] = "gold_recovered"
        recovered["confidence"] = min(float(recovered.get("confidence", 1.0)), 0.95)
        recovered["recovery_strategy"] = strategy
        recovered["recovered_from_quarantine"] = True
        recovered = recovered.drop(labels=["_orig_index"], errors="ignore")
        return recovered

    return None


def combined_size_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.combine(b, max) / a.combine(b, min).clip(lower=1)


def _append_quality_flag(base: pd.Series, mask: pd.Series, flag: str) -> pd.Series:
    updated = base.copy()
    empty_mask = mask & updated.eq("")
    nonempty_mask = mask & updated.ne("")
    updated.loc[empty_mask] = flag
    updated.loc[nonempty_mask] = updated.loc[nonempty_mask] + "|" + flag
    return updated


def filter_parallel_text_quality(
    df: pd.DataFrame,
    *,
    min_translation_chars: int,
    max_length_ratio: float,
    min_source_words: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same coarse quality filter used by prepared-data training."""
    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    src_chars = out["transliteration"].astype(str).str.len()
    tgt_chars = out["translation"].astype(str).str.len()
    size_ratio = combined_size_ratio(src_chars, tgt_chars)
    meta_like = out["translation"].astype(str).apply(lambda text: bool(META_RE.search(text)))
    blank_src = out["transliteration"].astype(str).str.strip().eq("")
    blank_tgt = out["translation"].astype(str).str.strip().eq("")
    short_tgt = tgt_chars < min_translation_chars
    bad_ratio = size_ratio > max_length_ratio
    if min_source_words is None:
        too_short_source = pd.Series(False, index=out.index)
    else:
        too_short_source = (
            out["transliteration"].astype(str).apply(text_word_count) < min_source_words
        )

    flags = pd.Series("", index=out.index, dtype="object")
    flags = _append_quality_flag(flags, short_tgt, "short_translation")
    flags = _append_quality_flag(flags, bad_ratio, "bad_length_ratio")
    flags = _append_quality_flag(flags, meta_like, "meta_like_translation")
    flags = _append_quality_flag(flags, blank_src, "blank_source")
    flags = _append_quality_flag(flags, blank_tgt, "blank_target")
    flags = _append_quality_flag(flags, too_short_source, "too_short_source")

    keep_mask = ~(short_tgt | bad_ratio | meta_like | blank_src | blank_tgt | too_short_source)
    kept = out.loc[keep_mask].copy()
    dropped = out.loc[~keep_mask].copy()
    if not kept.empty:
        kept["quality_flags"] = ""
    if not dropped.empty:
        dropped["quality_flags"] = flags.loc[dropped.index].values
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


def filter_non_english_sentence_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop published-text sentence rows whose targets are likely non-English."""
    if df.empty or "source" not in df.columns:
        return df.copy()

    out = df.copy()
    published_mask = out["source"].astype(str).eq("published_texts_sentence_gold")
    if not published_mask.any():
        return out

    from akkadian_mt.data.extract_publication_translations import detect_language

    langs = pd.Series("en", index=out.index, dtype="object")
    hints = pd.Series(0, index=out.index, dtype="int64")
    foreign_hits = pd.Series(0, index=out.index, dtype="int64")
    langs.loc[published_mask] = (
        out.loc[published_mask, "translation"].astype(str).apply(detect_language)
    )
    hints.loc[published_mask] = (
        out.loc[published_mask, "translation"].astype(str).apply(english_hint_hits)
    )
    foreign_hits.loc[published_mask] = [
        foreign_stopword_hits(text, language)
        for text, language in zip(
            out.loc[published_mask, "translation"].astype(str),
            langs.loc[published_mask],
            strict=False,
        )
    ]
    out["translation_language"] = langs
    out["translation_english_hint_hits"] = hints
    out["translation_foreign_stopword_hits"] = foreign_hits

    drop_mask = (
        (published_mask & langs.eq("de") & foreign_hits.ge(1))
        | (published_mask & langs.eq("de") & hints.lt(2))
        | (published_mask & langs.isin(STRICT_FOREIGN_TARGET_LANGS) & foreign_hits.ge(1))
        | (published_mask & langs.isin(STRICT_FOREIGN_TARGET_LANGS) & hints.eq(0))
    )
    return out.loc[~drop_mask].reset_index(drop=True)


def filter_sentence_rows_by_source_length(
    df: pd.DataFrame,
    *,
    max_source_words: int,
    max_source_bytes: int,
) -> pd.DataFrame:
    """Drop sentence rows that would exceed the Stage 2 source-length guardrails."""
    if df.empty:
        return df.copy()

    out = df.copy()
    word_counts = out["transliteration"].astype(str).apply(text_word_count)
    byte_counts = out["transliteration"].astype(str).apply(utf8_len)
    keep_mask = word_counts.le(max_source_words) & byte_counts.le(max_source_bytes)
    removed = int((~keep_mask).sum())
    if removed > 0:
        logger.info("Dropped %d publication sentence rows above the source-length cap", removed)
    return out.loc[keep_mask].reset_index(drop=True)


def split_sentence_gold_by_quality(
    df: pd.DataFrame,
    *,
    max_source_words: int,
    max_source_bytes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sentence gold into a core set and a quarantine set for long spans."""
    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    out["source_word_count"] = out["transliteration"].astype(str).apply(text_word_count)
    out["source_byte_count"] = out["transliteration"].astype(str).apply(utf8_len)

    too_many_words = out["source_word_count"] > max_source_words
    too_many_bytes = out["source_byte_count"] > max_source_bytes
    quarantine_mask = too_many_words | too_many_bytes

    flags = pd.Series("", index=out.index, dtype="object")
    flags = _append_quality_flag(flags, too_many_words, "overlong_sentence_words")
    flags = _append_quality_flag(flags, too_many_bytes, "overlong_sentence_bytes")

    core = out.loc[~quarantine_mask].copy()
    quarantine_candidates = out.loc[quarantine_mask].copy()

    recovered_rows: list[pd.Series] = []
    unrecovered_rows: list[pd.Series] = []
    for idx, row in quarantine_candidates.iterrows():
        row_series = row.copy()
        row_series["_orig_index"] = idx
        recovered = maybe_recover_overlong_sentence_row(
            row_series,
            max_source_words=max_source_words,
            max_source_bytes=max_source_bytes,
        )
        if recovered is None:
            unrecovered_rows.append(row_series)
        else:
            recovered_rows.append(recovered)

    if recovered_rows:
        recovered_df = pd.DataFrame(recovered_rows)
        core = pd.concat([core, recovered_df], ignore_index=True, sort=False)
    quarantine = pd.DataFrame(unrecovered_rows)

    if not core.empty:
        if "recovered_from_quarantine" not in core.columns:
            core["recovered_from_quarantine"] = False
        else:
            core["recovered_from_quarantine"] = core["recovered_from_quarantine"].fillna(False)
        if "recovery_strategy" not in core.columns:
            core["recovery_strategy"] = ""
        else:
            core["recovery_strategy"] = core["recovery_strategy"].fillna("")
        core["quality_flags"] = ""
    if not quarantine.empty:
        quarantine["quality_flags"] = flags.loc[quarantine["_orig_index"]].values
        quarantine["tier"] = "gold_quarantine"
        quarantine["confidence"] = quarantine["confidence"].astype(float).clip(upper=0.7)
        quarantine["recovered_from_quarantine"] = False
        quarantine["recovery_strategy"] = ""
        quarantine = quarantine.drop(columns=["_orig_index"], errors="ignore")

    return core.reset_index(drop=True), quarantine.reset_index(drop=True)


def translation_similarity(a: str, b: str) -> float:
    seq = SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()
    a_toks = set(english_tokens(a))
    b_toks = set(english_tokens(b))
    if not a_toks or not b_toks:
        return seq
    jaccard = len(a_toks & b_toks) / len(a_toks | b_toks)
    return max(seq, jaccard)


def resolve_holdout_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for candidate in DEFAULT_HOLDOUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_holdout_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def filter_holdout_docs(df: pd.DataFrame, holdout_ids: set[str]) -> pd.DataFrame:
    if not holdout_ids or "oare_id" not in df.columns:
        return df.copy()
    mask = ~df["oare_id"].fillna("").astype(str).isin(holdout_ids)
    return df[mask].copy()


def dedupe_parallel_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if {"transliteration", "translation"}.issubset(df.columns):
        return df.drop_duplicates(
            subset=["transliteration", "translation"], keep="first"
        ).reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def write_optional_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.unlink(missing_ok=True)
        return
    df.to_csv(path, index=False)


def repair_sentence_start_indices(
    words: list[str],
    sentence_rows: pd.DataFrame,
    *,
    search_radius: int = 3,
) -> list[int]:
    """Repair slightly misaligned sentence starts using the first-word anchor."""
    starts: list[int] = []
    last = -1
    sentence_rows = sentence_rows.sort_values("first_word_number").reset_index(drop=True)

    for row in sentence_rows.itertuples():
        nominal = int(row.first_word_number) - 1
        expected = str(getattr(row, "first_word_spelling", "")).strip()
        chosen: int | None = None

        if (
            0 <= nominal < len(words)
            and nominal > last
            and _validate_anchor(words[nominal], expected)
        ):
            chosen = nominal
        else:
            for dist in range(0, search_radius + 1):
                candidates = [nominal] if dist == 0 else [nominal - dist, nominal + dist]
                matches = []
                for cand in candidates:
                    if cand <= last or cand < 0 or cand >= len(words):
                        continue
                    if _validate_anchor(words[cand], expected):
                        matches.append(cand)
                if matches:
                    chosen = min(matches, key=lambda cand: (abs(cand - nominal), cand))
                    break

        if chosen is None:
            chosen = max(nominal, last + 1)
        starts.append(chosen)
        last = chosen

    return starts


def build_repaired_sentence_gold(
    data_dir: Path,
    *,
    search_radius: int = 3,
) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    pub_df = pd.read_csv(data_dir / "published_texts.csv")
    sent_df = pd.read_csv(data_dir / "Sentences_Oare_FirstWord_LinNum.csv")

    train_by_id = {
        str(row["oare_id"]): str(row["transliteration"])
        for _, row in train_df.dropna(subset=["oare_id", "transliteration"]).iterrows()
    }
    pub_by_id = {
        str(row["oare_id"]): str(row["transliteration"])
        for _, row in pub_df.dropna(subset=["oare_id", "transliteration"]).iterrows()
        if str(row["oare_id"]) not in train_by_id
    }
    genre_map = {
        str(row["oare_id"]): str(row.get("genre_label", "") or "")
        for _, row in pub_df.dropna(subset=["oare_id"]).iterrows()
    }

    rows: list[dict[str, Any]] = []
    repaired_count = 0

    for doc_id, group in sent_df.groupby("text_uuid"):
        doc_id_str = str(doc_id)
        source_label = ""
        full_text = ""
        if doc_id_str in train_by_id:
            source_label = "train_sentence_gold"
            full_text = train_by_id[doc_id_str]
        elif doc_id_str in pub_by_id:
            source_label = "published_texts_sentence_gold"
            full_text = pub_by_id[doc_id_str]
        else:
            continue

        words = full_text.split()
        if not words:
            continue

        group = group.sort_values("first_word_number").reset_index(drop=True)
        nominal_starts = [int(v) - 1 for v in group["first_word_number"].tolist()]
        starts = repair_sentence_start_indices(words, group, search_radius=search_radius)

        for idx, row in enumerate(group.itertuples()):
            translation = str(getattr(row, "translation", "") or "").strip()
            if not translation or translation == "nan":
                continue
            start = starts[idx]
            end = starts[idx + 1] if idx + 1 < len(starts) else len(words)
            if start < 0 or start >= len(words) or end <= start:
                continue

            sentence_words = words[start:end]
            expected = str(getattr(row, "first_word_spelling", "")).strip()
            anchor_valid = (
                _validate_anchor(sentence_words[0], expected) if sentence_words else False
            )
            repaired = start != nominal_starts[idx]
            repaired_count += int(repaired)

            rows.append(
                {
                    "oare_id": doc_id_str,
                    "transliteration": normalize_source_text(" ".join(sentence_words)),
                    "translation": normalize_target_text(translation),
                    "source": source_label,
                    "granularity": "sentence",
                    "tier": "gold",
                    "confidence": 1.0 if anchor_valid else 0.0,
                    "genre": genre_map.get(doc_id_str, ""),
                    "sentence_idx": int(idx),
                    "anchor_valid": anchor_valid,
                    "word_start": int(start),
                    "word_end": int(end),
                    "anchor_repaired": repaired,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    before = len(df)
    df = (
        df[
            df["anchor_valid"]
            & df["transliteration"].astype(str).str.strip().ne("")
            & df["translation"].astype(str).str.strip().ne("")
        ]
        .copy()
        .reset_index(drop=True)
    )
    logger.info(
        "Repaired sentence gold: %d -> %d rows (repaired starts=%d)",
        before,
        len(df),
        repaired_count,
    )
    return df


def build_document_aux(data_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    pub_df = pd.read_csv(data_dir / "published_texts.csv")
    sentence_ids = set(
        pd.read_csv(data_dir / "Sentences_Oare_FirstWord_LinNum.csv")["text_uuid"]
        .dropna()
        .astype(str)
    )
    genre_map = {
        str(row["oare_id"]): str(row.get("genre_label", "") or "")
        for _, row in pub_df.dropna(subset=["oare_id"]).iterrows()
    }
    df = train_df[~train_df["oare_id"].fillna("").astype(str).isin(sentence_ids)].copy()
    df["transliteration"] = df["transliteration"].astype(str).apply(normalize_source_text)
    df["translation"] = df["translation"].astype(str).apply(normalize_target_text)
    df = df[
        df["transliteration"].astype(str).str.strip().ne("")
        & df["translation"].astype(str).str.strip().ne("")
    ].copy()
    df["source"] = "train_document_aux"
    df["granularity"] = "document"
    df["tier"] = "aux_document"
    df["confidence"] = 0.9
    df["genre"] = df["oare_id"].astype(str).map(genre_map).fillna("")
    return df.reset_index(drop=True)


def filter_publication_candidates(
    df: pd.DataFrame,
    *,
    confidence_min: float,
    max_repeat: int,
    min_chars: int,
    max_chars: int,
) -> pd.DataFrame:
    from akkadian_mt.data.extract_publication_translations import score_translation_confidence

    out = df.copy()
    out["translation"] = out["translation"].astype(str).apply(normalize_target_text)
    out["translation_language"] = out.get("translation_language", "en").astype(str)
    out = out[out["translation_language"] == "en"].copy()

    # Re-score cleaned text to make filtering deterministic.
    rescored = out["translation"].apply(lambda text: score_translation_confidence(text, "en"))
    out["translation_confidence"] = rescored.apply(lambda item: item["confidence"])
    out["translation_confidence_flags"] = rescored.apply(lambda item: "|".join(item["flags"]))
    out["translation_common_word_hits"] = rescored.apply(lambda item: item["common_word_hits"])
    out["translation_alpha_fraction"] = rescored.apply(lambda item: item["alpha_fraction"])
    out["translation_repeat_count"] = out.groupby("translation")["translation"].transform("size")

    out = out[
        (out["translation_confidence"] >= confidence_min)
        & (out["translation_repeat_count"] <= max_repeat)
        & (out["translation"].str.len().between(min_chars, max_chars))
        & (out["translation_common_word_hits"] >= 2)
        & (out["translation_alpha_fraction"] >= 0.55)
        & (
            ~out["translation_confidence_flags"]
            .fillna("")
            .astype(str)
            .str.contains(BAD_PUBLICATION_FLAGS_RE)
        )
        & (
            out["extraction_pattern"]
            .astype(str)
            .isin(["quoted", "line", "marker", "translation_block"])
        )
    ].copy()
    out = out.drop_duplicates(subset=["oare_id", "translation"], keep="first").reset_index(
        drop=True
    )
    return out


def match_publication_to_sentence(
    translation: str,
    refs: list[dict[str, Any]],
    *,
    similarity_min: float,
    similarity_margin: float,
    min_word_ratio: float,
    max_word_ratio: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = 0.0
    second_score = 0.0

    for ref in refs:
        ref_words = max(text_word_count(ref["translation"]), 1)
        cand_words = max(text_word_count(translation), 1)
        word_ratio = cand_words / ref_words
        if word_ratio < min_word_ratio or word_ratio > max_word_ratio:
            continue
        score = translation_similarity(translation, ref["translation"])
        if score > best_score:
            second_score = best_score
            best_score = score
            best = ref
        elif score > second_score:
            second_score = score

    if best is None:
        return None
    if best_score < similarity_min:
        return None
    if (best_score - second_score) < similarity_margin:
        return None
    ref_words = max(text_word_count(best["translation"]), 1)
    cand_words = max(text_word_count(translation), 1)
    word_ratio = cand_words / ref_words
    return {
        "ref": best,
        "score": best_score,
        "margin": best_score - second_score,
        "word_ratio": word_ratio,
    }


def build_publication_sentence_aux(
    data_dir: Path,
    sentence_gold: pd.DataFrame,
    *,
    publication_cache: Path,
    similarity_min: float,
    similarity_margin: float,
    confidence_min: float,
    max_repeat: int,
    min_chars: int,
    max_chars: int,
    review_n: int,
    min_word_ratio: float,
    max_word_ratio: float,
    force_extract: bool = False,
    sample_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if force_extract or not publication_cache.exists():
        from akkadian_mt.data.extract_publication_translations import process_publications

        publication_cache.parent.mkdir(parents=True, exist_ok=True)
        process_publications(
            data_dir=data_dir,
            output_path=publication_cache,
            sample_size=sample_rows,
            min_translation_length=min_chars,
        )

    if not publication_cache.exists():
        logger.warning("Publication cache missing: %s", publication_cache)
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.read_csv(publication_cache)
    filtered = filter_publication_candidates(
        raw,
        confidence_min=confidence_min,
        max_repeat=max_repeat,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    refs_by_doc: dict[str, list[dict[str, Any]]] = {}
    for _, row in sentence_gold.iterrows():
        refs_by_doc.setdefault(str(row["oare_id"]), []).append(
            {
                "transliteration": row["transliteration"],
                "translation": row["translation"],
                "sentence_idx": int(row["sentence_idx"]),
                "genre": row.get("genre", ""),
            }
        )

    aligned_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for _, row in filtered.iterrows():
        oare_id = str(row["oare_id"])
        refs = refs_by_doc.get(oare_id)
        if not refs:
            continue
        match = match_publication_to_sentence(
            row["translation"],
            refs,
            similarity_min=similarity_min,
            similarity_margin=similarity_margin,
            min_word_ratio=min_word_ratio,
            max_word_ratio=max_word_ratio,
        )
        if match is None:
            continue
        ref = match["ref"]
        aligned_rows.append(
            {
                "oare_id": oare_id,
                "transliteration": ref["transliteration"],
                "translation": row["translation"],
                "source": "publications_sentence_aux",
                "granularity": "sentence",
                "tier": "experimental_publication",
                "confidence": float(match["score"]),
                "genre": ref.get("genre", ""),
                "sentence_idx": int(ref["sentence_idx"]),
                "alignment_score": float(match["score"]),
                "alignment_margin": float(match["margin"]),
                "alignment_word_ratio": float(match["word_ratio"]),
                "pdf_name": row.get("pdf_name", ""),
                "page": row.get("page", ""),
                "extraction_pattern": row.get("extraction_pattern", ""),
                "reference_translation": ref["translation"],
            }
        )
        review_rows.append(
            {
                "oare_id": oare_id,
                "alignment_score": float(match["score"]),
                "alignment_margin": float(match["margin"]),
                "alignment_word_ratio": float(match["word_ratio"]),
                "pdf_name": row.get("pdf_name", ""),
                "page": row.get("page", ""),
                "candidate_translation": row["translation"],
                "reference_translation": ref["translation"],
                "extraction_pattern": row.get("extraction_pattern", ""),
            }
        )

    aligned = pd.DataFrame(aligned_rows).reset_index(drop=True)

    review = pd.DataFrame(review_rows)
    if not review.empty:
        review = review.sort_values(
            ["alignment_score", "alignment_margin"], ascending=[False, False]
        ).head(review_n)

    return aligned, review


def build_legacy_publication_sentence_aux(
    legacy_path: Path,
    *,
    alignment_method: str,
    alignment_score_min: float,
    heuristic_confidence_min: float,
) -> pd.DataFrame:
    """Load a strict subset of legacy LLM-aligned publication sentence pairs."""
    if not legacy_path.exists():
        logger.warning("Legacy publication alignment cache missing: %s", legacy_path)
        return pd.DataFrame()

    legacy = pd.read_csv(legacy_path)
    required = {"transliteration", "translation", "oare_id", "alignment_method", "alignment_score"}
    if not required.issubset(legacy.columns):
        logger.warning(
            "Legacy publication alignment cache is missing required columns: %s",
            legacy_path,
        )
        return pd.DataFrame()

    out = legacy.copy()
    out["transliteration"] = out["transliteration"].astype(str).apply(normalize_source_text)
    out["translation"] = out["translation"].astype(str).apply(normalize_target_text)
    out = out[
        out["transliteration"].str.strip().ne("")
        & out["translation"].str.strip().ne("")
        & out["alignment_method"].astype(str).eq(alignment_method)
        & out["alignment_score"].fillna(0).astype(float).ge(alignment_score_min)
    ].copy()

    if "original_language" in out.columns:
        out = out[out["original_language"].astype(str).eq("en")].copy()
    if "translation_type" in out.columns:
        out = out[out["translation_type"].astype(str).eq("direct")].copy()
    if "heuristic_confidence" in out.columns:
        out = out[
            out["heuristic_confidence"].fillna(0).astype(float).ge(heuristic_confidence_min)
        ].copy()

    if out.empty:
        return out

    out["source"] = "publications_llm_aligned"
    out["granularity"] = "sentence"
    out["tier"] = "experimental_publication_llm"
    if "genre" not in out.columns:
        out["genre"] = ""
    if "sentence_idx" not in out.columns:
        out["sentence_idx"] = pd.NA
    out["confidence"] = out.apply(
        lambda row: min(
            float(row.get("alignment_score", 0.0)),
            float(row.get("heuristic_confidence", row.get("alignment_score", 0.0))),
        ),
        axis=1,
    )
    return out.reset_index(drop=True)


def build_target_logodds(
    positive_texts: pd.Series,
    background_texts: pd.Series,
) -> dict[str, float]:
    pos_counts = Counter(tok for text in positive_texts for tok in english_tokens(text))
    bg_counts = Counter(tok for text in background_texts for tok in english_tokens(text))
    all_words = set(pos_counts) | set(bg_counts)
    pos_total = sum(pos_counts.values())
    bg_total = sum(bg_counts.values())
    logodds: dict[str, float] = {}
    denom_pos = pos_total + len(all_words)
    denom_bg = bg_total + len(all_words)
    for word in all_words:
        p_pos = (pos_counts[word] + 1) / denom_pos
        p_bg = (bg_counts[word] + 1) / denom_bg
        logodds[word] = math.log(p_pos / p_bg)
    return logodds


def score_domain_candidate(
    transliteration: str,
    translation: str,
    *,
    source_vocab: set[str],
    target_logodds: dict[str, float],
) -> float:
    src = source_tokens(transliteration)
    tgt = english_tokens(translation)
    src_overlap = sum(tok in source_vocab for tok in src) / max(len(src), 1)
    domain_hits = [target_logodds[tok] for tok in tgt if tok in target_logodds]
    pos = sum(val for val in domain_hits if val > 0)
    neg = sum(val for val in domain_hits if val < 0)
    target_score = (pos + 0.2 * neg) / max(len(tgt), 1)
    return 0.75 * src_overlap + 0.25 * target_score


def keyword_hits(tokens: list[str], keywords: frozenset[str]) -> int:
    return sum(tok in keywords for tok in tokens)


def build_oracc_domain_pretrain(
    oracc_path: Path,
    competition_df: pd.DataFrame,
    *,
    min_score: float,
    min_keyword_hits: int,
    min_target_words: int,
    min_source_words: int,
) -> pd.DataFrame:
    if not oracc_path.exists():
        logger.warning("ORACC cache missing: %s", oracc_path)
        return pd.DataFrame()

    oracc = pd.read_csv(oracc_path)
    oracc["transliteration"] = oracc["transliteration"].astype(str).apply(normalize_source_text)
    oracc["translation"] = oracc["translation"].astype(str).apply(normalize_target_text)
    oracc = oracc[
        oracc["transliteration"].str.strip().ne("")
        & oracc["translation"].str.strip().ne("")
        & (oracc["translation"].str.len() >= 15)
        & (oracc["translation"].str.count(r"\w+") >= 3)
        & (~oracc["translation"].str.contains(r"^\.\.\.|^\[|^—", regex=True))
    ].copy()

    source_vocab = {
        tok for text in competition_df["transliteration"].astype(str) for tok in source_tokens(text)
    }
    target_logodds = build_target_logodds(
        competition_df["translation"].astype(str),
        oracc["translation"].astype(str),
    )
    oracc["domain_score"] = oracc.apply(
        lambda row: score_domain_candidate(
            row["transliteration"],
            row["translation"],
            source_vocab=source_vocab,
            target_logodds=target_logodds,
        ),
        axis=1,
    )
    oracc["source_word_count"] = oracc["transliteration"].astype(str).apply(text_word_count)
    oracc["target_word_count"] = oracc["translation"].astype(str).apply(text_word_count)
    oracc["target_keyword_hits"] = (
        oracc["translation"]
        .astype(str)
        .apply(lambda text: keyword_hits(english_tokens(text), ORACC_COMMERCE_TARGET_KEYWORDS))
    )
    oracc["source_keyword_hits"] = (
        oracc["transliteration"]
        .astype(str)
        .apply(lambda text: keyword_hits(source_tokens(text), ORACC_COMMERCE_SOURCE_KEYWORDS))
    )
    oracc["offdomain_keyword_hits"] = (
        oracc["translation"]
        .astype(str)
        .apply(lambda text: keyword_hits(english_tokens(text), ORACC_OFFDOMAIN_TARGET_KEYWORDS))
    )
    oracc = oracc[
        (oracc["domain_score"] >= min_score)
        & ((oracc["target_keyword_hits"] + oracc["source_keyword_hits"]) >= min_keyword_hits)
        & (oracc["offdomain_keyword_hits"] == 0)
        & (oracc["target_word_count"] >= min_target_words)
        & (oracc["source_word_count"] >= min_source_words)
    ].copy()
    oracc["oare_id"] = [f"oracc::{idx}" for idx in range(len(oracc))]
    oracc["tier"] = "pretrain_domain"
    oracc["confidence"] = oracc["domain_score"]
    oracc["granularity"] = "sentence"
    oracc["source"] = "oracc_domain_filtered"
    return dedupe_parallel_pairs(oracc)


def summarize_df(df: pd.DataFrame, path: Path | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df))}
    if path is not None:
        summary["path"] = str(path)
    if df.empty:
        return summary
    if "oare_id" in df.columns:
        summary["docs"] = int(
            df["oare_id"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()
        )
    if "source" in df.columns:
        summary["by_source"] = {k: int(v) for k, v in df["source"].value_counts().to_dict().items()}
    if "tier" in df.columns:
        summary["by_tier"] = {k: int(v) for k, v in df["tier"].value_counts().to_dict().items()}
    if "granularity" in df.columns:
        summary["by_granularity"] = {
            k: int(v) for k, v in df["granularity"].value_counts().to_dict().items()
        }
    return summary


def make_bundle_paths(out_dir: Path) -> BundlePaths:
    return BundlePaths(
        sentence_gold_full=out_dir / "sentence_gold_full.csv",
        sentence_gold_evalsafe=out_dir / "sentence_gold_evalsafe.csv",
        sentence_gold_quarantine_full=out_dir / "sentence_gold_quarantine_full.csv",
        sentence_gold_quarantine_evalsafe=out_dir / "sentence_gold_quarantine_evalsafe.csv",
        document_aux_full=out_dir / "document_aux_full.csv",
        document_aux_evalsafe=out_dir / "document_aux_evalsafe.csv",
        publication_sentence_full=out_dir / "publication_sentence_experimental_full.csv",
        publication_sentence_evalsafe=out_dir / "publication_sentence_experimental_evalsafe.csv",
        publication_sentence_legacy_full=out_dir / "publication_sentence_legacy_full.csv",
        publication_sentence_legacy_evalsafe=out_dir / "publication_sentence_legacy_evalsafe.csv",
        stage1_full=out_dir / "stage1_curriculum_full.csv",
        stage1_evalsafe=out_dir / "stage1_curriculum_evalsafe.csv",
        stage2_full=out_dir / "stage2_finetune_full.csv",
        stage2_evalsafe=out_dir / "stage2_finetune_evalsafe.csv",
        oracc_domain_full=out_dir / "oracc_domain_pretrain.csv",
        oracc_domain_evalsafe=out_dir / "oracc_domain_pretrain_evalsafe.csv",
        publication_review=out_dir / "publication_alignment_review.csv",
        report=out_dir / "REPORT.json",
    )


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = resolve_holdout_path(args.holdout_doc_ids)
    holdout_ids = load_holdout_ids(holdout_path)
    paths = make_bundle_paths(out_dir)

    sentence_gold_raw = build_repaired_sentence_gold(
        data_dir,
        search_radius=args.search_radius,
    )
    sentence_gold_full_all = dedupe_parallel_pairs(sentence_gold_raw)
    sentence_gold_evalsafe_all = dedupe_parallel_pairs(
        filter_holdout_docs(sentence_gold_raw, holdout_ids)
    )
    sentence_gold_full_all, _ = filter_parallel_text_quality(
        sentence_gold_full_all,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=2,
    )
    sentence_gold_evalsafe_all, _ = filter_parallel_text_quality(
        sentence_gold_evalsafe_all,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=2,
    )
    sentence_gold_full_all = filter_non_english_sentence_targets(sentence_gold_full_all)
    sentence_gold_evalsafe_all = filter_non_english_sentence_targets(sentence_gold_evalsafe_all)
    sentence_gold_full, sentence_gold_quarantine_full = split_sentence_gold_by_quality(
        sentence_gold_full_all,
        max_source_words=args.max_sentence_source_words,
        max_source_bytes=args.max_sentence_source_bytes,
    )
    sentence_gold_evalsafe, sentence_gold_quarantine_evalsafe = split_sentence_gold_by_quality(
        sentence_gold_evalsafe_all,
        max_source_words=args.max_sentence_source_words,
        max_source_bytes=args.max_sentence_source_bytes,
    )

    document_aux_raw = build_document_aux(data_dir)
    document_aux_full = dedupe_parallel_pairs(document_aux_raw)
    document_aux_evalsafe = dedupe_parallel_pairs(
        filter_holdout_docs(document_aux_raw, holdout_ids)
    )
    document_aux_full, _ = filter_parallel_text_quality(
        document_aux_full,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
    )
    document_aux_evalsafe, _ = filter_parallel_text_quality(
        document_aux_evalsafe,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
    )

    publication_raw, publication_review = build_publication_sentence_aux(
        data_dir,
        sentence_gold_full,
        publication_cache=Path(args.publication_cache),
        similarity_min=args.publication_similarity_min,
        similarity_margin=args.publication_similarity_margin,
        confidence_min=args.publication_confidence_min,
        max_repeat=args.publication_max_repeat,
        min_chars=args.publication_min_chars,
        max_chars=args.publication_max_chars,
        review_n=args.publication_review_n,
        min_word_ratio=args.publication_min_word_ratio,
        max_word_ratio=args.publication_max_word_ratio,
        force_extract=args.force_publication_extract,
        sample_rows=args.publication_sample_rows,
    )
    publication_full = dedupe_parallel_pairs(publication_raw)
    publication_evalsafe = dedupe_parallel_pairs(filter_holdout_docs(publication_raw, holdout_ids))
    publication_full, _ = filter_parallel_text_quality(
        publication_full,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=2,
    )
    publication_evalsafe, _ = filter_parallel_text_quality(
        publication_evalsafe,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=2,
    )
    publication_full = filter_sentence_rows_by_source_length(
        publication_full,
        max_source_words=args.max_sentence_source_words,
        max_source_bytes=args.max_sentence_source_bytes,
    )
    publication_evalsafe = filter_sentence_rows_by_source_length(
        publication_evalsafe,
        max_source_words=args.max_sentence_source_words,
        max_source_bytes=args.max_sentence_source_bytes,
    )
    publication_legacy_raw = build_legacy_publication_sentence_aux(
        Path(args.legacy_publication_aligned_path),
        alignment_method=args.legacy_publication_alignment_method,
        alignment_score_min=args.legacy_publication_alignment_score_min,
        heuristic_confidence_min=args.legacy_publication_heuristic_confidence_min,
    )
    publication_legacy_full = dedupe_parallel_pairs(publication_legacy_raw)
    publication_legacy_evalsafe = dedupe_parallel_pairs(
        filter_holdout_docs(publication_legacy_raw, holdout_ids)
    )
    publication_legacy_full, _ = filter_parallel_text_quality(
        publication_legacy_full,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=2,
    )
    publication_legacy_evalsafe, _ = filter_parallel_text_quality(
        publication_legacy_evalsafe,
        min_translation_chars=args.min_translation_chars,
        max_length_ratio=args.max_length_ratio,
        min_source_words=2,
    )
    publication_legacy_full = filter_sentence_rows_by_source_length(
        publication_legacy_full,
        max_source_words=args.max_sentence_source_words,
        max_source_bytes=args.max_sentence_source_bytes,
    )
    publication_legacy_evalsafe = filter_sentence_rows_by_source_length(
        publication_legacy_evalsafe,
        max_source_words=args.max_sentence_source_words,
        max_source_bytes=args.max_sentence_source_bytes,
    )

    stage2_full = dedupe_parallel_pairs(
        pd.concat(
            [sentence_gold_full, publication_full, publication_legacy_full],
            ignore_index=True,
            sort=False,
        )
    )
    stage2_evalsafe = dedupe_parallel_pairs(
        pd.concat(
            [sentence_gold_evalsafe, publication_evalsafe, publication_legacy_evalsafe],
            ignore_index=True,
            sort=False,
        )
    )

    competition_like_full = dedupe_parallel_pairs(
        pd.concat([sentence_gold_full, document_aux_full], ignore_index=True, sort=False)
    )
    competition_like_evalsafe = dedupe_parallel_pairs(
        pd.concat([sentence_gold_evalsafe, document_aux_evalsafe], ignore_index=True, sort=False)
    )
    if args.no_oracc:
        oracc_domain_full = pd.DataFrame()
        oracc_domain_evalsafe = pd.DataFrame()
    else:
        oracc_path = Path(args.oracc_path)
        oracc_domain_full = build_oracc_domain_pretrain(
            oracc_path,
            competition_like_full,
            min_score=args.oracc_min_score,
            min_keyword_hits=args.oracc_min_keyword_hits,
            min_target_words=args.oracc_min_target_words,
            min_source_words=args.oracc_min_source_words,
        )
        oracc_domain_evalsafe = build_oracc_domain_pretrain(
            oracc_path,
            competition_like_evalsafe,
            min_score=args.oracc_min_score,
            min_keyword_hits=args.oracc_min_keyword_hits,
            min_target_words=args.oracc_min_target_words,
            min_source_words=args.oracc_min_source_words,
        )

    stage1_full = dedupe_parallel_pairs(
        pd.concat(
            [
                sentence_gold_full,
                sentence_gold_quarantine_full,
                document_aux_full,
            ],
            ignore_index=True,
            sort=False,
        )
    )
    stage1_evalsafe = dedupe_parallel_pairs(
        pd.concat(
            [
                sentence_gold_evalsafe,
                sentence_gold_quarantine_evalsafe,
                document_aux_evalsafe,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    sentence_gold_full.to_csv(paths.sentence_gold_full, index=False)
    sentence_gold_evalsafe.to_csv(paths.sentence_gold_evalsafe, index=False)
    write_optional_csv(sentence_gold_quarantine_full, paths.sentence_gold_quarantine_full)
    write_optional_csv(sentence_gold_quarantine_evalsafe, paths.sentence_gold_quarantine_evalsafe)
    document_aux_full.to_csv(paths.document_aux_full, index=False)
    document_aux_evalsafe.to_csv(paths.document_aux_evalsafe, index=False)
    publication_full.to_csv(paths.publication_sentence_full, index=False)
    publication_evalsafe.to_csv(paths.publication_sentence_evalsafe, index=False)
    write_optional_csv(publication_legacy_full, paths.publication_sentence_legacy_full)
    write_optional_csv(publication_legacy_evalsafe, paths.publication_sentence_legacy_evalsafe)
    stage1_full.to_csv(paths.stage1_full, index=False)
    stage1_evalsafe.to_csv(paths.stage1_evalsafe, index=False)
    stage2_full.to_csv(paths.stage2_full, index=False)
    stage2_evalsafe.to_csv(paths.stage2_evalsafe, index=False)
    write_optional_csv(oracc_domain_full, paths.oracc_domain_full)
    write_optional_csv(oracc_domain_evalsafe, paths.oracc_domain_evalsafe)
    write_optional_csv(publication_review, paths.publication_review)

    report = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "holdout_doc_ids_path": str(holdout_path) if holdout_path else None,
        "holdout_doc_count": len(holdout_ids),
        "config": {
            "search_radius": args.search_radius,
            "min_translation_chars": args.min_translation_chars,
            "max_length_ratio": args.max_length_ratio,
            "max_sentence_source_words": args.max_sentence_source_words,
            "max_sentence_source_bytes": args.max_sentence_source_bytes,
            "publication_similarity_min": args.publication_similarity_min,
            "publication_similarity_margin": args.publication_similarity_margin,
            "publication_min_word_ratio": args.publication_min_word_ratio,
            "publication_max_word_ratio": args.publication_max_word_ratio,
            "publication_confidence_min": args.publication_confidence_min,
            "publication_max_repeat": args.publication_max_repeat,
            "legacy_publication_aligned_path": args.legacy_publication_aligned_path,
            "legacy_publication_alignment_method": args.legacy_publication_alignment_method,
            "legacy_publication_alignment_score_min": args.legacy_publication_alignment_score_min,
            "legacy_publication_heuristic_confidence_min": args.legacy_publication_heuristic_confidence_min,
            "oracc_min_score": args.oracc_min_score,
            "oracc_min_keyword_hits": args.oracc_min_keyword_hits,
            "oracc_min_target_words": args.oracc_min_target_words,
            "oracc_min_source_words": args.oracc_min_source_words,
        },
        "artifacts": {
            "sentence_gold_full": summarize_df(sentence_gold_full, paths.sentence_gold_full),
            "sentence_gold_evalsafe": summarize_df(
                sentence_gold_evalsafe, paths.sentence_gold_evalsafe
            ),
            "sentence_gold_quarantine_full": summarize_df(
                sentence_gold_quarantine_full, paths.sentence_gold_quarantine_full
            ),
            "sentence_gold_quarantine_evalsafe": summarize_df(
                sentence_gold_quarantine_evalsafe,
                paths.sentence_gold_quarantine_evalsafe,
            ),
            "document_aux_full": summarize_df(document_aux_full, paths.document_aux_full),
            "document_aux_evalsafe": summarize_df(
                document_aux_evalsafe, paths.document_aux_evalsafe
            ),
            "publication_sentence_full": summarize_df(
                publication_full, paths.publication_sentence_full
            ),
            "publication_sentence_evalsafe": summarize_df(
                publication_evalsafe, paths.publication_sentence_evalsafe
            ),
            "publication_sentence_legacy_full": summarize_df(
                publication_legacy_full, paths.publication_sentence_legacy_full
            ),
            "publication_sentence_legacy_evalsafe": summarize_df(
                publication_legacy_evalsafe,
                paths.publication_sentence_legacy_evalsafe,
            ),
            "stage1_full": summarize_df(stage1_full, paths.stage1_full),
            "stage1_evalsafe": summarize_df(stage1_evalsafe, paths.stage1_evalsafe),
            "stage2_full": summarize_df(stage2_full, paths.stage2_full),
            "stage2_evalsafe": summarize_df(stage2_evalsafe, paths.stage2_evalsafe),
            "oracc_domain_full": summarize_df(oracc_domain_full, paths.oracc_domain_full),
            "oracc_domain_evalsafe": summarize_df(
                oracc_domain_evalsafe, paths.oracc_domain_evalsafe
            ),
        },
    }
    paths.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Raw Kaggle data directory")
    parser.add_argument(
        "--out-dir",
        default="data/processed/programmatic_v1",
        help="Output directory for the generated corpora",
    )
    parser.add_argument(
        "--holdout-doc-ids",
        default=None,
        help="Optional holdout JSON path; defaults to the newest known local file",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=3,
        help="Local word search radius for repairing sentence anchors",
    )
    parser.add_argument("--min-translation-chars", type=int, default=10)
    parser.add_argument("--max-length-ratio", type=float, default=10.0)
    parser.add_argument("--max-sentence-source-words", type=int, default=100)
    parser.add_argument("--max-sentence-source-bytes", type=int, default=480)
    parser.add_argument(
        "--publication-cache",
        default="data/processed/publication_translations.csv",
        help="Cache path for regex-based publication extractions",
    )
    parser.add_argument(
        "--legacy-publication-aligned-path",
        default="data/processed/filtered_translations_all.csv",
        help="Optional legacy LLM-aligned publication sentence cache to mine conservatively",
    )
    parser.add_argument(
        "--legacy-publication-alignment-method",
        default="transliteration_span",
        help="Alignment method required for legacy LLM publication rows",
    )
    parser.add_argument(
        "--legacy-publication-alignment-score-min",
        type=float,
        default=0.90,
        help="Minimum alignment score for legacy LLM publication rows",
    )
    parser.add_argument(
        "--legacy-publication-heuristic-confidence-min",
        type=float,
        default=0.65,
        help="Minimum heuristic confidence for legacy LLM publication rows",
    )
    parser.add_argument(
        "--force-publication-extract",
        action="store_true",
        help="Re-run the regex extractor instead of using the cached CSV",
    )
    parser.add_argument(
        "--publication-sample-rows",
        type=int,
        default=None,
        help="Optional row limit when force-running publication extraction",
    )
    parser.add_argument("--publication-confidence-min", type=float, default=0.80)
    parser.add_argument("--publication-max-repeat", type=int, default=2)
    parser.add_argument("--publication-min-chars", type=int, default=25)
    parser.add_argument("--publication-max-chars", type=int, default=260)
    parser.add_argument("--publication-similarity-min", type=float, default=0.40)
    parser.add_argument("--publication-similarity-margin", type=float, default=0.08)
    parser.add_argument("--publication-min-word-ratio", type=float, default=0.5)
    parser.add_argument("--publication-max-word-ratio", type=float, default=2.0)
    parser.add_argument("--publication-review-n", type=int, default=200)
    parser.add_argument(
        "--oracc-path",
        default="data/processed/oracc_pairs.csv",
        help="Path to the Akkademia/ORACC parallel data CSV",
    )
    parser.add_argument("--oracc-min-score", type=float, default=0.25)
    parser.add_argument("--oracc-min-keyword-hits", type=int, default=2)
    parser.add_argument("--oracc-min-target-words", type=int, default=8)
    parser.add_argument("--oracc-min-source-words", type=int, default=3)
    parser.add_argument("--no-oracc", action="store_true", help="Skip the ORACC pretraining tier")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = build_bundle(args)
    logger.info("Programmatic corpus bundle written to %s", args.out_dir)
    logger.info(
        "Stage 2 finetune rows: %d | Stage 1 curriculum rows: %d",
        report["artifacts"]["stage2_full"]["rows"],
        report["artifacts"]["stage1_full"]["rows"],
    )
    return 0
