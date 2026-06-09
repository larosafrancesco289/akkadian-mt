"""Sentence-level data extraction from document-level train.csv using Sentences_Oare."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentencePair:
    """A sentence-level transliteration/translation pair."""

    transliteration: str
    translation: str
    doc_id: str
    sentence_idx: int
    anchor_valid: bool
    word_start: int
    word_end: int


def extract_sentence_spans(
    doc_transliteration: str,
    sentence_rows: pd.DataFrame,
) -> list[SentencePair]:
    """Extract sentence spans from a document using word positions.

    Args:
        doc_transliteration: Full document transliteration (space-separated words)
        sentence_rows: DataFrame rows from Sentences_Oare for this document,
                      sorted by first_word_number

    Returns:
        List of SentencePair objects with extracted transliterations
    """
    words = doc_transliteration.split()
    num_words = len(words)
    results: list[SentencePair] = []

    # Sort by first_word_number to process in order
    sentence_rows = sentence_rows.sort_values("first_word_number").reset_index(drop=True)

    for i, row in sentence_rows.iterrows():
        first_word_num = int(row["first_word_number"])
        first_word_spelling = str(row.get("first_word_spelling", "")).strip()
        translation = row.get("translation", "")
        doc_id = row.get("text_uuid", "")

        # Convert to 0-indexed
        start_idx = first_word_num - 1

        # Determine end index: either next sentence start or document end
        if i + 1 < len(sentence_rows):
            next_first_word = int(sentence_rows.iloc[i + 1]["first_word_number"])
            end_idx = next_first_word - 1
        else:
            end_idx = num_words

        # Validate bounds
        if start_idx < 0 or start_idx >= num_words:
            logger.debug(
                "Skipping sentence: start_idx %d out of bounds (doc has %d words)",
                start_idx,
                num_words,
            )
            continue

        if end_idx <= start_idx:
            logger.debug(
                "Skipping sentence: end_idx %d <= start_idx %d",
                end_idx,
                start_idx,
            )
            continue

        # Extract sentence transliteration
        sentence_words = words[start_idx:end_idx]
        sentence_translit = " ".join(sentence_words)

        # Validate anchor word
        anchor_valid = _validate_anchor(
            sentence_words[0] if sentence_words else "",
            first_word_spelling,
        )

        # Skip sentences with empty translation
        if not translation or pd.isna(translation) or str(translation).strip() == "":
            continue

        results.append(
            SentencePair(
                transliteration=sentence_translit,
                translation=str(translation).strip(),
                doc_id=str(doc_id),
                sentence_idx=int(i),
                anchor_valid=anchor_valid,
                word_start=start_idx,
                word_end=end_idx,
            )
        )

    return results


def _validate_anchor(actual_word: str, expected_spelling: str) -> bool:
    """Validate that the actual word matches the expected anchor spelling.

    Handles common variations:
    - Case differences
    - Subscript normalization
    - Diacritic variations
    """
    if not actual_word or not expected_spelling:
        return False

    # Normalize for comparison
    actual = actual_word.lower().strip()
    expected = expected_spelling.lower().strip()

    # Direct match
    if actual == expected:
        return True

    # Remove subscript digits for comparison
    from akkadian_mt.data.preprocessing import SUBSCRIPT_TRANS

    actual_norm = actual.translate(SUBSCRIPT_TRANS)
    expected_norm = expected.translate(SUBSCRIPT_TRANS)

    if actual_norm == expected_norm:
        return True

    # Check if one is a substring of the other (for compound words)
    if actual_norm.startswith(expected_norm) or expected_norm.startswith(actual_norm):
        return True

    return False


def load_sentence_level_data(data_dir: str | Path) -> pd.DataFrame:
    """Load sentence-level training data extracted from documents.

    Args:
        data_dir: Path to data directory containing train.csv and Sentences_Oare

    Returns:
        DataFrame with columns:
          - transliteration, translation
          - doc_id (oare_id), sentence_idx
          - anchor_valid, word_start, word_end
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"

    train_df = pd.read_csv(train_path)
    sentences_df = pd.read_csv(sentences_path)

    # Find documents that have sentence info
    train_ids = set(train_df["oare_id"].dropna().unique())
    sentence_uuids = set(sentences_df["text_uuid"].dropna().unique())
    docs_with_sentences = train_ids & sentence_uuids

    logger.info(
        "Documents with sentence info: %d / %d",
        len(docs_with_sentences),
        len(train_ids),
    )

    # Pre-group for O(1) lookups instead of repeated DataFrame scans
    train_by_id = {oid: group.iloc[0] for oid, group in train_df.groupby("oare_id")}
    sentences_by_id = dict(iter(sentences_df.groupby("text_uuid")))

    all_pairs: list[dict] = []
    valid_count = 0
    invalid_count = 0

    for doc_id in docs_with_sentences:
        doc_row = train_by_id[doc_id]
        doc_sentences = sentences_by_id[doc_id]

        pairs = extract_sentence_spans(
            doc_row["transliteration"],
            doc_sentences,
        )

        for pair in pairs:
            all_pairs.append(
                {
                    "transliteration": pair.transliteration,
                    "translation": pair.translation,
                    "doc_id": pair.doc_id,
                    "anchor_valid": pair.anchor_valid,
                    "sentence_idx": pair.sentence_idx,
                    "word_start": pair.word_start,
                    "word_end": pair.word_end,
                }
            )
            if pair.anchor_valid:
                valid_count += 1
            else:
                invalid_count += 1

    result_df = pd.DataFrame(all_pairs)
    logger.info(
        "Extracted %d sentence pairs (anchor valid: %d, invalid: %d)",
        len(result_df),
        valid_count,
        invalid_count,
    )

    return result_df


def load_mixed_training_data(
    data_dir: str | Path,
    include_invalid_anchors: bool = False,
) -> pd.DataFrame:
    """Load both sentence-level and document-level training data.

    Documents with sentence info contribute sentence-level pairs.
    Documents without sentence info contribute document-level pairs.

    Args:
        data_dir: Path to data directory
        include_invalid_anchors: Whether to include sentences with invalid anchors

    Returns:
        DataFrame with columns: transliteration, translation, data_type
        where data_type is 'sentence' or 'document'
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"

    train_df = pd.read_csv(train_path)
    sentences_df = pd.read_csv(sentences_path)

    # Find documents with/without sentence info
    train_ids = set(train_df["oare_id"].dropna().unique())
    sentence_uuids = set(sentences_df["text_uuid"].dropna().unique())
    docs_with_sentences = train_ids & sentence_uuids
    docs_without_sentences = train_ids - sentence_uuids

    # Pre-group for O(1) lookups
    train_by_id = {oid: group.iloc[0] for oid, group in train_df.groupby("oare_id")}
    sentences_by_id = dict(iter(sentences_df.groupby("text_uuid")))

    all_data: list[dict] = []

    # Extract sentence-level data
    for doc_id in docs_with_sentences:
        doc_row = train_by_id[doc_id]
        doc_sentences = sentences_by_id[doc_id]

        pairs = extract_sentence_spans(
            doc_row["transliteration"],
            doc_sentences,
        )

        for pair in pairs:
            if not include_invalid_anchors and not pair.anchor_valid:
                continue
            all_data.append(
                {
                    "transliteration": pair.transliteration,
                    "translation": pair.translation,
                    "data_type": "sentence",
                    "doc_id": pair.doc_id,
                    "anchor_valid": pair.anchor_valid,
                    "sentence_idx": pair.sentence_idx,
                    "word_start": pair.word_start,
                    "word_end": pair.word_end,
                }
            )

    # Add document-level data
    for doc_id in docs_without_sentences:
        doc_row = train_by_id[doc_id]
        transliteration = doc_row["transliteration"]
        translation = doc_row["translation"]

        if pd.isna(transliteration) or pd.isna(translation):
            continue

        all_data.append(
            {
                "transliteration": transliteration,
                "translation": translation,
                "data_type": "document",
                "doc_id": str(doc_id),
            }
        )

    result_df = pd.DataFrame(all_data)

    sentence_count = len(result_df[result_df["data_type"] == "sentence"])
    doc_count = len(result_df[result_df["data_type"] == "document"])
    logger.info(
        "Mixed training data: %d total (%d sentences, %d documents)",
        len(result_df),
        sentence_count,
        doc_count,
    )

    return result_df


def load_extended_sentence_data(
    data_dir: str | Path,
    include_published_texts: bool = True,
    exclude_doc_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load sentence-level data from BOTH train.csv and published_texts.csv.

    Uses Sentences_Oare to extract sentence boundaries from published_texts.csv
    transliterations, paired with sentence-level translations from Sentences_Oare.

    This dramatically increases training data (~1000 → ~8000+ sentence pairs).

    Args:
        data_dir: Path to data directory
        include_published_texts: Whether to include extra data from published_texts
        exclude_doc_ids: Optional set of document IDs to exclude (for holdout test set)

    Returns:
        DataFrame with columns:
          - transliteration, translation
          - doc_id (oare_id), sentence_idx
          - anchor_valid, word_start, word_end
          - source (train vs published_texts)
    """
    data_dir = Path(data_dir)
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
    sentences_df = pd.read_csv(sentences_path)

    # First, get standard sentence data from train.csv
    train_df = pd.read_csv(data_dir / "train.csv")
    train_ids = set(train_df["oare_id"].dropna().unique())
    sentence_uuids = set(sentences_df["text_uuid"].dropna().unique())
    docs_with_sentences_in_train = train_ids & sentence_uuids

    # Pre-group for O(1) lookups
    train_by_id = {oid: group.iloc[0] for oid, group in train_df.groupby("oare_id")}
    sentences_by_id = dict(iter(sentences_df.groupby("text_uuid")))

    all_pairs: list[dict] = []

    # Extract from train.csv (same as load_sentence_level_data)
    for doc_id in docs_with_sentences_in_train:
        doc_row = train_by_id[doc_id]
        doc_sentences = sentences_by_id[doc_id]

        pairs = extract_sentence_spans(doc_row["transliteration"], doc_sentences)
        for pair in pairs:
            all_pairs.append(
                {
                    "transliteration": pair.transliteration,
                    "translation": pair.translation,
                    "doc_id": pair.doc_id,
                    "anchor_valid": pair.anchor_valid,
                    "sentence_idx": pair.sentence_idx,
                    "word_start": pair.word_start,
                    "word_end": pair.word_end,
                    "source": "train",
                }
            )

    logger.info("Extracted %d sentence pairs from train.csv", len(all_pairs))

    if include_published_texts:
        # Extract from published_texts.csv
        pub_texts = pd.read_csv(data_dir / "published_texts.csv")
        pub_oare_ids = set(pub_texts["oare_id"].dropna().unique())

        # Only use texts NOT already in train.csv (and not in holdout set)
        extra_ids = (pub_oare_ids & sentence_uuids) - train_ids
        if exclude_doc_ids:
            excluded_count = len(extra_ids & exclude_doc_ids)
            extra_ids = extra_ids - exclude_doc_ids
            logger.info(
                "Excluded %d holdout documents from published_texts",
                excluded_count,
            )

        # Pre-group published texts for O(1) lookups
        pub_by_id = {oid: group.iloc[0] for oid, group in pub_texts.groupby("oare_id")}

        extra_count = 0
        for doc_id in extra_ids:
            pub_row = pub_by_id.get(doc_id)
            if pub_row is None:
                continue
            transliteration = pub_row.get("transliteration")
            if pd.isna(transliteration) or not str(transliteration).strip():
                continue

            doc_sentences = sentences_by_id.get(doc_id)
            if doc_sentences is None:
                continue
            pairs = extract_sentence_spans(str(transliteration), doc_sentences)

            for pair in pairs:
                all_pairs.append(
                    {
                        "transliteration": pair.transliteration,
                        "translation": pair.translation,
                        "doc_id": pair.doc_id,
                        "anchor_valid": pair.anchor_valid,
                        "sentence_idx": pair.sentence_idx,
                        "word_start": pair.word_start,
                        "word_end": pair.word_end,
                        "source": "published_texts",
                    }
                )
                extra_count += 1

        logger.info(
            "Extracted %d additional sentence pairs from published_texts.csv",
            extra_count,
        )

    result_df = pd.DataFrame(all_pairs)

    if result_df.empty:
        logger.info("Total extended sentence data: 0 pairs")
    else:
        train_count = len(result_df[result_df["source"] == "train"])
        pub_count = len(result_df[result_df["source"] == "published_texts"])
        logger.info(
            "Total extended sentence data: %d pairs (train: %d, published_texts: %d)",
            len(result_df),
            train_count,
            pub_count,
        )

    return result_df


def get_extraction_statistics(data_dir: str | Path) -> dict:
    """Get detailed statistics about sentence extraction.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with extraction statistics
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"

    train_df = pd.read_csv(train_path)
    sentences_df = pd.read_csv(sentences_path)

    train_ids = set(train_df["oare_id"].dropna().unique())
    sentence_uuids = set(sentences_df["text_uuid"].dropna().unique())
    docs_with_sentences = train_ids & sentence_uuids

    # Pre-group for O(1) lookups
    train_by_id = {oid: group.iloc[0] for oid, group in train_df.groupby("oare_id")}
    sentences_by_id = dict(iter(sentences_df.groupby("text_uuid")))

    total_pairs = 0
    valid_anchors = 0
    invalid_anchors = 0
    sentence_lengths: list[int] = []

    for doc_id in docs_with_sentences:
        doc_row = train_by_id[doc_id]
        doc_sentences = sentences_by_id[doc_id]

        pairs = extract_sentence_spans(
            doc_row["transliteration"],
            doc_sentences,
        )

        for pair in pairs:
            total_pairs += 1
            if pair.anchor_valid:
                valid_anchors += 1
            else:
                invalid_anchors += 1
            sentence_lengths.append(len(pair.transliteration.split()))

    return {
        "total_train_docs": len(train_ids),
        "docs_with_sentence_info": len(docs_with_sentences),
        "docs_without_sentence_info": len(train_ids - sentence_uuids),
        "total_extracted_sentences": total_pairs,
        "valid_anchor_count": valid_anchors,
        "invalid_anchor_count": invalid_anchors,
        "anchor_match_rate": valid_anchors / total_pairs if total_pairs > 0 else 0,
        "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths)
        if sentence_lengths
        else 0,
        "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
        "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
    }
