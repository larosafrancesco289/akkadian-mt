from __future__ import annotations

import pandas as pd

from akkadian_mt.data.coursework_variants import (
    annotate_numerals_in_text,
    build_genre_conditioned_frame,
    build_numeral_normalized_frame,
    build_retrieval_augmented_frames,
    canonicalize_genre,
    read_holdout_ids,
)


def test_canonicalize_genre_normalizes_spacing_and_case() -> None:
    assert canonicalize_genre("Debt note") == "debt_note"
    assert canonicalize_genre("  LETTER  ") == "letter"
    assert canonicalize_genre("") == "unknown"


def test_annotate_numerals_in_text_adds_fractional_aliases() -> None:
    text, count = annotate_numerals_in_text("ma-na 7.5 GÍN 1/2")

    assert text == "ma-na 7.5 num_value_7_1_2 GÍN 1/2 num_value_1_2"
    assert count == 2


def test_build_genre_conditioned_frame_prepends_genre_tag() -> None:
    df = pd.DataFrame([{"transliteration": "a-na", "translation": "to", "genre": "Debt note"}])

    out = build_genre_conditioned_frame(df)

    assert out.loc[0, "genre_tag"] == "debt_note"
    assert out.loc[0, "transliteration"].startswith("genre_debt_note source_text ")


def test_build_numeral_normalized_frame_tracks_annotation_count() -> None:
    df = pd.DataFrame([{"transliteration": "ma-na 7.5 GÍN", "translation": "1 mina 7 1/2 shekels"}])

    out = build_numeral_normalized_frame(df)

    assert out.loc[0, "numeral_annotation_count"] == 1
    assert "num_value_7_1_2" in out.loc[0, "transliteration"]


def test_build_retrieval_augmented_frames_uses_training_bank_for_eval() -> None:
    train_df = pd.DataFrame(
        [
            {"oare_id": "A", "transliteration": "a-na be-li2-ia", "translation": "To my lord."},
            {"oare_id": "B", "transliteration": "um-ma šu-ut-ma", "translation": "Thus he said."},
            {
                "oare_id": "C",
                "transliteration": "1 ma-na KÙ.BABBAR",
                "translation": "1 mina of silver.",
            },
        ]
    )
    eval_df = pd.DataFrame(
        [
            {
                "doc_id": "X",
                "transliteration": "1 ma-na KÙ.BABBAR",
                "translation": "1 mina of silver.",
            }
        ]
    )

    outputs = build_retrieval_augmented_frames(
        train_df=train_df,
        eval_dfs={"primary": eval_df, "secondary": eval_df},
        holdout_doc_ids=set(),
    )

    primary = outputs["primary"]
    assert "retrieved_source" in primary.loc[0, "transliteration"]
    assert primary.loc[0, "retrieved_translation"] == "1 mina of silver."


def test_build_retrieval_augmented_frames_supports_source_only_mode() -> None:
    train_df = pd.DataFrame(
        [
            {"oare_id": "A", "transliteration": "a-na be-li2-ia", "translation": "To my lord."},
            {"oare_id": "B", "transliteration": "um-ma šu-ut-ma", "translation": "Thus he said."},
        ]
    )
    eval_df = pd.DataFrame(
        [{"doc_id": "X", "transliteration": "a-na be-li2-ia", "translation": "To my lord."}]
    )

    outputs = build_retrieval_augmented_frames(
        train_df=train_df,
        eval_dfs={"primary": eval_df, "secondary": eval_df},
        holdout_doc_ids=set(),
        include_retrieved_translation=False,
        retrieved_translation_max_bytes=0,
    )

    primary = outputs["primary"]
    assert "retrieved_source" in primary.loc[0, "transliteration"]
    assert "retrieved_translation" not in primary.loc[0, "transliteration"]


def test_read_holdout_ids_requires_existing_union_file(tmp_path) -> None:
    missing = tmp_path / "missing_holdout.json"

    try:
        read_holdout_ids(missing)
    except FileNotFoundError as exc:
        assert "build_coursework_holdout_union.py" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing holdout union")
