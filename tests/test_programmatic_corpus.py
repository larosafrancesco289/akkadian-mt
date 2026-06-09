from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory

from akkadian_mt.data.programmatic_corpus import (
    build_bundle,
    filter_non_english_sentence_targets,
    filter_publication_candidates,
    match_publication_to_sentence,
    repair_sentence_start_indices,
    score_domain_candidate,
    split_sentence_gold_by_quality,
)


def test_repair_sentence_start_indices_fixes_local_anchor_drift() -> None:
    words = ["x", "um-ma", "a-na", "be-li2", "qi2-bi4-ma"]
    rows = pd.DataFrame(
        [
            {"first_word_number": 1, "first_word_spelling": "um-ma"},
            {"first_word_number": 4, "first_word_spelling": "be-li₂"},
        ]
    )

    starts = repair_sentence_start_indices(words, rows, search_radius=2)

    assert starts == [1, 3]


def test_match_publication_to_sentence_requires_clear_best_match() -> None:
    refs = [
        {"translation": "Send 2 minas of silver to us.", "transliteration": "x", "sentence_idx": 0},
        {"translation": "Seal of Puzur-Ashur.", "transliteration": "y", "sentence_idx": 1},
    ]

    match = match_publication_to_sentence(
        "Send 2 minas of silver to us now.",
        refs,
        similarity_min=0.3,
        similarity_margin=0.05,
        min_word_ratio=0.5,
        max_word_ratio=2.0,
    )

    assert match is not None
    assert match["ref"]["sentence_idx"] == 0
    assert match["score"] > 0.7


def test_filter_publication_candidates_drops_bad_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "oare_id": "a",
                "translation": "Send 5 minas of silver to me.",
                "translation_language": "en",
                "extraction_pattern": "quoted",
            },
            {
                "oare_id": "b",
                "translation": "cf. Dercksen 1996 p. 12",
                "translation_language": "en",
                "extraction_pattern": "quoted",
            },
            {
                "oare_id": "c",
                "translation": "enveloppe fragmentaire",
                "translation_language": "fr",
                "extraction_pattern": "quoted",
            },
        ]
    )

    filtered = filter_publication_candidates(
        df,
        confidence_min=0.8,
        max_repeat=2,
        min_chars=25,
        max_chars=260,
    )

    assert filtered["oare_id"].tolist() == ["a"]


def test_score_domain_candidate_prefers_competition_like_text() -> None:
    source_vocab = {"silver", "tin", "send", "mina"}
    target_logodds = {"silver": 1.0, "tin": 0.8, "send": 0.6, "king": -1.0}

    good = score_domain_candidate(
        "send 2 minas silver",
        "Send silver and tin.",
        source_vocab=source_vocab,
        target_logodds=target_logodds,
    )
    bad = score_domain_candidate(
        "palace king battle",
        "The king marched to battle.",
        source_vocab=source_vocab,
        target_logodds=target_logodds,
    )

    assert good > bad


def test_split_sentence_gold_by_quality_quarantines_overlong_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "oare_id": "core",
                "transliteration": " ".join(["a-na"] * 20),
                "translation": "Send the silver to me.",
                "source": "train_sentence_gold",
                "tier": "gold",
                "granularity": "sentence",
                "confidence": 1.0,
            },
            {
                "oare_id": "quarantine",
                "transliteration": " ".join(["a-na"] * 130),
                "translation": "To Alahum from Puzur-Assur.",
                "source": "train_sentence_gold",
                "tier": "gold",
                "granularity": "sentence",
                "confidence": 1.0,
            },
        ]
    )

    core, quarantine = split_sentence_gold_by_quality(
        df,
        max_source_words=100,
        max_source_bytes=640,
    )

    assert core["oare_id"].tolist() == ["core"]
    assert quarantine["oare_id"].tolist() == ["quarantine"]
    assert quarantine["tier"].tolist() == ["gold_quarantine"]
    assert "overlong_sentence_words" in quarantine["quality_flags"].iloc[0]


def test_split_sentence_gold_by_quality_recovers_formulaic_headers() -> None:
    df = pd.DataFrame(
        [
            {
                "oare_id": "header-doc",
                "transliteration": (
                    "um-ma a-mur-IŠTAR-ma a-na a-zu-na-a en-nam-a-šùr ù ar-ší-ah "
                    "qí-bi-ma a-na en-nam-a-šùr qí-bi-ma KÙ.BABBAR 5 ma-na KÙ.BABBAR "
                    "iš-tù 13 ša-na-tim a-na ṣí-ib-tim i-lá-kà-kum"
                ),
                "translation": "From Amur-Ištar to Azunaya, Ennam-Aššur, and Aršiaḫ; specifically to Ennam-Aššur:",
                "source": "published_texts_sentence_gold",
                "tier": "gold",
                "granularity": "sentence",
                "confidence": 1.0,
            },
        ]
    )

    core, quarantine = split_sentence_gold_by_quality(
        df,
        max_source_words=100,
        max_source_bytes=110,
    )

    assert quarantine.empty
    assert core["tier"].tolist() == ["gold_recovered"]
    assert core["recovered_from_quarantine"].tolist() == [True]
    assert core["recovery_strategy"].tolist() == ["formulaic_header"]
    assert core["transliteration"].iloc[0].endswith("a-na en-nam-a-šùr qí-bi-ma")
    assert "KÙ.BABBAR 5 ma-na" not in core["transliteration"].iloc[0]
    assert len(core["transliteration"].iloc[0].encode("utf-8")) <= 110


def test_split_sentence_gold_by_quality_recovers_seal_lists() -> None:
    df = pd.DataFrame(
        [
            {
                "oare_id": "seal-doc",
                "transliteration": (
                    "KIŠIB ku-ku-a DUMU ma-nu-ki-ì-lí-a KIŠIB da-da-a DUMU bu-zi "
                    "KIŠIB ì-lí-dan DUMU i-dí-a-šùr 21 GÚ ša-áp-tám ša kà-né-eš"
                ),
                "translation": "Seals: Kukuwa s. Mannum-kī-iliya, Dadaya s. Buzi, Ilī-dān s. Idī-Aššur.",
                "source": "published_texts_sentence_gold",
                "tier": "gold",
                "granularity": "sentence",
                "confidence": 1.0,
            },
        ]
    )

    core, quarantine = split_sentence_gold_by_quality(
        df,
        max_source_words=100,
        max_source_bytes=110,
    )

    assert quarantine.empty
    assert core["tier"].tolist() == ["gold_recovered"]
    assert core["recovery_strategy"].tolist() == ["seal_list"]
    assert core["transliteration"].iloc[0].startswith("KIŠIB ku-ku-a DUMU")
    assert "21 GÚ" not in core["transliteration"].iloc[0]
    assert len(core["transliteration"].iloc[0].encode("utf-8")) <= 110


def test_filter_non_english_sentence_targets_drops_foreign_rows_but_keeps_short_english(
    monkeypatch,
) -> None:
    df = pd.DataFrame(
        [
            {
                "oare_id": "german",
                "transliteration": "a-na be-li2 q2-bi4-ma",
                "translation": "Was den Dadā’a betrifft, so hat der hier dein Kupfer gegeben.",
                "source": "published_texts_sentence_gold",
            },
            {
                "oare_id": "english",
                "transliteration": "li-mu-um e-na-sú-en",
                "translation": "Eponymate: Enna-Suen.",
                "source": "published_texts_sentence_gold",
            },
            {
                "oare_id": "train",
                "transliteration": "a-na be-li2 q2-bi4-ma",
                "translation": "A clearly English training row.",
                "source": "train_sentence_gold",
            },
        ]
    )

    def fake_detect_language(text: str) -> str:
        if text.startswith("Was den"):
            return "de"
        if text.startswith("Eponymate"):
            return "pt"
        return "en"

    monkeypatch.setattr(
        "akkadian_mt.data.extract_publication_translations.detect_language",
        fake_detect_language,
    )

    filtered = filter_non_english_sentence_targets(df)

    assert filtered["oare_id"].tolist() == ["english", "train"]
    assert (
        filtered.loc[filtered["oare_id"] == "english", "translation_foreign_stopword_hits"].iloc[0]
        == 0
    )


def test_publication_language_detection_is_seeded_for_deterministic_rebuilds() -> None:
    importlib.import_module("akkadian_mt.data.extract_publication_translations")

    assert DetectorFactory.seed == 0


def test_build_bundle_filters_holdouts_before_dedup_and_rebuilds_evalsafe_oracc(
    tmp_path: Path, monkeypatch
) -> None:
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(["holdout-doc"]), encoding="utf-8")

    sentence_gold_raw = pd.DataFrame(
        [
            {
                "oare_id": "holdout-doc",
                "transliteration": "same pair",
                "translation": "same translation",
                "source": "train_sentence_gold",
                "tier": "gold",
                "granularity": "sentence",
            },
            {
                "oare_id": "safe-doc",
                "transliteration": "same pair",
                "translation": "same translation",
                "source": "train_sentence_gold",
                "tier": "gold",
                "granularity": "sentence",
            },
        ]
    )
    document_aux_raw = pd.DataFrame(
        [
            {
                "oare_id": "holdout-doc",
                "transliteration": "aux holdout",
                "translation": "aux holdout translation",
                "source": "train_document_aux",
                "tier": "aux_document",
                "granularity": "document",
            },
            {
                "oare_id": "safe-doc",
                "transliteration": "aux safe",
                "translation": "aux safe translation",
                "source": "train_document_aux",
                "tier": "aux_document",
                "granularity": "document",
            },
        ]
    )

    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_repaired_sentence_gold",
        lambda data_dir, search_radius: sentence_gold_raw.copy(),
    )
    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_document_aux",
        lambda data_dir: document_aux_raw.copy(),
    )
    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_publication_sentence_aux",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    captured_competition_ids: list[set[str]] = []

    def fake_build_oracc(
        oracc_path: Path,
        competition_df: pd.DataFrame,
        *,
        min_score: float,
        min_keyword_hits: int,
        min_target_words: int,
        min_source_words: int,
    ) -> pd.DataFrame:
        del oracc_path, min_score, min_keyword_hits, min_target_words, min_source_words
        ids = set(competition_df["oare_id"].astype(str))
        captured_competition_ids.append(ids)
        marker = "evalsafe" if "holdout-doc" not in ids else "full"
        return pd.DataFrame(
            [
                {
                    "oare_id": "",
                    "transliteration": f"oracc {marker}",
                    "translation": f"oracc {marker} translation",
                    "source": "oracc_domain_filtered",
                    "tier": "pretrain_domain",
                    "granularity": "sentence",
                }
            ]
        )

    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_oracc_domain_pretrain", fake_build_oracc
    )

    args = Namespace(
        data_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        holdout_doc_ids=str(holdout_path),
        search_radius=3,
        min_translation_chars=10,
        max_length_ratio=10.0,
        max_sentence_source_words=100,
        max_sentence_source_bytes=640,
        publication_cache=str(tmp_path / "pub.csv"),
        force_publication_extract=False,
        publication_sample_rows=None,
        publication_similarity_min=0.75,
        publication_similarity_margin=0.10,
        publication_min_word_ratio=0.5,
        publication_max_word_ratio=2.0,
        publication_confidence_min=0.80,
        publication_max_repeat=2,
        legacy_publication_aligned_path=str(tmp_path / "legacy.csv"),
        legacy_publication_alignment_method="transliteration_span",
        legacy_publication_alignment_score_min=0.90,
        legacy_publication_heuristic_confidence_min=0.65,
        publication_min_chars=25,
        publication_max_chars=260,
        publication_review_n=25,
        oracc_path=str(tmp_path / "oracc.csv"),
        oracc_min_score=0.30,
        oracc_min_keyword_hits=2,
        oracc_min_target_words=8,
        oracc_min_source_words=3,
        no_oracc=False,
    )

    build_bundle(args)

    sentence_gold_safe = pd.read_csv(tmp_path / "out" / "sentence_gold_evalsafe.csv")
    stage1_safe = pd.read_csv(tmp_path / "out" / "stage1_curriculum_evalsafe.csv")

    assert sentence_gold_safe["oare_id"].tolist() == ["safe-doc"]
    assert set(stage1_safe["transliteration"]) == {"same pair", "aux safe"}
    assert captured_competition_ids == [{"holdout-doc", "safe-doc"}, {"safe-doc"}]


def test_build_bundle_removes_stale_optional_artifacts(tmp_path: Path, monkeypatch) -> None:
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps([]), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale_oracc = out_dir / "oracc_domain_pretrain.csv"
    stale_oracc.write_text("stale\n", encoding="utf-8")
    stale_oracc_safe = out_dir / "oracc_domain_pretrain_evalsafe.csv"
    stale_oracc_safe.write_text("stale\n", encoding="utf-8")
    stale_review = out_dir / "publication_alignment_review.csv"
    stale_review.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_repaired_sentence_gold",
        lambda data_dir, search_radius: pd.DataFrame(
            [
                {
                    "oare_id": "safe-doc",
                    "transliteration": "safe pair",
                    "translation": "safe translation",
                    "source": "train_sentence_gold",
                    "tier": "gold",
                    "granularity": "sentence",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_document_aux",
        lambda data_dir: pd.DataFrame(
            [
                {
                    "oare_id": "safe-doc",
                    "transliteration": "aux pair",
                    "translation": "aux translation",
                    "source": "train_document_aux",
                    "tier": "aux_document",
                    "granularity": "document",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_publication_sentence_aux",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    args = Namespace(
        data_dir=str(tmp_path),
        out_dir=str(out_dir),
        holdout_doc_ids=str(holdout_path),
        search_radius=3,
        min_translation_chars=10,
        max_length_ratio=10.0,
        max_sentence_source_words=100,
        max_sentence_source_bytes=640,
        publication_cache=str(tmp_path / "pub.csv"),
        force_publication_extract=False,
        publication_sample_rows=None,
        publication_similarity_min=0.75,
        publication_similarity_margin=0.10,
        publication_min_word_ratio=0.5,
        publication_max_word_ratio=2.0,
        publication_confidence_min=0.80,
        publication_max_repeat=2,
        legacy_publication_aligned_path=str(tmp_path / "legacy.csv"),
        legacy_publication_alignment_method="transliteration_span",
        legacy_publication_alignment_score_min=0.90,
        legacy_publication_heuristic_confidence_min=0.65,
        publication_min_chars=25,
        publication_max_chars=260,
        publication_review_n=25,
        oracc_path=str(tmp_path / "oracc.csv"),
        oracc_min_score=0.30,
        oracc_min_keyword_hits=2,
        oracc_min_target_words=8,
        oracc_min_source_words=3,
        no_oracc=True,
    )

    build_bundle(args)

    assert not stale_oracc.exists()
    assert not stale_oracc_safe.exists()
    assert not stale_review.exists()


def test_build_bundle_adds_strict_legacy_publication_rows(tmp_path: Path, monkeypatch) -> None:
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(["holdout-doc"]), encoding="utf-8")
    legacy_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "oare_id": "safe-doc",
                "transliteration": "a-na be-li2 q2-bi4-ma",
                "translation": "Say to my lord.",
                "alignment_method": "transliteration_span",
                "alignment_score": 0.97,
                "heuristic_confidence": 1.0,
                "original_language": "en",
                "translation_type": "direct",
                "pdf_name": "x.pdf",
                "page": 1,
            },
            {
                "oare_id": "holdout-doc",
                "transliteration": "ki-am iq-bi",
                "translation": "Thus he spoke.",
                "alignment_method": "transliteration_span",
                "alignment_score": 0.99,
                "heuristic_confidence": 1.0,
                "original_language": "en",
                "translation_type": "direct",
                "pdf_name": "y.pdf",
                "page": 2,
            },
            {
                "oare_id": "safe-doc",
                "transliteration": "bad candidate",
                "translation": "This should stay out.",
                "alignment_method": "sentences_oare",
                "alignment_score": 0.99,
                "heuristic_confidence": 1.0,
                "original_language": "en",
                "translation_type": "direct",
                "pdf_name": "z.pdf",
                "page": 3,
            },
            {
                "oare_id": "safe-doc",
                "transliteration": " ".join(["a-na"] * 200),
                "translation": "This one is too long for Stage 2.",
                "alignment_method": "transliteration_span",
                "alignment_score": 0.99,
                "heuristic_confidence": 1.0,
                "original_language": "en",
                "translation_type": "direct",
                "pdf_name": "long.pdf",
                "page": 4,
            },
        ]
    ).to_csv(legacy_path, index=False)

    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_repaired_sentence_gold",
        lambda data_dir, search_radius: pd.DataFrame(
            [
                {
                    "oare_id": "base-doc",
                    "transliteration": "base pair",
                    "translation": "base translation",
                    "source": "train_sentence_gold",
                    "tier": "gold",
                    "granularity": "sentence",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_document_aux",
        lambda data_dir: pd.DataFrame(
            [
                {
                    "oare_id": "aux-doc",
                    "transliteration": "aux pair",
                    "translation": "aux translation",
                    "source": "train_document_aux",
                    "tier": "aux_document",
                    "granularity": "document",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "akkadian_mt.data.programmatic_corpus.build_publication_sentence_aux",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    args = Namespace(
        data_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        holdout_doc_ids=str(holdout_path),
        search_radius=3,
        min_translation_chars=10,
        max_length_ratio=10.0,
        max_sentence_source_words=100,
        max_sentence_source_bytes=640,
        publication_cache=str(tmp_path / "pub.csv"),
        force_publication_extract=False,
        publication_sample_rows=None,
        publication_similarity_min=0.75,
        publication_similarity_margin=0.10,
        publication_min_word_ratio=0.5,
        publication_max_word_ratio=2.0,
        publication_confidence_min=0.80,
        publication_max_repeat=2,
        legacy_publication_aligned_path=str(legacy_path),
        legacy_publication_alignment_method="transliteration_span",
        legacy_publication_alignment_score_min=0.90,
        legacy_publication_heuristic_confidence_min=0.65,
        publication_min_chars=10,
        publication_max_chars=260,
        publication_review_n=25,
        oracc_path=str(tmp_path / "oracc.csv"),
        oracc_min_score=0.30,
        oracc_min_keyword_hits=2,
        oracc_min_target_words=8,
        oracc_min_source_words=3,
        no_oracc=True,
    )

    build_bundle(args)

    legacy_safe = pd.read_csv(tmp_path / "out" / "publication_sentence_legacy_evalsafe.csv")
    stage2_safe = pd.read_csv(tmp_path / "out" / "stage2_finetune_evalsafe.csv")

    assert legacy_safe["oare_id"].tolist() == ["safe-doc"]
    assert legacy_safe["source"].tolist() == ["publications_llm_aligned"]
    assert legacy_safe["tier"].tolist() == ["experimental_publication_llm"]
    assert set(stage2_safe["translation"]) == {"base translation", "Say to my lord."}
