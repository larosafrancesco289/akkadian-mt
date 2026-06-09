from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from akkadian_mt.config import DataConfig
from akkadian_mt.data.dataset import create_dataloaders, load_train_data


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ):
        del padding, truncation, return_tensors
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        input_ids = torch.ones((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _write_train_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"oare_id": "A", "transliteration": "a-na", "translation": "to me"},
            {"oare_id": "B", "transliteration": "ki-am", "translation": "thus"},
            {"oare_id": "C", "transliteration": "lu2", "translation": "man"},
        ]
    ).to_csv(path, index=False)


def test_load_train_data_excludes_holdout_docs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    _write_train_csv(train_path)

    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(["A"]), encoding="utf-8")

    cfg = DataConfig(
        data_dir=str(data_dir),
        data_mode="document",
        val_split=0.5,
        holdout_doc_ids_file=str(holdout_path),
    )

    train_df, val_df = load_train_data(cfg)
    ids = set(train_df["oare_id"].tolist()) | set(val_df["oare_id"].tolist())

    assert ids == {"B", "C"}
    assert "A" not in ids


def test_create_dataloaders_keeps_partial_batch_when_dataset_is_small() -> None:
    train_df = pd.DataFrame([{"transliteration": "a-na", "translation": "to me"}])
    val_df = pd.DataFrame([{"transliteration": "ki-am", "translation": "thus"}])

    cfg = DataConfig(max_source_length=8, max_target_length=8)
    tokenizer = _DummyTokenizer()

    train_loader, _ = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        config=cfg,
        batch_size=2,
        num_workers=0,
        dynamic_padding=False,
    )

    assert len(train_loader) == 1
    assert train_loader.pin_memory is torch.cuda.is_available()
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape[0] == 1


def test_load_train_data_drops_rows_that_become_blank_after_normalization(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"oare_id": "A", "transliteration": "?", "translation": "x"},
            {"oare_id": "B", "transliteration": "a-na", "translation": "to me"},
            {"oare_id": "C", "transliteration": "ki-am", "translation": "thus"},
        ]
    ).to_csv(data_dir / "train.csv", index=False)

    cfg = DataConfig(
        data_dir=str(data_dir),
        data_mode="document",
        val_split=0.5,
    )

    train_df, val_df = load_train_data(cfg)
    transliterations = set(train_df["transliteration"].tolist()) | set(
        val_df["transliteration"].tolist()
    )

    assert "" not in transliterations
    assert transliterations == {"a-na", "ki-am"}


def test_doc_id_split_requires_multiple_documents_after_filtering(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"oare_id": "A", "transliteration": "a-na", "translation": "to me"},
        ]
    ).to_csv(data_dir / "train.csv", index=False)

    cfg = DataConfig(
        data_dir=str(data_dir),
        data_mode="document",
        val_split=0.5,
        val_split_strategy="doc_id",
        val_group_column="oare_id",
    )

    try:
        load_train_data(cfg)
    except ValueError as exc:
        assert "at least 2 unique document IDs" in str(exc)
    else:
        raise AssertionError("Expected doc_id split to fail for a single remaining document")


def test_load_train_data_applies_sentence_length_filter_to_prepared_sentence_data(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        [
            {
                "oare_id": "A",
                "transliteration": "a-na be-li2-ia q2-bi4-ma",
                "translation": "Say to my lord.",
                "granularity": "sentence",
            },
            {
                "oare_id": "D",
                "transliteration": "ki-am awi-lum iq-bi",
                "translation": "Thus the man spoke.",
                "granularity": "sentence",
            },
            {
                "oare_id": "B",
                "transliteration": " ".join(["a-na"] * 160),
                "translation": "This row should be filtered for sentence length.",
                "granularity": "sentence",
            },
            {
                "oare_id": "C",
                "transliteration": " ".join(["ku3"] * 220),
                "translation": "Document-like prepared rows should survive when granularity is not sentence-filtered.",
                "granularity": "document",
            },
        ]
    ).to_csv(prepared_path, index=False)

    cfg = DataConfig(
        data_dir=str(data_dir),
        data_mode="prepared",
        prepared_file=str(prepared_path),
        prepared_granularity="sentence",
        val_split=0.5,
        min_sentence_words=2,
        max_sentence_words=150,
    )

    train_df, val_df = load_train_data(cfg)
    ids = set(train_df["oare_id"].tolist()) | set(val_df["oare_id"].tolist())

    assert ids == {"A", "D"}


def test_load_train_data_can_preserve_prebuilt_retrieval_sources(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.csv"
    original = "retrieved_source a-na retrieved_translation because of debt current_source a-na"
    pd.DataFrame(
        [
            {
                "oare_id": "A",
                "transliteration": original,
                "translation": "because of debt",
            },
            {
                "oare_id": "B",
                "transliteration": "retrieved_source ki-am retrieved_translation thus current_source ki-am",
                "translation": "thus",
            },
        ]
    ).to_csv(prepared_path, index=False)

    cfg = DataConfig(
        data_mode="prepared",
        prepared_file=str(prepared_path),
        skip_source_normalization=True,
        val_split=0.5,
    )

    train_df, val_df = load_train_data(cfg)
    sources = set(train_df["transliteration"].tolist()) | set(val_df["transliteration"].tolist())

    assert original in sources
    assert "becšuse" not in " ".join(sources)
