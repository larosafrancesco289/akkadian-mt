"""PyTorch Dataset and DataLoader creation for Akkadian translation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from akkadian_mt.config import DataConfig
from akkadian_mt.data.preprocessing import META_RE, clean_translation, normalize_transliteration
from akkadian_mt.data.sentence_extractor import (
    load_extended_sentence_data,
    load_mixed_training_data,
    load_sentence_level_data,
)

logger = logging.getLogger(__name__)


def ablation_kwargs(config: DataConfig) -> dict[str, bool]:
    """Build keyword arguments for ablation transforms in normalize_transliteration."""
    return {
        "do_strip_determinatives": config.strip_determinatives,
        "do_tag_sumerograms": config.tag_sumerograms,
        "do_remove_gaps": config.remove_gaps,
        "do_strip_homophone_subscripts": getattr(config, "strip_homophone_subscripts", True),
    }


def _apply_dictionary_augmentation(
    df: pd.DataFrame,
    data_dir: Path,
    max_glosses: int,
) -> pd.DataFrame:
    """Augment transliterations with dictionary glosses in-place."""
    from akkadian_mt.data.dictionary import augment_with_glosses, build_gloss_lookup

    gloss_lookup = build_gloss_lookup(data_dir)
    df["transliteration"] = df["transliteration"].apply(
        lambda x: augment_with_glosses(x, gloss_lookup, max_glosses=max_glosses)
    )
    logger.info("Applied dictionary gloss augmentation (max_glosses=%d)", max_glosses)
    return df


def _load_holdout_doc_ids(path: str | None) -> set[str]:
    """Load holdout doc IDs from JSON list, returning an empty set when unavailable."""
    if not path:
        return set()

    holdout_path = Path(path)
    try:
        payload = json.loads(holdout_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Holdout file not found: %s", holdout_path)
        return set()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid holdout_doc_ids_file JSON: {holdout_path} ({exc})") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Holdout file must contain a JSON list: {holdout_path}")

    holdout_ids = {str(item) for item in payload if str(item).strip()}
    logger.info("Loaded %d holdout doc IDs from %s", len(holdout_ids), holdout_path)
    return holdout_ids


def _resolve_doc_id_column(df: pd.DataFrame) -> str | None:
    """Resolve the canonical document-id column when available."""
    if "oare_id" in df.columns:
        return "oare_id"
    if "doc_id" in df.columns:
        return "doc_id"
    return None


def _exclude_holdout_docs(
    df: pd.DataFrame,
    holdout_ids: set[str],
) -> pd.DataFrame:
    """Exclude rows whose document IDs are in the holdout set."""
    if not holdout_ids:
        return df

    doc_col = _resolve_doc_id_column(df)
    if doc_col is None:
        logger.warning(
            "holdout_doc_ids_file is set but no doc-id column (oare_id/doc_id) exists; "
            "cannot exclude holdout docs",
        )
        return df

    before = len(df)
    filtered = df[~df[doc_col].astype(str).isin(holdout_ids)].copy()
    removed = before - len(filtered)
    if removed > 0:
        logger.info("Excluded %d holdout rows using %s", removed, doc_col)
    return filtered


def _drop_empty_text_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose normalized source or target text is blank."""
    src = df["transliteration"].astype(str).str.strip()
    tgt = df["translation"].astype(str).str.strip()
    keep_mask = src.ne("") & tgt.ne("")
    removed = int((~keep_mask).sum())
    if removed > 0:
        logger.info("Dropped %d rows with blank normalized text", removed)
    return df.loc[keep_mask].copy()


def _is_sentence_like_prepared_dataset(df: pd.DataFrame, config: DataConfig) -> bool:
    """Detect when a prepared CSV should receive sentence-length filtering."""
    if config.data_mode != "prepared":
        return False
    if config.prepared_granularity == "sentence":
        return True
    if "granularity" not in df.columns:
        return False
    gran = df["granularity"].dropna().astype(str)
    return not gran.empty and gran.eq("sentence").all()


def _split_train_validation(
    df: pd.DataFrame, config: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train/validation partitions with clearer failure modes."""
    if config.val_split_strategy != "doc_id":
        try:
            return train_test_split(df, test_size=config.val_split, random_state=42)
        except ValueError as exc:
            raise ValueError(
                "Random train/validation split failed. Adjust data filtering or data.val_split."
            ) from exc

    group_col = config.val_group_column
    if group_col not in df.columns:
        alt = "oare_id"
        if alt in df.columns and group_col == "doc_id":
            group_col = alt
        else:
            logger.warning(
                "val_split_strategy=doc_id but column %r not found; falling back to random split",
                config.val_group_column,
            )
            return _split_train_validation(
                df,
                DataConfig(
                    **{
                        **config.__dict__,
                        "val_split_strategy": "random",
                    }
                ),
            )

    doc_ids = df[group_col].dropna().astype(str).unique().tolist()
    if len(doc_ids) < 2:
        raise ValueError(
            "Doc-level validation split requires at least 2 unique document IDs after filtering."
        )

    try:
        train_docs, val_docs = train_test_split(
            doc_ids,
            test_size=config.val_split,
            random_state=42,
        )
    except ValueError as exc:
        raise ValueError(
            "Doc-level train/validation split failed. Adjust data filtering or data.val_split."
        ) from exc

    train_df = df[df[group_col].astype(str).isin(train_docs)].copy()
    val_df = df[df[group_col].astype(str).isin(val_docs)].copy()
    logger.info(
        "Doc-level split by %s: docs(train)=%d docs(val)=%d",
        group_col,
        len(train_docs),
        len(val_docs),
    )
    return train_df, val_df


class AkkadianDataset(Dataset):
    """Dataset for Akkadian transliteration → English translation pairs."""

    def __init__(
        self,
        sources: list[str],
        targets: list[str],
        tokenizer: Any,
        max_source_length: int = 256,
        max_target_length: int = 256,
        source_prefix: str = "",
    ) -> None:
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        source = self.source_prefix + self.sources[idx]
        target = self.targets[idx]

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_train_data(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data and split into train/val.

    Returns (train_df, val_df) with columns: transliteration, translation

    Uses data_mode config to determine data loading strategy:
    - "document": Original document-level data only
    - "sentence_only": Only sentence-level extracted pairs
    - "mixed": Sentence-level for docs with info, document-level for rest
    - "prepared": Load a pre-materialized CSV (see DataConfig.prepared_file)
    """
    data_dir = Path(config.data_dir)
    data_mode = config.data_mode
    holdout_doc_ids = _load_holdout_doc_ids(config.holdout_doc_ids_file)

    if data_mode == "prepared":
        if not config.prepared_file:
            raise ValueError("data_mode=prepared requires data.prepared_file to be set")
        prepared_path = Path(config.prepared_file)
        if not prepared_path.exists():
            raise FileNotFoundError(f"Prepared dataset not found: {prepared_path}")

        df = pd.read_csv(prepared_path)
        # Normalize doc id column name if present
        if "doc_id" in df.columns and "oare_id" not in df.columns:
            df = df.rename(columns={"doc_id": "oare_id"})

        if config.prepared_granularity and "granularity" in df.columns:
            before = len(df)
            df = df[df["granularity"] == config.prepared_granularity]
            logger.info(
                "Filtered prepared dataset by granularity=%s: %d -> %d",
                config.prepared_granularity,
                before,
                len(df),
            )

        logger.info("Loaded %d prepared examples from %s", len(df), prepared_path)
    elif data_mode == "sentence_only":
        df = load_sentence_level_data(data_dir)
        logger.info("Loaded %d sentence-level examples", len(df))
    elif data_mode == "extended_sentence":
        df = load_extended_sentence_data(
            data_dir,
            include_published_texts=True,
            exclude_doc_ids=holdout_doc_ids or None,
        )
        logger.info("Loaded %d extended sentence-level examples", len(df))
    elif data_mode == "mixed":
        include_invalid = config.include_invalid_anchors
        df = load_mixed_training_data(data_dir, include_invalid_anchors=include_invalid)
        logger.info("Loaded %d mixed examples (sentence + document)", len(df))
    else:
        # Default: document-level only
        train_path = data_dir / config.train_file
        df = pd.read_csv(train_path)
        logger.info("Loaded %d document-level examples from %s", len(df), train_path)

    df = _exclude_holdout_docs(df, holdout_doc_ids)
    df = df.dropna(subset=["transliteration", "translation"])

    # Apply length filters for sentence mode
    if data_mode in (
        "sentence_only",
        "mixed",
        "extended_sentence",
    ) or _is_sentence_like_prepared_dataset(df, config):
        min_words = config.min_sentence_words
        max_words = config.max_sentence_words
        df["word_count"] = df["transliteration"].apply(lambda x: len(str(x).split()))
        before = len(df)
        df = df[(df["word_count"] >= min_words) & (df["word_count"] <= max_words)]
        after = len(df)
        if before != after:
            logger.info("Filtered by word count: %d -> %d", before, after)
        df = df.drop(columns=["word_count"])

    # Apply quality filtering before normalization (patterns match raw text better)
    if config.filter_quality:
        before = len(df)
        df = df[df["translation"].astype(str).str.len() >= config.min_translation_chars]
        src_len = df["transliteration"].astype(str).str.len()
        tgt_len = df["translation"].astype(str).str.len()
        ratio = src_len.combine(tgt_len, max) / src_len.combine(tgt_len, min).clip(lower=1)
        df = df[ratio <= config.max_length_ratio]
        df = df[~df["translation"].astype(str).apply(lambda x: bool(META_RE.search(x)))]
        after = len(df)
        logger.info(
            "Quality filter: %d -> %d (removed %d noisy pairs)", before, after, before - after
        )

    # Some prepared datasets already contain final source strings and should not
    # be passed back through Akkadian-specific normalization.
    if config.skip_source_normalization:
        df["transliteration"] = df["transliteration"].astype(str)
    else:
        kwargs = ablation_kwargs(config)
        df["transliteration"] = df["transliteration"].apply(
            lambda x: normalize_transliteration(x, **kwargs)
        )
    df["translation"] = df["translation"].apply(clean_translation)

    if config.use_dictionary_gloss:
        df = _apply_dictionary_augmentation(df, data_dir, config.max_glosses)

    df = _drop_empty_text_pairs(df)

    # Split into train/val
    train_df, val_df = _split_train_validation(df, config)

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Train/validation split produced an empty partition. "
            "Adjust data filtering or data.val_split."
        )

    logger.info("Train: %d, Val: %d", len(train_df), len(val_df))

    return train_df, val_df


def load_test_data(config: DataConfig) -> pd.DataFrame:
    """Load test data for generating submissions."""
    data_dir = Path(config.data_dir)
    test_path = data_dir / config.test_file

    df = pd.read_csv(test_path)
    kwargs = ablation_kwargs(config)
    df["transliteration"] = df["transliteration"].apply(
        lambda x: normalize_transliteration(x, **kwargs)
    )

    if config.use_dictionary_gloss:
        df = _apply_dictionary_augmentation(df, data_dir, config.max_glosses)

    logger.info("Loaded %d test examples from %s", len(df), test_path)
    return df


class DynamicPaddingDataset(Dataset):
    """Dataset that returns raw text pairs — padding is deferred to collate_fn."""

    def __init__(
        self,
        sources: list[str],
        targets: list[str],
        source_prefix: str = "",
    ) -> None:
        self.sources = sources
        self.targets = targets
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "source": self.source_prefix + self.sources[idx],
            "target": self.targets[idx],
        }


class PreTokenizedDataset(Dataset):
    """Dataset that pre-tokenizes all examples at init for multi-worker loading.

    Tokenization happens once upfront. Workers return pre-computed integer lists,
    and padding is deferred to the collate function.
    """

    def __init__(
        self,
        sources: list[str],
        targets: list[str],
        tokenizer: Any,
        max_source_length: int = 512,
        max_target_length: int = 256,
        source_prefix: str = "",
    ) -> None:
        if len(sources) != len(targets):
            raise ValueError(
                f"Source/target length mismatch: {len(sources)} sources vs {len(targets)} targets"
            )
        if not sources:
            raise ValueError("Cannot build PreTokenizedDataset from an empty dataset")

        self.pad_token_id = tokenizer.pad_token_id

        logger.info("Pre-tokenizing %d examples...", len(sources))
        prefixed = [source_prefix + s for s in sources]

        src_enc = tokenizer(
            prefixed,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )
        tgt_enc = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )

        self.src_ids: list[list[int]] = src_enc["input_ids"]
        self.tgt_ids: list[list[int]] = tgt_enc["input_ids"]
        self.src_lengths: list[int] = [len(ids) for ids in self.src_ids]
        logger.info(
            "Pre-tokenized %d examples (src len range: %d-%d)",
            len(sources),
            min(self.src_lengths),
            max(self.src_lengths),
        )

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "src_ids": self.src_ids[idx],
            "tgt_ids": self.tgt_ids[idx],
            "pad_token_id": self.pad_token_id,
        }


def _pretokenized_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate pre-tokenized examples with dynamic padding."""
    pad_id = batch[0]["pad_token_id"]

    src_tensors = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
    tgt_tensors = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]

    input_ids = pad_sequence(src_tensors, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(tgt_tensors, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SortishSampler(Sampler[int]):
    """Sampler that groups examples by source length within random megabatches.

    Reduces padding waste by 15-30% while preserving stochasticity across epochs.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        mega_factor: int = 50,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega_size = batch_size * mega_factor

    def __iter__(self):  # type: ignore[override]
        indices = np.random.permutation(len(self.lengths))
        for start in range(0, len(indices), self.mega_size):
            mega = indices[start : start + self.mega_size]
            mega_sorted = mega[np.argsort([self.lengths[i] for i in mega])]
            yield from mega_sorted.tolist()

    def __len__(self) -> int:
        return len(self.lengths)


def _make_dynamic_collate_fn(
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
) -> Any:
    """Create a collate function that pads to the longest sequence in the batch."""

    def collate_fn(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]

        source_encoding = tokenizer(
            sources,
            max_length=max_source_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = tokenizer(
            targets,
            max_length=max_target_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": labels,
        }

    return collate_fn


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: Any,
    config: DataConfig,
    batch_size: int = 8,
    num_workers: int = 4,
    dynamic_padding: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        dynamic_padding: If True, pad to longest in batch instead of max_length.
                        Reduces wasted compute significantly for ByT5.
    """
    source_prefix = config.source_prefix

    if dynamic_padding:
        # Pre-tokenize for multi-worker loading with dynamic padding
        train_dataset = PreTokenizedDataset(
            sources=train_df["transliteration"].tolist(),
            targets=train_df["translation"].tolist(),
            tokenizer=tokenizer,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
            source_prefix=source_prefix,
        )
        val_dataset = PreTokenizedDataset(
            sources=val_df["transliteration"].tolist(),
            targets=val_df["translation"].tolist(),
            tokenizer=tokenizer,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
            source_prefix=source_prefix,
        )
        collate_fn = _pretokenized_collate_fn
        effective_workers = num_workers

        # SortishSampler for length-bucketed batching (reduces padding waste)
        train_sampler = SortishSampler(
            train_dataset.src_lengths,
            batch_size=batch_size,
        )
    else:
        train_dataset = AkkadianDataset(
            sources=train_df["transliteration"].tolist(),
            targets=train_df["translation"].tolist(),
            tokenizer=tokenizer,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
            source_prefix=source_prefix,
        )
        val_dataset = AkkadianDataset(
            sources=val_df["transliteration"].tolist(),
            targets=val_df["translation"].tolist(),
            tokenizer=tokenizer,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
            source_prefix=source_prefix,
        )
        collate_fn = None
        effective_workers = num_workers
        train_sampler = None

    worker_kwargs: dict[str, Any] = {}
    if effective_workers > 0:
        worker_kwargs["persistent_workers"] = True
        worker_kwargs["prefetch_factor"] = 2

    drop_last = len(train_dataset) >= batch_size
    if not drop_last:
        logger.info(
            "Train set smaller than batch_size (%d < %d); keeping last partial batch",
            len(train_dataset),
            batch_size,
        )
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **worker_kwargs,
    )

    return train_loader, val_loader
