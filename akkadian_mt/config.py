"""Configuration system using dataclasses with YAML loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

logger = logging.getLogger(__name__)

# Special tokens added to the tokenizer during training
SPECIAL_TOKENS = ["<gap>", "<big_gap>", "<SUM>", "</SUM>"]


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sentences_oare_file: str = "Sentences_Oare_FirstWord_LinNum.csv"
    ebl_dictionary_file: str = "eBL_Dictionary.csv"
    oa_lexicon_file: str = "OA_Lexicon_eBL.csv"
    sample_submission_file: str = "sample_submission.csv"
    val_split: float = 0.1
    # Validation split strategy:
    # - "random": row-level split (default)
    # - "doc_id": group split by document id (requires `doc_id` or `oare_id` column)
    val_split_strategy: str = "random"
    val_group_column: str = "doc_id"
    use_sentences_oare: bool = True
    max_source_length: int = 256
    max_target_length: int = 256
    # Sentence-level training options
    data_mode: str = "document"  # "sentence_only", "document_only", "mixed"
    min_sentence_words: int = 2
    max_sentence_words: int = 150
    include_invalid_anchors: bool = False
    # Source prefix for T5-family models
    source_prefix: str = ""
    # Dictionary augmentation
    use_dictionary_gloss: bool = False  # Append dictionary glosses to source
    max_glosses: int = 8  # Max glosses per sentence
    # Data quality filtering
    filter_quality: bool = False  # Filter out noisy/bad pairs
    min_translation_chars: int = 5  # Min translation length to keep
    max_length_ratio: float = 10.0  # Max src/tgt character length ratio
    # Ablation flags (for RQ experiments)
    strip_determinatives: bool = False  # RQ1: Remove {d}, {m}, {ki} etc.
    tag_sumerograms: bool = False  # RQ2: Wrap Sumerograms with <SUM>...</SUM>
    remove_gaps: bool = False  # RQ4: Don't use <gap>/<big_gap> tokens
    strip_homophone_subscripts: bool = True  # Strip scholarly subscript digits (bi4→bi)

    # Prepared dataset support (materialized CSVs under data/processed/)
    # Use with: data_mode: prepared
    prepared_file: str | None = (
        None  # Path to prepared CSV (e.g., data/processed/extended_sentence_pairs_clean.csv)
    )
    prepared_granularity: str | None = (
        None  # Optional filter if CSV has `granularity` column (e.g., "sentence")
    )
    skip_source_normalization: bool = False
    eval_primary_test_file: str | None = None
    eval_secondary_test_file: str | None = None

    # Holdout document exclusion (for independent test set evaluation)
    holdout_doc_ids_file: str | None = None  # Path to JSON list of doc IDs to exclude from training


@dataclass
class ModelConfig:
    model_type: str = "hf_seq2seq"  # "hf_seq2seq" or "lstm_baseline"
    model_name: str = "google/mt5-small"
    # LSTM baseline params
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    vocab_size: int = 32000


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    bf16: bool = False  # Use BF16 instead of FP16 (better for Blackwell/Ampere+)
    gradient_checkpointing: bool = False  # Trade compute for memory
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    eval_steps: int = 200
    eval_epochs: int = 1  # Evaluate every N epochs (1 = every epoch, 5 = every 5th)
    save_steps: int = 200
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "deep-past"
    wandb_run_name: str | None = None
    num_beams: int = 4
    eval_num_beams: int = 1  # Use greedy decoding during training for speed
    num_workers: int = 4
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 = no smoothing)
    no_repeat_ngram_size: int = 0  # Prevent repeated n-grams in generation (0 = disabled)
    repetition_penalty: float = 1.0  # Repetition penalty for generation (1.0 = no penalty)
    length_penalty: float = 1.0  # Length penalty for beam search
    lr_schedule: str = "linear"  # "linear" or "cosine"
    optimizer: str = "adamw"  # "adamw" or "adafactor" (T5 native, no momentum)
    log_perf_every: int = 0  # Log throughput/latency every N optimizer steps (0 = disabled)
    profile_steps: int = 0  # Stop after N optimizer steps for cheap speed benchmarks (0 = full run)
    ema_decay: float = 0.0  # EMA decay factor (0.0 = no EMA, 0.999 = typical)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def active_ablation_flags(data: DataConfig) -> list[str]:
    """Return sorted list of ablation flag names that are set to True."""
    flags = []
    if data.strip_determinatives:
        flags.append("strip_determinatives")
    if data.tag_sumerograms:
        flags.append("tag_sumerograms")
    if data.remove_gaps:
        flags.append("remove_gaps")
    return flags


_VALID_DATA_MODES = frozenset(
    {"document", "sentence_only", "extended_sentence", "mixed", "prepared"}
)
_VALID_VAL_SPLIT_STRATEGIES = frozenset({"random", "doc_id"})
_VALID_MODEL_TYPES = frozenset({"hf_seq2seq", "lstm_baseline"})
_VALID_OPTIMIZERS = frozenset({"adamw", "adafactor"})
_VALID_LR_SCHEDULES = frozenset({"linear", "cosine"})


def _resolve_config_path(config_path: Path) -> Path:
    """Resolve a config path, falling back to ``coursework/configs/`` by name.

    Configs may be referenced by full path (``coursework/configs/foo.yaml``) or
    by bare filename (``foo.yaml``); the latter is resolved against the
    coursework config directory.
    """
    if config_path.exists():
        return config_path

    config_root = Path("coursework") / "configs"
    candidate = config_root / config_path.name
    if candidate.exists():
        logger.info("Resolved config path %s -> %s", config_path, candidate)
        return candidate

    if config_root.exists():
        matches = sorted(config_root.rglob(config_path.name))
        if len(matches) == 1:
            logger.info("Resolved config path %s -> %s", config_path, matches[0])
            return matches[0]

    raise FileNotFoundError(f"Config file not found: {config_path}")


def validate_config(config: Config) -> None:
    """Validate config for common errors."""
    if config.train.fp16 and config.train.bf16:
        raise ValueError("Cannot enable both fp16 and bf16 simultaneously.")
    if config.data.max_source_length <= 0 or config.data.max_target_length <= 0:
        raise ValueError("max_source_length and max_target_length must be > 0.")
    if not 0 < config.data.val_split < 1:
        raise ValueError("data.val_split must be between 0 and 1 (exclusive).")
    if config.data.data_mode not in _VALID_DATA_MODES:
        raise ValueError(
            f"Unknown data_mode {config.data.data_mode!r}. "
            f"Valid options: {sorted(_VALID_DATA_MODES)}"
        )
    if config.data.val_split_strategy not in _VALID_VAL_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown val_split_strategy {config.data.val_split_strategy!r}. "
            f"Valid options: {sorted(_VALID_VAL_SPLIT_STRATEGIES)}"
        )
    if config.model.model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type {config.model.model_type!r}. "
            f"Valid options: {sorted(_VALID_MODEL_TYPES)}"
        )
    if config.train.optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {config.train.optimizer!r}. "
            f"Valid options: {sorted(_VALID_OPTIMIZERS)}"
        )
    if config.train.lr_schedule not in _VALID_LR_SCHEDULES:
        raise ValueError(
            f"Unknown lr_schedule {config.train.lr_schedule!r}. "
            f"Valid options: {sorted(_VALID_LR_SCHEDULES)}"
        )
    if config.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0.")
    if config.train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0.")
    if config.train.gradient_accumulation_steps <= 0:
        raise ValueError("train.gradient_accumulation_steps must be > 0.")
    if config.train.eval_steps <= 0:
        raise ValueError("train.eval_steps must be > 0.")
    if config.train.eval_epochs <= 0:
        raise ValueError("train.eval_epochs must be > 0.")
    if config.train.num_beams <= 0 or config.train.eval_num_beams <= 0:
        raise ValueError("train.num_beams and train.eval_num_beams must be > 0.")
    if config.train.log_perf_every < 0:
        raise ValueError("train.log_perf_every must be >= 0.")
    if config.train.profile_steps < 0:
        raise ValueError("train.profile_steps must be >= 0.")


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    config_path = _resolve_config_path(Path(config_path))

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config top-level must be a mapping: {config_path}")

    def _section(name: str) -> dict[str, Any]:
        payload = raw.get(name, {})
        if not isinstance(payload, dict):
            raise ValueError(f"Config section {name!r} must be a mapping in {config_path}")
        return payload

    config = Config()
    config.data = DataConfig(**_section("data"))
    config.model = ModelConfig(**_section("model"))
    config.train = TrainConfig(**_section("train"))

    validate_config(config)
    logger.info("Loaded config from %s", config_path)
    return config


def _parse_bool_override(raw: str, key: str) -> bool:
    value = raw.strip().lower()
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    if value in truthy:
        return True
    if value in falsy:
        return False
    raise ValueError(f"Invalid boolean override for {key}: {raw!r}")


def _field_allows_none(sub_config: Any, param: str) -> bool:
    hints = get_type_hints(type(sub_config))
    field_type = hints.get(param)
    if field_type is None:
        return False
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin in (Union, UnionType):
        return type(None) in args
    return False


def _parse_override_value(
    current_value: Any,
    raw_value: str,
    key: str,
    *,
    allows_none: bool,
) -> Any:
    """Coerce CLI override values based on the existing config field value."""
    if allows_none and raw_value.strip().lower() in {"none", "null"}:
        return None
    if isinstance(current_value, bool):
        return _parse_bool_override(raw_value, key)
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return raw_value
    if current_value is None:
        # Optional fields defaulting to None are currently string-like paths/names.
        return raw_value
    raise TypeError(
        f"Unsupported override type for {key}: {type(current_value).__name__}. "
        "Set this value in YAML instead."
    )


def apply_cli_overrides(config: Config, unknown: list[str]) -> None:
    """Apply CLI overrides like --train.batch_size 16 to config."""
    if len(unknown) % 2 != 0:
        logger.warning(
            "Odd number of CLI override tokens (%d); last token %r will be ignored",
            len(unknown),
            unknown[-1] if unknown else "",
        )
    # Pair up arguments: ["--train.batch_size", "16", "--model.dropout", "0.5"]
    pairs = zip(unknown[::2], unknown[1::2])

    for key, value in pairs:
        key = key.lstrip("-")
        if "." not in key:
            logger.warning("Ignoring unrecognized CLI argument: --%s %s", key, value)
            continue

        section, param = key.split(".", 1)
        sub_config = getattr(config, section, None)

        if sub_config is None or not hasattr(sub_config, param):
            logger.warning("Ignoring unknown config override: --%s.%s %s", section, param, value)
            continue

        current_value = getattr(sub_config, param)
        key_name = f"{section}.{param}"
        allows_none = _field_allows_none(sub_config, param)
        try:
            parsed_value = _parse_override_value(
                current_value,
                value,
                key_name,
                allows_none=allows_none,
            )
        except Exception as exc:
            logger.warning("Ignoring invalid override --%s %s (%s)", key_name, value, exc)
            continue

        setattr(sub_config, param, parsed_value)
        logger.info("CLI override: %s = %r", key_name, parsed_value)


# Keep old name as alias for backward compatibility
_apply_cli_overrides = apply_cli_overrides
