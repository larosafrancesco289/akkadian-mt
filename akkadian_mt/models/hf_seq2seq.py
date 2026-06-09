"""HuggingFace seq2seq model wrapper for mT5, NLLB, mBART, etc."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from akkadian_mt.config import SPECIAL_TOKENS, ModelConfig
from akkadian_mt.models.checkpoint_compat import model_uses_tied_embeddings
from akkadian_mt.utils import count_parameters

logger = logging.getLogger(__name__)


def load_hf_model(config: ModelConfig) -> tuple[Any, Any]:
    """Load a HuggingFace seq2seq model and tokenizer.

    Supports any model compatible with AutoModelForSeq2SeqLM:
    - google/mt5-small (primary baseline, 300M params)
    - facebook/nllb-200-distilled-600M
    - facebook/mbart-large-50

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name = config.model_name
    logger.info("Loading model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Track whether this checkpoint uses tied input/output embeddings.
    # Some T5-family checkpoints (notably mT5) ship with an *untied* lm_head; others expect tying.
    was_tied = model_uses_tied_embeddings(model)

    # Add special tokens used in preprocessing (<gap>, <big_gap>)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if num_added > 0:
        # IMPORTANT: For *untied* checkpoints (e.g., mT5), some Transformers versions can
        # inadvertently re-tie lm_head to shared embeddings during resize_token_embeddings(),
        # massively changing logits scale and breaking training. Preserve the checkpoint's tying.
        if not was_tied and getattr(model.config, "tie_word_embeddings", None) is True:
            model.config.tie_word_embeddings = False
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added %d special tokens, resized embeddings to %d", num_added, len(tokenizer))

    logger.info("Model loaded: %s (%.1fM params)", model_name, count_parameters(model) / 1e6)

    return model, tokenizer


def build_gen_kwargs(
    max_length: int = 256,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> dict[str, Any]:
    """Build a generation kwargs dict for model.generate().

    Centralises the conditional logic for beam search vs greedy settings.
    """
    gen_kwargs: dict[str, Any] = {
        "max_length": max_length,
        "num_beams": num_beams,
    }
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
        gen_kwargs["length_penalty"] = length_penalty
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    return gen_kwargs


def generate_translations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
    max_source_length: int | None = None,
    num_beams: int = 4,
    device: str = "cpu",
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    source_prefix: str = "",
    batch_size: int = 32,
) -> list[str]:
    """Generate translations for a list of source texts.

    Args:
        model: HuggingFace seq2seq model.
        tokenizer: Corresponding tokenizer.
        texts: List of source transliterations.
        max_length: Maximum generation (target) length.
        max_source_length: Tokenizer truncation for source texts. Defaults to max_length.
        num_beams: Beam search width.
        device: Device string.
        no_repeat_ngram_size: Prevent repeated n-grams of this size.
        repetition_penalty: Penalty for repeated tokens.
        length_penalty: Length penalty for beam search.
        source_prefix: Prefix to prepend to source texts.
        batch_size: Number of texts to process per batch.

    Returns:
        List of generated English translations.
    """
    import torch

    model.eval()
    translations: list[str] = []
    src_max_len = max_source_length if max_source_length is not None else max_length

    gen_kwargs = build_gen_kwargs(
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
    )

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [source_prefix + t for t in texts[i : i + batch_size]]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=src_max_len,
                truncation=True,
                padding=True,
            ).to(device)

            outputs = model.generate(**inputs, **gen_kwargs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)

    return translations
