"""Compatibility helpers for loading seq2seq checkpoints across model families."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)

_INPUT_EMBEDDING_KEY_SUFFIXES = (
    "shared.weight",
    "model.shared.weight",
    "encoder.embed_tokens.weight",
    "model.encoder.embed_tokens.weight",
    "decoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
)
_OUTPUT_EMBEDDING_KEY_SUFFIXES = (
    "lm_head.weight",
    "output_projection.weight",
)


def find_state_dict_tensor_key(
    state_dict: Mapping[str, Any],
    suffixes: tuple[str, ...],
) -> str | None:
    """Find a tensor key in a state dict, preferring exact suffix matches."""
    for suffix in suffixes:
        tensor = state_dict.get(suffix)
        if isinstance(tensor, torch.Tensor):
            return suffix

    for suffix in suffixes:
        for key, value in state_dict.items():
            if key.endswith(suffix) and isinstance(value, torch.Tensor):
                return key

    return None


def infer_vocab_size_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    """Infer checkpoint vocabulary size from embedding-like tensors when available."""
    for suffixes in (_INPUT_EMBEDDING_KEY_SUFFIXES, _OUTPUT_EMBEDDING_KEY_SUFFIXES):
        key = find_state_dict_tensor_key(state_dict, suffixes)
        if key is None:
            continue
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
            return int(tensor.shape[0])
    return None


def model_uses_tied_embeddings(model: Any) -> bool:
    """Return True when input and output embeddings share storage."""
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return False

    try:
        input_embeddings = get_input()
        output_embeddings = get_output()
    except Exception:
        return False

    if input_embeddings is None or output_embeddings is None:
        return False

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if not isinstance(input_weight, torch.Tensor) or not isinstance(output_weight, torch.Tensor):
        return False
    if input_weight.shape != output_weight.shape:
        return False
    return input_weight.data_ptr() == output_weight.data_ptr()


def prepare_model_for_checkpoint_load(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    device: torch.device | None = None,
    log: logging.Logger | None = None,
) -> Any:
    """Adapt a model instance so a checkpoint state dict can be loaded safely.

    This handles two cases:
    - checkpoints whose vocab size differs from the base pretrained model
    - checkpoints with untied output embeddings, even if the base model ships tied
    """
    active_logger = log or logger

    input_key = find_state_dict_tensor_key(state_dict, _INPUT_EMBEDDING_KEY_SUFFIXES)
    output_key = find_state_dict_tensor_key(state_dict, _OUTPUT_EMBEDDING_KEY_SUFFIXES)

    if input_key and output_key:
        input_weight = state_dict[input_key]
        output_weight = state_dict[output_key]
        if (
            isinstance(input_weight, torch.Tensor)
            and isinstance(output_weight, torch.Tensor)
            and input_weight.shape == output_weight.shape
            and not torch.equal(input_weight, output_weight)
        ):
            # The checkpoint has diverged embeddings — the model MUST be loaded
            # with tie_word_embeddings=False regardless of the current model
            # state.  Newer Transformers may auto-untie during from_pretrained
            # (so model_uses_tied_embeddings can return False), but the config
            # flag may still say True, causing load_state_dict to re-tie.
            needs_rebuild = (
                model_uses_tied_embeddings(model)
                or getattr(getattr(model, "config", None), "tie_word_embeddings", None) is True
            )
            if needs_rebuild:
                active_logger.warning(
                    "Checkpoint stores untied embeddings (%s vs %s); rebuilding model untied",
                    input_key,
                    output_key,
                )
                model.config.tie_word_embeddings = False
                # Pre-set vocab_size from checkpoint so the rebuilt model has
                # the correct embedding dimensions from the start.
                ckpt_vocab = infer_vocab_size_from_state_dict(state_dict)
                if (
                    ckpt_vocab is not None
                    and getattr(model.config, "vocab_size", None) != ckpt_vocab
                ):
                    active_logger.info(
                        "Setting config vocab_size to %d (was %d) before untied rebuild",
                        ckpt_vocab,
                        getattr(model.config, "vocab_size", -1),
                    )
                    model.config.vocab_size = ckpt_vocab
                model = type(model)(model.config)
                if device is not None:
                    model = model.to(device)

    checkpoint_vocab_size = infer_vocab_size_from_state_dict(state_dict)
    if checkpoint_vocab_size is None:
        return model

    get_input = getattr(model, "get_input_embeddings", None)
    if not callable(get_input):
        return model

    try:
        input_embeddings = get_input()
    except Exception:
        return model

    current_vocab_size = getattr(input_embeddings, "num_embeddings", None)
    if current_vocab_size is None and isinstance(
        getattr(input_embeddings, "weight", None), torch.Tensor
    ):
        current_vocab_size = int(input_embeddings.weight.shape[0])

    if current_vocab_size is not None and checkpoint_vocab_size != current_vocab_size:
        active_logger.info(
            "Resizing token embeddings from %d to %d to match checkpoint",
            current_vocab_size,
            checkpoint_vocab_size,
        )
        model.resize_token_embeddings(checkpoint_vocab_size)

    return model
