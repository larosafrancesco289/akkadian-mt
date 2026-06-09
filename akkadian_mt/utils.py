"""Utility functions for reproducibility, device handling, and logging."""

from __future__ import annotations

import dataclasses
import logging
import random
import sys
from typing import Any

import numpy as np
import torch

from akkadian_mt.models.checkpoint_compat import prepare_model_for_checkpoint_load

logger = logging.getLogger(__name__)


def configure_torch_runtime(*, fast: bool = True) -> None:
    """Apply the repo's preferred PyTorch runtime flags."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        return
    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def set_seed(seed: int = 42, *, fast: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        fast: If True, enable TF32 and cudnn.benchmark for faster training
              at the cost of bitwise non-determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configure_torch_runtime(fast=fast)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, Any, dict]:
    """Load a trained model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (config, model, tokenizer, checkpoint_dict).
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from akkadian_mt.config import Config, DataConfig, ModelConfig, TrainConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    # Filter config dicts to only known fields (handles checkpoints from older code)
    def _known_fields(cls: type) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)}

    config = Config(
        data=DataConfig(
            **{k: v for k, v in config_dict["data"].items() if k in _known_fields(DataConfig)}
        ),
        model=ModelConfig(
            **{k: v for k, v in config_dict["model"].items() if k in _known_fields(ModelConfig)}
        ),
        train=TrainConfig(
            **{k: v for k, v in config_dict["train"].items() if k in _known_fields(TrainConfig)}
        ),
    )

    from akkadian_mt.config import SPECIAL_TOKENS

    model_name = config.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Re-add special tokens that were added during training (<gap>, <big_gap>, <SUM>, </SUM>)
    # This ensures the tokenizer matches the one used at training time.
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    state_dict = checkpoint["model_state_dict"]
    model = prepare_model_for_checkpoint_load(model, state_dict, device=device, log=logger)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return config, model, tokenizer, checkpoint
