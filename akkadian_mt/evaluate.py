"""Evaluate a trained model checkpoint on the validation set.

Usage:
    uv run python scripts/evaluate.py --checkpoint outputs/best_model.pt
    uv run python scripts/evaluate.py --checkpoint outputs/best_model.pt --num-beams 4 --no-repeat 0 --length-penalty 1.2
    uv run python scripts/evaluate.py --checkpoint outputs/best_model.pt --num-beams 4 --mbr --postprocess
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch

from akkadian_mt.data.dataset import create_dataloaders, load_train_data
from akkadian_mt.data.preprocessing import postprocess_translation
from akkadian_mt.evaluation.metrics import compute_combined_score, mbr_pick
from akkadian_mt.utils import get_device, load_checkpoint, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-beams", type=int, default=None, help="Override beam width (default: from config)")
    parser.add_argument("--no-repeat", type=int, default=None, help="Override no_repeat_ngram_size (default: from config). Use 0 to disable.")
    parser.add_argument("--length-penalty", type=float, default=None, help="Override length penalty (default: from config)")
    parser.add_argument("--label", type=str, default=None, help="Optional label for this eval run (printed in output)")
    parser.add_argument("--postprocess", action="store_true", help="Apply output postprocessing (remove repeated words/n-grams, fix punctuation spacing)")
    parser.add_argument("--mbr", action="store_true", help="Use MBR decoding: generate num_beams candidates per input and pick by max avg chrF++")
    args = parser.parse_args()

    device = get_device()
    logger.info("Using device: %s", device)

    config, model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    set_seed(config.train.seed)

    logger.info(
        "Loaded checkpoint from %s (best_score_at_train=%.2f, step=%d)",
        args.checkpoint,
        checkpoint["best_score"],
        checkpoint["step"],
    )

    # Load validation data
    _, val_df = load_train_data(config.data)
    _, val_loader = create_dataloaders(
        val_df,
        val_df,
        tokenizer,
        config.data,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    # Build generation kwargs, with CLI overrides taking precedence
    num_beams = args.num_beams if args.num_beams is not None else config.train.num_beams
    no_repeat = args.no_repeat if args.no_repeat is not None else getattr(config.train, "no_repeat_ngram_size", 0)
    rep_penalty = getattr(config.train, "repetition_penalty", 1.0)
    len_penalty = args.length_penalty if args.length_penalty is not None else getattr(config.train, "length_penalty", 1.0)

    gen_kwargs: dict[str, Any] = {
        "max_length": config.data.max_target_length,
        "num_beams": num_beams,
        "early_stopping": True,
        "length_penalty": len_penalty,
    }
    if no_repeat > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat
    if rep_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = rep_penalty
    if args.mbr:
        # Return all beam hypotheses for MBR selection
        gen_kwargs["num_return_sequences"] = num_beams

    logger.info(
        "Generation settings: num_beams=%d, no_repeat_ngram_size=%d, length_penalty=%.2f, mbr=%s",
        num_beams, no_repeat, len_penalty, args.mbr,
    )
    if args.postprocess:
        logger.info("Output postprocessing: ENABLED")
    if args.label:
        logger.info("Run label: %s", args.label)

    # Generate predictions
    all_preds: list[str] = []
    all_refs: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            if args.mbr:
                # generated shape: [batch_size * num_beams, seq_len]
                # Reshape to [batch_size, num_beams] list-of-lists of decoded strings
                raw = tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_size_actual = input_ids.shape[0]
                n = num_beams
                # Group candidates: raw[0..n-1] are for input[0], raw[n..2n-1] for input[1], etc.
                preds = [
                    mbr_pick(raw[i * n: (i + 1) * n])
                    for i in range(batch_size_actual)
                ]
            else:
                preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if args.postprocess:
                preds = [postprocess_translation(p) for p in preds]
            labels = labels.clone()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    metrics = compute_combined_score(all_preds, all_refs)
    logger.info("=== RESULTS ===")
    logger.info("BLEU:     %.2f", metrics["bleu"])
    logger.info("chrF++:   %.2f", metrics["chrf"])
    logger.info("Combined: %.2f", metrics["combined"])


if __name__ == "__main__":
    main()
