"""Sentence-level evaluation on the held-out test set.

Loads an exported model directory (safetensors + config) or a training
checkpoint, generates translations with beam search, applies postprocessing,
and reports BLEU / chrF++ / combined score. This is the exact path behind every
number reported in the paper.

Usage:
    uv run akkadian-mt eval lb --model exported/byt5-base/
    uv run akkadian-mt eval lb --checkpoint outputs/coursework/byt5_base_baseline/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ByT5Tokenizer

from akkadian_mt.data.preprocessing import postprocess_translation
from akkadian_mt.evaluation.metrics import compute_combined_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Side-car files that indicate a complete, loadable tokenizer in an export dir.
_TOKENIZER_SIDE_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SentenceDataset(Dataset):
    """Sentence pairs for evaluation, with an optional source prefix."""

    def __init__(self, texts: list[str], refs: list[str], *, prefix: str = "") -> None:
        self.texts = [prefix + text for text in texts]
        self.refs = refs

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.texts[idx], self.refs[idx]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_exported_model(
    model_dir: str | Path,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load an exported model (safetensors + config.json + metadata).

    Returns (model, tokenizer, metadata).
    """
    model_dir = Path(model_dir)

    # config.json may be nested if the export was uploaded as a zip.
    if not (model_dir / "config.json").exists():
        configs = list(model_dir.rglob("config.json"))
        if configs:
            model_dir = configs[0].parent
        else:
            raise FileNotFoundError(f"config.json not found in {model_dir}")

    meta: dict[str, Any] = {}
    meta_path = model_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    tokenizer = _load_exported_tokenizer(model_dir)
    model = model.to(device).eval()

    return model, tokenizer, meta


def _load_exported_tokenizer(model_dir: Path) -> Any:
    """Load a tokenizer from an export, falling back to ByT5 for legacy exports."""
    try:
        return AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as exc:
        if any((model_dir / name).exists() for name in _TOKENIZER_SIDE_FILES):
            raise

        config_path = model_dir / "config.json"
        with open(config_path) as f:
            config_payload = json.load(f)

        if config_payload.get("model_type") != "byt5":
            raise

        logger.warning(
            "Tokenizer assets missing in %s; falling back to ByT5Tokenizer (%s)",
            model_dir,
            exc,
        )
        tokenizer = ByT5Tokenizer()
        added_path = model_dir / "added_tokens.json"
        if added_path.exists():
            with open(added_path) as f:
                added = json.load(f)
            if added:
                tokenizer.add_special_tokens({"additional_special_tokens": list(added.keys())})
        return tokenizer


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a model from a training checkpoint.

    Returns (model, tokenizer, metadata).
    """
    from akkadian_mt.utils import load_checkpoint

    config, model, tokenizer, checkpoint = load_checkpoint(str(checkpoint_path), device)
    meta = {
        "source_prefix": config.data.source_prefix,
        "remove_gaps": config.data.remove_gaps,
        "best_score": checkpoint.get("best_score", 0),
    }
    return model, tokenizer, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: SentenceDataset,
    device: torch.device,
    *,
    num_beams: int = 8,
    length_penalty: float = 1.3,
    no_repeat_ngram: int = 0,
    max_length: int = 512,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    use_postprocessing: bool = True,
) -> dict[str, Any]:
    """Generate with beam search and return metrics plus predictions."""

    def collate_fn(batch: list[tuple[str, str]]) -> tuple[list[str], Any]:
        texts = [text for text, _ in batch]
        refs = [ref for _, ref in batch]
        enc = tokenizer(
            texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )
        return refs, enc

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": True,
        "length_penalty": length_penalty,
    }
    if no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram

    all_preds: list[str] = []
    all_refs: list[str] = []

    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx:
        for refs, enc in tqdm(loader, desc="Evaluating", leave=False):
            generated = model.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                **gen_kwargs,
            )
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if use_postprocessing:
                preds = [postprocess_translation(p) for p in preds]
            all_preds.extend(preds)
            all_refs.extend(refs)

    metrics = compute_combined_score(all_preds, all_refs)
    return {**metrics, "predictions": all_preds, "references": all_refs}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentence-level held-out evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Path to an exported model directory")
    group.add_argument("--checkpoint", type=str, help="Path to a training checkpoint (.pt)")

    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/independent_test_set.csv",
        help="Sentence-level test CSV with transliteration/translation columns",
    )
    parser.add_argument("--num-beams", type=int, default=8, help="Beam width")
    parser.add_argument("--length-penalty", type=float, default=1.3, help="Length penalty")
    parser.add_argument("--no-repeat", type=int, default=0, help="No-repeat n-gram size (0 = off)")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable postprocessing")
    parser.add_argument("--save-preds", type=str, default=None, help="Write predictions to CSV")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    test_df = pd.read_csv(args.test_file)
    logger.info("Loaded %d test pairs from %s", len(test_df), args.test_file)

    model_path = Path(args.checkpoint or args.model)
    name = model_path.stem if model_path.suffix == ".pt" else model_path.name
    if args.checkpoint:
        model, tokenizer, meta = load_checkpoint_model(model_path, device)
    else:
        model, tokenizer, meta = load_exported_model(model_path, device)

    prefix = meta.get("source_prefix", "")
    remove_gaps = meta.get("remove_gaps", False)
    logger.info("Evaluating %s (prefix=%r, remove_gaps=%s)", name, prefix, remove_gaps)

    texts = test_df["transliteration"].tolist()
    refs = test_df["translation"].tolist()
    if remove_gaps:
        texts = [re.sub(r"\s+", " ", t.replace("<big_gap>", "").replace("<gap>", "")).strip() for t in texts]

    dataset = SentenceDataset(texts, refs, prefix=prefix)
    result = evaluate_model(
        model,
        tokenizer,
        dataset,
        device,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram=args.no_repeat,
        batch_size=args.batch_size,
        use_postprocessing=not args.no_postprocess,
    )
    logger.info(
        "%s -> BLEU=%.2f  chrF++=%.2f  Combined=%.2f",
        name,
        result["bleu"],
        result["chrf"],
        result["combined"],
    )

    if args.save_preds:
        pd.DataFrame(
            {
                "transliteration": texts,
                "reference": result["references"],
                "prediction": result["predictions"],
            }
        ).to_csv(args.save_preds, index=False)
        logger.info("Predictions saved to %s", args.save_preds)


if __name__ == "__main__":
    main()
