"""Evaluate models on sentence-level independent test to correlate with Kaggle LB.

Loads exported models (safetensors) or training checkpoints and evaluates on
independent_test_set.csv at sentence level — mimicking the Kaggle hidden test
which is also sentence-level.

Usage:
    # Single exported model
    uv run python scripts/evaluate_lb.py --model exported/byt5-oracc-finetune/

    # Single checkpoint
    uv run python scripts/evaluate_lb.py --checkpoint outputs/oracc_finetune/best_model.pt

    # All exported models (batch mode)
    uv run python scripts/evaluate_lb.py --all-exported

    # Custom generation params
    uv run python scripts/evaluate_lb.py --model exported/byt5-oracc-finetune/ \
        --num-beams 8 --length-penalty 1.3 --postprocess

    # Sweep generation params on one model
    uv run python scripts/evaluate_lb.py --model exported/byt5-oracc-finetune/ --sweep
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ByT5Tokenizer

from akkadian_mt.data.preprocessing import (
    normalize_transliteration as _shared_normalize_transliteration,
)
from akkadian_mt.data.preprocessing import (
    postprocess_translation as _shared_postprocess_translation,
)
from akkadian_mt.evaluation.metrics import (
    compute_combined_score as _shared_compute_combined_score,
)
from akkadian_mt.evaluation.metrics import (
    mbr_pick as _shared_mbr_pick,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

_DEFAULT_INFERENCE_SETTINGS = {
    "use_fuzzy_retrieval": True,
    "use_reranking": True,
    "rerank_num_return": 4,
    "retrieval_top_k": 3,
    "retrieval_shortlist": 64,
    "retrieval_min_score": 0.38,
    "retrieval_direct_threshold": 0.92,
    "retrieval_direct_margin": 0.05,
}


# ---------------------------------------------------------------------------
# Postprocessing (mirrors notebooks/kaggle_submission.py)
# ---------------------------------------------------------------------------

_repeated_words = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_repeated_bigram = re.compile(r"\b((?:\w+\s+)\w+)(?:\s+\1\b)+")
_repeated_trigram = re.compile(r"\b((?:\w+\s+){2}\w+)(?:\s+\1\b)+")
_punct_space = re.compile(r"\s+([.,;:!?])")
_repeated_punct = re.compile(r"([.,;:])\\1+")
_name_token_re = re.compile(r"\b([A-Za-z][A-Za-z'’-]*)\b")
_doc_title_patterns = (
    "the king",
    "my lord",
    "the lord",
    "the queen",
    "the mayor",
    "the steward",
    "the merchant",
    "the scribe",
    "the servant",
)


def _remove_phrase_repeats(text: str, max_phrase_len: int = 12) -> str:
    """Remove repeated phrases of any length up to max_phrase_len words."""
    words = text.split()
    if len(words) < 4:
        return text
    for plen in range(min(max_phrase_len, len(words) // 2), 1, -1):
        i = 0
        result: list[str] = []
        while i < len(words):
            phrase = words[i : i + plen]
            if len(phrase) < plen:
                result.extend(words[i:])
                break
            j = i + plen
            while j + plen <= len(words) and words[j : j + plen] == phrase:
                j += plen
            result.extend(phrase)
            i = j if j > i + plen else i + plen
        words = result
    return " ".join(words)


def postprocess_translation(text: str) -> str:
    """Clean up model output: remove repeated words/n-grams, fix punctuation."""
    if not text or not text.strip():
        return text
    text = _remove_phrase_repeats(text)
    text = _repeated_words.sub(r"\1", text)
    text = _repeated_bigram.sub(r"\1", text)
    text = _repeated_trigram.sub(r"\1", text)
    text = _punct_space.sub(r"\1", text)
    text = _repeated_punct.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Metrics (same as src/evaluation/metrics.py)
# ---------------------------------------------------------------------------


def compute_combined_score(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute √(BLEU × chrF++) — the Kaggle competition metric."""
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2).score
    combined = math.sqrt(max(bleu, 0) * max(chrf, 0))
    return {"bleu": bleu, "chrf": chrf, "combined": combined}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SentenceDataset(Dataset):
    """Simple dataset for sentence-level evaluation."""

    def __init__(
        self,
        texts: list[str],
        refs: list[str],
        *,
        doc_ids: list[str] | None = None,
        prefix: str = "",
    ) -> None:
        self.raw_texts = texts
        self.texts = [prefix + t for t in texts]
        self.refs = refs
        self.doc_ids = doc_ids or [str(i) for i in range(len(texts))]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, str, str, str]:
        return self.texts[idx], self.refs[idx], self.raw_texts[idx], self.doc_ids[idx]


_RETRIEVAL_SKIP_TOKENS = frozenset({"<gap>", "<big_gap>"})


def _source_tokens(text: str) -> list[str]:
    return [tok for tok in str(text).split() if tok and tok not in _RETRIEVAL_SKIP_TOKENS]


def _word_set_score(a: str, b: str) -> float:
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words | b_words), 1)


def _retrieval_source_priority(source: str, tier: str) -> int:
    source = str(source or "").lower()
    tier = str(tier or "").lower()
    if source in {"train.csv", "train_sentence", "golden_corpus_v2_prefilt"}:
        return 3
    if tier == "gold":
        return 3
    if "oa" in source or "external_alignment_phucthaiv_translation_oa" in source:
        return 2
    if "external_alignment" in source:
        return 1
    return 0


def build_retrieval_bank(
    retrieval_file: str | Path | None,
    *,
    remove_gaps: bool,
    max_postings: int = 600,
) -> tuple[dict[str, str], dict[str, Any]]:
    if retrieval_file is None:
        retrieval_df = pd.read_csv("data/raw/train.csv", low_memory=False)
        retrieval_df["source"] = "train.csv"
        retrieval_df["tier"] = "gold"
        retrieval_df["confidence"] = 1.0
    else:
        retrieval_df = pd.read_csv(retrieval_file, low_memory=False)
        if "source" not in retrieval_df.columns:
            retrieval_df["source"] = Path(retrieval_file).name
        if "tier" not in retrieval_df.columns:
            retrieval_df["tier"] = "external_alignment"
        if "confidence" not in retrieval_df.columns:
            retrieval_df["confidence"] = 1.0

    retrieval_df["transliteration"] = retrieval_df["transliteration"].fillna("").astype(str)
    retrieval_df["translation"] = retrieval_df["translation"].fillna("").astype(str)
    retrieval_df = retrieval_df[
        retrieval_df["transliteration"].str.strip().ne("")
        & retrieval_df["translation"].str.strip().ne("")
    ].copy()
    retrieval_df["norm_source"] = retrieval_df["transliteration"].map(
        lambda text: _shared_normalize_transliteration(
            text,
            do_remove_gaps=remove_gaps,
        )
    )
    retrieval_df["norm_translation"] = retrieval_df["translation"].map(postprocess_translation)
    retrieval_df = retrieval_df[
        retrieval_df["norm_source"].str.strip().ne("")
        & retrieval_df["norm_translation"].str.strip().ne("")
    ].copy()
    retrieval_df["source_priority"] = [
        _retrieval_source_priority(source, tier)
        for source, tier in zip(retrieval_df["source"], retrieval_df["tier"])
    ]
    retrieval_df["confidence"] = pd.to_numeric(retrieval_df["confidence"], errors="coerce").fillna(1.0)
    retrieval_df["translation_len"] = retrieval_df["norm_translation"].str.len()
    retrieval_df = retrieval_df.sort_values(
        ["source_priority", "confidence", "translation_len"],
        ascending=[False, False, True],
    )
    retrieval_df = retrieval_df.drop_duplicates(subset=["norm_source"], keep="first").reset_index(drop=True)

    exact_memory = dict(zip(retrieval_df["norm_source"], retrieval_df["norm_translation"]))
    doc_freq: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for row in retrieval_df.itertuples(index=False):
        tokens = _source_tokens(row.norm_source)
        token_set = set(tokens)
        for token in token_set:
            doc_freq[token] += 1
        records.append(
            {
                "source": row.norm_source,
                "translation": row.norm_translation,
                "tokens": tokens,
                "token_set": token_set,
                "confidence": float(row.confidence),
                "source_priority": int(row.source_priority),
            }
        )

    total_docs = max(len(records), 1)
    idf = {
        token: math.log((total_docs + 1) / (count + 1)) + 1.0
        for token, count in doc_freq.items()
    }
    postings: dict[str, list[int]] = {}
    first_token_index: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        if record["tokens"]:
            first_token_index.setdefault(record["tokens"][0], []).append(idx)
        for token in record["token_set"]:
            if doc_freq[token] <= max_postings:
                postings.setdefault(token, []).append(idx)

    index = {
        "records": records,
        "idf": idf,
        "doc_freq": doc_freq,
        "postings": postings,
        "first_token_index": first_token_index,
    }
    return exact_memory, index


def _weighted_overlap(query_set: set[str], cand_set: set[str], idf: dict[str, float]) -> float:
    if not query_set or not cand_set:
        return 0.0
    overlap = sum(idf.get(tok, 1.0) for tok in (query_set & cand_set))
    total = sum(idf.get(tok, 1.0) for tok in (query_set | cand_set))
    return overlap / max(total, 1e-9)


def fuzzy_retrieve(
    query_text: str,
    index: dict[str, Any],
    *,
    top_k: int,
    shortlist: int,
    min_score: float,
) -> list[dict[str, Any]]:
    query_tokens = _source_tokens(query_text)
    if not query_tokens:
        return []

    query_set = set(query_tokens)
    doc_freq = index["doc_freq"]
    postings = index["postings"]
    idf = index["idf"]
    scores: Counter[int] = Counter()

    selected_tokens = sorted(
        query_set,
        key=lambda tok: (doc_freq.get(tok, 10**9), -idf.get(tok, 1.0), tok),
    )[:8]
    for token in selected_tokens:
        for idx in postings.get(token, []):
            scores[idx] += idf.get(token, 1.0)

    if not scores and query_tokens:
        for idx in index["first_token_index"].get(query_tokens[0], [])[:shortlist]:
            scores[idx] += 0.5

    ranked: list[dict[str, Any]] = []
    for idx, _ in scores.most_common(shortlist):
        record = index["records"][idx]
        char_ratio = difflib.SequenceMatcher(None, query_text, record["source"]).ratio()
        overlap = _weighted_overlap(query_set, record["token_set"], idf)
        length_sim = min(len(query_tokens), len(record["tokens"])) / max(
            len(query_tokens), len(record["tokens"]), 1
        )
        prefix = 0.0
        if record["tokens"] and query_tokens and record["tokens"][0] == query_tokens[0]:
            prefix += 0.5
        if len(record["tokens"]) >= 2 and len(query_tokens) >= 2 and record["tokens"][:2] == query_tokens[:2]:
            prefix += 0.5
        score = 0.55 * overlap + 0.25 * char_ratio + 0.10 * length_sim + 0.10 * prefix
        score += 0.02 * min(record["confidence"], 1.0)
        score += 0.02 * max(record["source_priority"], 0)
        if score >= min_score:
            ranked.append(
                {
                    "source": record["source"],
                    "translation": record["translation"],
                    "score": float(score),
                }
            )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]


def should_use_direct_retrieval(
    hits: list[dict[str, Any]],
    *,
    threshold: float,
    margin: float,
) -> bool:
    if not hits:
        return False
    best = hits[0]
    gap = best["score"] - hits[1]["score"] if len(hits) > 1 else best["score"]
    return best["score"] >= threshold and gap >= margin


def _fold_name(text: str) -> str:
    return re.sub(r"[^a-z]+", "", str(text or "").lower())


def _extract_doc_signals(text: str) -> dict[str, set[str]]:
    names = set()
    for token in _name_token_re.findall(text):
        if not token or not token[0].isupper():
            continue
        folded = _fold_name(token)
        if len(folded) >= 4:
            names.add(folded)
    lower = f" {text.lower()} "
    titles = {pattern for pattern in _doc_title_patterns if f" {pattern} " in lower}
    return {"names": names, "titles": titles}


def update_doc_history(doc_history: dict[str, Counter], text: str) -> None:
    signals = _extract_doc_signals(text)
    for folded in signals["names"]:
        doc_history["names"][folded] += 1
    for title in signals["titles"]:
        doc_history["titles"][title] += 1


def _doc_history_score(text: str, doc_history: dict[str, Counter]) -> float:
    if not doc_history["names"] and not doc_history["titles"]:
        return 0.0
    signals = _extract_doc_signals(text)
    score_parts = []
    if signals["names"]:
        name_match = sum(1 for folded in signals["names"] if folded in doc_history["names"])
        score_parts.append(name_match / max(len(signals["names"]), 1))
    if signals["titles"]:
        title_match = sum(1 for title in signals["titles"] if title in doc_history["titles"])
        score_parts.append(title_match / max(len(signals["titles"]), 1))
    if not score_parts:
        return 0.0
    return sum(score_parts) / len(score_parts)


def rerank_candidates(
    candidates: list[dict[str, Any]],
    retrieval_hits: list[dict[str, Any]],
    doc_history: dict[str, Counter],
) -> str:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in candidates:
        text = cand.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(cand)
    if not deduped:
        return ""
    if len(deduped) == 1:
        return deduped[0]["text"]

    best_text = deduped[0]["text"]
    best_score = -1e9
    for cand in deduped:
        text = cand["text"]
        support = 0.0
        for hit in retrieval_hits:
            chrf = sacrebleu.sentence_chrf(text, [hit["translation"]], word_order=2).score / 100.0
            word_score = _word_set_score(text, hit["translation"])
            support = max(support, hit["score"] * (0.75 * chrf + 0.25 * word_score))
        if cand.get("origin") == "beam":
            prior = max(0.0, 1.0 - 0.12 * cand.get("rank", 0))
        else:
            prior = 0.42 + 0.08 * cand.get("retrieval_score", 0.0)
        doc_score = _doc_history_score(text, doc_history)
        style = 1.0
        words = text.split()
        if len(words) > 40:
            style -= min(0.30, (len(words) - 40) * 0.01)
        total = (
            0.50 * prior
            + 0.30 * support
            + 0.15 * doc_score
            + 0.05 * max(0.0, min(1.0, style))
        )
        if cand.get("origin") == "retrieval":
            total += 0.03
        if total > best_score:
            best_score = total
            best_text = text
    return best_text


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

    # Find config.json (may be nested if uploaded as zip)
    if not (model_dir / "config.json").exists():
        configs = list(model_dir.rglob("config.json"))
        if configs:
            model_dir = configs[0].parent
        else:
            raise FileNotFoundError(f"config.json not found in {model_dir}")

    # Load metadata
    meta: dict[str, Any] = {}
    meta_path = model_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    overrides_path = model_dir / "inference_overrides.json"
    if overrides_path.exists():
        with open(overrides_path) as f:
            overrides = json.load(f)
        if isinstance(overrides, dict):
            meta["inference_overrides"] = {
                **meta.get("inference_overrides", {}),
                **overrides,
            }

    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    tokenizer = _load_exported_tokenizer(model_dir)
    model = model.to(device).eval()

    return model, tokenizer, meta


def _load_exported_tokenizer(model_dir: Path) -> Any:
    """Load tokenizer from a full HF export, with a fallback for legacy ByT5 exports."""
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
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": list(added.keys())}
                )
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
        "data_dir": config.data.data_dir,
        "source_prefix": config.data.source_prefix,
        "remove_gaps": config.data.remove_gaps,
        "strip_determinatives": config.data.strip_determinatives,
        "tag_sumerograms": config.data.tag_sumerograms,
        "strip_homophone_subscripts": getattr(config.data, "strip_homophone_subscripts", True),
        "skip_source_normalization": getattr(config.data, "skip_source_normalization", False),
        "use_dictionary_gloss": getattr(config.data, "use_dictionary_gloss", False),
        "max_glosses": getattr(config.data, "max_glosses", 8),
        "best_score": checkpoint.get("best_score", 0),
    }
    return model, tokenizer, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def mbr_pick(candidates: list[str]) -> str:
    """Pick the candidate with highest average chrF++ against all others (MBR)."""
    unique: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    n = len(unique)
    best_i, best_score = 0, -1e9
    for i, hyp in enumerate(unique):
        score = sum(
            sacrebleu.sentence_chrf(hyp, [unique[j]], word_order=2).score
            for j in range(n) if j != i
        ) / max(1, n - 1)
        if score > best_score:
            best_score, best_i = score, i
    return unique[best_i]


# Canonical shared implementations override local duplicates for consistency.
postprocess_translation = _shared_postprocess_translation  # noqa: F811
compute_combined_score = _shared_compute_combined_score  # noqa: F811
mbr_pick = _shared_mbr_pick  # noqa: F811


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: SentenceDataset,
    device: torch.device,
    num_beams: int = 8,
    length_penalty: float = 1.5,
    no_repeat_ngram: int = 0,
    max_length: int = 512,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    use_postprocessing: bool = True,
    use_mbr: bool = False,
    mbr_num_return: int = 4,
    mbr_sample_cands: int = 2,
    exact_memory: dict[str, str] | None = None,
    retrieval_index: dict[str, Any] | None = None,
    use_fuzzy_retrieval: bool = False,
    use_rerank: bool = False,
    rerank_num_return: int = 4,
    retrieval_top_k: int = 3,
    retrieval_shortlist: int = 64,
    retrieval_min_score: float = 0.38,
    retrieval_direct_threshold: float = 0.92,
    retrieval_direct_margin: float = 0.05,
) -> dict[str, Any]:
    """Run evaluation and return metrics + predictions."""

    exact_memory = exact_memory or {}

    def collate_fn(batch: list[tuple[str, str, str, str]]) -> tuple[list[str], list[str], list[str], Any]:
        texts = [b[0] for b in batch]
        refs = [b[1] for b in batch]
        raw_texts = [b[2] for b in batch]
        doc_ids = [b[3] for b in batch]
        enc = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        return refs, raw_texts, doc_ids, enc

    # For MBR, process one at a time to gather multiple candidates per input
    effective_batch = 1 if use_mbr else batch_size
    loader = DataLoader(dataset, batch_size=effective_batch, collate_fn=collate_fn, num_workers=0)

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
    doc_histories: dict[str, dict[str, Counter]] = {}

    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.no_grad():
        with amp_ctx:
            for refs, raw_texts, doc_ids, enc in tqdm(loader, desc="Evaluating", leave=False):
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                if use_mbr:
                    # Generate beam candidates
                    mbr_gen = {**gen_kwargs, "num_return_sequences": mbr_num_return}
                    beam_out = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, **mbr_gen,
                    )
                    candidates = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

                    # Add sampling candidates for diversity
                    if mbr_sample_cands > 0:
                        sample_out = model.generate(
                            input_ids=input_ids, attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens, do_sample=True,
                            top_p=0.9, temperature=0.7,
                            num_return_sequences=mbr_sample_cands,
                        )
                        candidates.extend(tokenizer.batch_decode(sample_out, skip_special_tokens=True))

                    if use_postprocessing:
                        candidates = [postprocess_translation(c) for c in candidates]
                    preds = [mbr_pick(candidates)]
                else:
                    preds = []
                    for row_idx, raw_text in enumerate(raw_texts):
                        doc_history = doc_histories.setdefault(
                            str(doc_ids[row_idx]),
                            {"names": Counter(), "titles": Counter()},
                        )
                        if raw_text in exact_memory:
                            pred = exact_memory[raw_text]
                            update_doc_history(doc_history, pred)
                            preds.append(pred)
                            continue

                        retrieval_hits = (
                            fuzzy_retrieve(
                                raw_text,
                                retrieval_index,
                                top_k=retrieval_top_k,
                                shortlist=retrieval_shortlist,
                                min_score=retrieval_min_score,
                            )
                            if use_fuzzy_retrieval and retrieval_index is not None
                            else []
                        )
                        if should_use_direct_retrieval(
                            retrieval_hits,
                            threshold=retrieval_direct_threshold,
                            margin=retrieval_direct_margin,
                        ):
                            pred = retrieval_hits[0]["translation"]
                            update_doc_history(doc_history, pred)
                            preds.append(pred)
                            continue

                        row_gen = {
                            **gen_kwargs,
                            "num_return_sequences": max(rerank_num_return, 1) if use_rerank else 1,
                        }
                        if use_rerank:
                            row_gen["return_dict_in_generate"] = True
                            row_gen["output_scores"] = True
                        generated = model.generate(
                            input_ids=input_ids[row_idx : row_idx + 1],
                            attention_mask=attention_mask[row_idx : row_idx + 1],
                            **row_gen,
                        )
                        sequences = generated.sequences if use_rerank else generated
                        beam_preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                        if use_postprocessing:
                            beam_preds = [postprocess_translation(p) for p in beam_preds]

                        if use_rerank:
                            candidates = [
                                {"text": text, "origin": "beam", "rank": rank}
                                for rank, text in enumerate(beam_preds)
                            ]
                            for hit in retrieval_hits:
                                candidates.append(
                                    {
                                        "text": hit["translation"],
                                        "origin": "retrieval",
                                        "rank": len(candidates),
                                        "retrieval_score": hit["score"],
                                    }
                                )
                            pred = rerank_candidates(candidates, retrieval_hits, doc_history)
                            update_doc_history(doc_history, pred)
                            preds.append(pred)
                        else:
                            pred = beam_preds[0] if beam_preds else ""
                            update_doc_history(doc_history, pred)
                            preds.append(pred)

                all_preds.extend(preds)
                all_refs.extend(refs)

    metrics = compute_combined_score(all_preds, all_refs)
    return {**metrics, "predictions": all_preds, "references": all_refs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on sentence-level independent test (LB proxy)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, help="Path to exported model directory")
    group.add_argument("--checkpoint", type=str, help="Path to training checkpoint")
    group.add_argument("--all-exported", action="store_true", help="Evaluate all exported models")

    parser.add_argument("--test-file", type=str, default="data/processed/independent_test_set.csv",
                       help="Path to sentence-level test CSV (default: independent_test_set.csv)")
    parser.add_argument("--num-beams", type=int, default=8, help="Beam width (default: 8)")
    parser.add_argument("--length-penalty", type=float, default=1.5, help="Length penalty (default: 1.5)")
    parser.add_argument("--no-repeat", type=int, default=0, help="No repeat ngram size (default: 0)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2, keep small to share GPU)")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable postprocessing")
    parser.add_argument("--sweep", action="store_true", help="Sweep generation parameters")
    parser.add_argument("--mbr", action="store_true", help="Use MBR decoding (4 beam + 2 sample candidates)")
    parser.add_argument("--mbr-return", type=int, default=4, help="Beam candidates for MBR (default: 4)")
    parser.add_argument("--mbr-samples", type=int, default=2, help="Sampling candidates for MBR (default: 2)")
    parser.add_argument("--save-preds", type=str, default=None, help="Save predictions to CSV file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--use-fuzzy-retrieval", action="store_true", help="Enable fuzzy retrieval over a parallel corpus")
    parser.add_argument("--use-rerank", action="store_true", help="Rerank beam candidates with fuzzy retrieval support")
    parser.add_argument("--rerank-num-return", type=int, default=None, help="Number of beam candidates to generate before reranking")
    parser.add_argument("--retrieval-file", type=str, default=None, help="Optional retrieval corpus CSV with transliteration/translation columns")
    parser.add_argument("--retrieval-top-k", type=int, default=None, help="Number of fuzzy retrieval hits to keep")
    parser.add_argument("--retrieval-shortlist", type=int, default=None, help="Shortlist size before exact fuzzy scoring")
    parser.add_argument("--retrieval-min-score", type=float, default=None, help="Minimum fuzzy retrieval score")
    parser.add_argument("--retrieval-direct-threshold", type=float, default=None, help="Direct-pick threshold for the top retrieval hit")
    parser.add_argument("--retrieval-direct-margin", type=float, default=None, help="Minimum score margin for direct retrieval picks")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # Load test data
    test_df = pd.read_csv(args.test_file)
    logger.info("Loaded %d sentence-level test pairs from %s", len(test_df), args.test_file)

    if args.all_exported:
        exported_dir = Path("exported")
        if not exported_dir.exists():
            parser.error("`exported/` directory not found. Run export first or pass --model/--checkpoint.")
        model_dirs = sorted(exported_dir.iterdir())
        model_dirs = [
            d for d in model_dirs
            if d.is_dir() and ((d / "config.json").exists() or any(d.rglob("config.json")))
        ]
        if not model_dirs:
            parser.error("No exported model directories with config.json found under exported/.")
    elif args.model:
        model_dirs = [Path(args.model)]
    elif args.checkpoint:
        model_dirs = [Path(args.checkpoint)]
    else:
        parser.error("Specify --model, --checkpoint, or --all-exported")

    results: list[dict[str, Any]] = []

    for model_path in model_dirs:
        name = model_path.stem if model_path.suffix == ".pt" else model_path.name
        logger.info("=" * 60)
        logger.info("Evaluating: %s", name)

        try:
            if model_path.suffix == ".pt" or args.checkpoint:
                model, tokenizer, meta = load_checkpoint_model(model_path, device)
            else:
                model, tokenizer, meta = load_exported_model(model_path, device)
        except Exception as e:
            logger.error("Failed to load %s: %s", name, e)
            continue

        prefix = meta.get("source_prefix", "")
        remove_gaps = meta.get("remove_gaps", False)
        logger.info("  prefix=%r, remove_gaps=%s", prefix, remove_gaps)
        inference_overrides = meta.get("inference_overrides", {})
        use_fuzzy_retrieval = args.use_fuzzy_retrieval or bool(
            inference_overrides.get(
                "use_fuzzy_retrieval",
                _DEFAULT_INFERENCE_SETTINGS["use_fuzzy_retrieval"],
            )
        )
        use_rerank = args.use_rerank or bool(
            inference_overrides.get(
                "use_reranking",
                _DEFAULT_INFERENCE_SETTINGS["use_reranking"],
            )
        )
        rerank_num_return = int(
            args.rerank_num_return
            if args.rerank_num_return is not None
            else inference_overrides.get(
                "rerank_num_return",
                _DEFAULT_INFERENCE_SETTINGS["rerank_num_return"],
            )
        )
        retrieval_top_k = int(
            args.retrieval_top_k
            if args.retrieval_top_k is not None
            else inference_overrides.get(
                "retrieval_top_k", _DEFAULT_INFERENCE_SETTINGS["retrieval_top_k"]
            )
        )
        retrieval_shortlist = int(
            args.retrieval_shortlist
            if args.retrieval_shortlist is not None
            else inference_overrides.get(
                "retrieval_shortlist",
                _DEFAULT_INFERENCE_SETTINGS["retrieval_shortlist"],
            )
        )
        retrieval_min_score = float(
            args.retrieval_min_score
            if args.retrieval_min_score is not None
            else inference_overrides.get(
                "retrieval_min_score",
                _DEFAULT_INFERENCE_SETTINGS["retrieval_min_score"],
            )
        )
        retrieval_direct_threshold = float(
            args.retrieval_direct_threshold
            if args.retrieval_direct_threshold is not None
            else inference_overrides.get(
                "retrieval_direct_threshold",
                _DEFAULT_INFERENCE_SETTINGS["retrieval_direct_threshold"],
            )
        )
        retrieval_direct_margin = float(
            args.retrieval_direct_margin
            if args.retrieval_direct_margin is not None
            else inference_overrides.get(
                "retrieval_direct_margin",
                _DEFAULT_INFERENCE_SETTINGS["retrieval_direct_margin"],
            )
        )
        logger.info(
            "  inference: fuzzy=%s rerank=%s rerank_num_return=%d top_k=%d shortlist=%d min_score=%.2f direct=(%.2f, %.2f)",
            use_fuzzy_retrieval,
            use_rerank,
            rerank_num_return,
            retrieval_top_k,
            retrieval_shortlist,
            retrieval_min_score,
            retrieval_direct_threshold,
            retrieval_direct_margin,
        )

        # Prepare data — independent_test_set.csv is already preprocessed
        texts = test_df["transliteration"].tolist()
        refs = test_df["translation"].tolist()
        fallback_doc_ids = pd.Series(test_df.index.astype(str), index=test_df.index)
        if "doc_id" in test_df.columns:
            doc_ids = test_df["doc_id"].fillna(fallback_doc_ids).astype(str).tolist()
        elif "text_id" in test_df.columns:
            doc_ids = test_df["text_id"].fillna(fallback_doc_ids).astype(str).tolist()
        elif "oare_id" in test_df.columns:
            doc_ids = test_df["oare_id"].fillna(fallback_doc_ids).astype(str).tolist()
        else:
            doc_ids = test_df.index.astype(str).tolist()

        # If model uses remove_gaps, remove any gap tokens from already-preprocessed text
        if remove_gaps:
            texts = [t.replace("<big_gap>", "").replace("<gap>", "") for t in texts]
            texts = [re.sub(r"\s+", " ", t).strip() for t in texts]

        retrieval_file = args.retrieval_file
        if retrieval_file is None and model_path.suffix != ".pt":
            bundled = Path(model_path) / "retrieval_corpus.csv"
            if not bundled.exists():
                matches = list(Path(model_path).rglob("retrieval_corpus.csv"))
                if matches:
                    bundled = matches[0]
            if bundled.exists():
                retrieval_file = str(bundled)

        exact_memory: dict[str, str] = {}
        retrieval_index: dict[str, Any] | None = None
        exact_memory, retrieval_index = build_retrieval_bank(
            retrieval_file,
            remove_gaps=remove_gaps,
        )
        logger.info(
            "  retrieval bank=%d exact entries (%s)",
            len(exact_memory),
            retrieval_file or "train.csv",
        )

        if args.sweep:
            # Sweep key generation parameters
            sweep_configs = [
                {"num_beams": 4, "length_penalty": 1.0},
                {"num_beams": 4, "length_penalty": 1.2},
                {"num_beams": 4, "length_penalty": 1.5},
                {"num_beams": 8, "length_penalty": 1.0},
                {"num_beams": 8, "length_penalty": 1.3},
                {"num_beams": 8, "length_penalty": 1.5},
                {"num_beams": 10, "length_penalty": 1.2},
                {"num_beams": 10, "length_penalty": 1.5},
            ]
            for sc in sweep_configs:
                dataset = SentenceDataset(texts, refs, doc_ids=doc_ids, prefix=prefix)
                start = time.time()
                result = evaluate_model(
                    model, tokenizer, dataset, device,
                    num_beams=sc["num_beams"],
                    length_penalty=sc["length_penalty"],
                    no_repeat_ngram=args.no_repeat,
                    batch_size=args.batch_size,
                    use_postprocessing=not args.no_postprocess,
                    exact_memory=exact_memory,
                    retrieval_index=retrieval_index,
                    use_fuzzy_retrieval=use_fuzzy_retrieval,
                    use_rerank=use_rerank,
                    rerank_num_return=rerank_num_return,
                    retrieval_top_k=retrieval_top_k,
                    retrieval_shortlist=retrieval_shortlist,
                    retrieval_min_score=retrieval_min_score,
                    retrieval_direct_threshold=retrieval_direct_threshold,
                    retrieval_direct_margin=retrieval_direct_margin,
                )
                elapsed = time.time() - start
                logger.info(
                    "  beams=%d lp=%.1f → BLEU=%.2f chrF++=%.2f Combined=%.2f (%.0fs)",
                    sc["num_beams"], sc["length_penalty"],
                    result["bleu"], result["chrf"], result["combined"],
                    elapsed,
                )
                results.append({
                    "model": name, **sc,
                    "bleu": result["bleu"], "chrf": result["chrf"],
                    "combined": result["combined"],
                })
        else:
            dataset = SentenceDataset(texts, refs, doc_ids=doc_ids, prefix=prefix)
            start = time.time()
            result = evaluate_model(
                model, tokenizer, dataset, device,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram=args.no_repeat,
                batch_size=args.batch_size,
                use_postprocessing=not args.no_postprocess,
                use_mbr=args.mbr,
                mbr_num_return=args.mbr_return,
                mbr_sample_cands=args.mbr_samples,
                exact_memory=exact_memory,
                retrieval_index=retrieval_index,
                use_fuzzy_retrieval=use_fuzzy_retrieval,
                use_rerank=use_rerank,
                rerank_num_return=rerank_num_return,
                retrieval_top_k=retrieval_top_k,
                retrieval_shortlist=retrieval_shortlist,
                retrieval_min_score=retrieval_min_score,
                retrieval_direct_threshold=retrieval_direct_threshold,
                retrieval_direct_margin=retrieval_direct_margin,
            )
            elapsed = time.time() - start
            mbr_tag = " [MBR]" if args.mbr else ""
            logger.info(
                "  BLEU=%.2f  chrF++=%.2f  Combined=%.2f  (%.0fs)%s",
                result["bleu"], result["chrf"], result["combined"], elapsed, mbr_tag,
            )
            results.append({
                "model": name,
                "num_beams": args.num_beams,
                "length_penalty": args.length_penalty,
                "bleu": result["bleu"],
                "chrf": result["chrf"],
                "combined": result["combined"],
            })

            # Save predictions if requested
            if args.save_preds:
                pred_df = pd.DataFrame({
                    "transliteration": texts,
                    "reference": result["references"],
                    "prediction": result["predictions"],
                })
                pred_df.to_csv(args.save_preds, index=False)
                logger.info("  Predictions saved to %s", args.save_preds)

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY — Sentence-Level Independent Test (LB Proxy)")
        logger.info("=" * 80)
        logger.info("%-35s %6s %4s %7s %7s %7s", "Model", "Beams", "LP", "BLEU", "chrF++", "Combined")
        logger.info("-" * 80)
        for r in sorted(results, key=lambda x: -x["combined"]):
            logger.info(
                "%-35s %6d %4.1f %7.2f %7.2f %7.2f",
                r["model"], r["num_beams"], r["length_penalty"],
                r["bleu"], r["chrf"], r["combined"],
            )


if __name__ == "__main__":
    main()
