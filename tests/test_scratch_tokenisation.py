"""Regression tests for the from-scratch tokenisation contrast script.

The original script built its SentencePiece tokenizer via
``T5Tokenizer(vocab_file=...)``, which under transformers v5 silently loads a
specials-only vocabulary: every word maps to <unk>, training collapses to
whitespace output with near-zero loss (the broken scratch_sp4k_* runs).
These tests pin the wrapper that replaced it.
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "train_scratch_tokenisation",
    Path(__file__).resolve().parents[1] / "experiments/scripts/train_scratch_tokenisation.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SENTENCES = [
    "translate Akkadian to English: um-ma a-šur-i-di-ma a-na a-šur-na-da",
    "From Ali-ahum to Damiq-pi-Assur: 30 minas of tin and 10 textiles.",
    "Thus says Assur-idi to Assur-nada, my son, concerning the silver.",
    "IGI pu-šu-ke-en DUMU su-e-a a-na a-lim ki-ma a-wa-tim",
    "You must pay the tin and the textiles to the merchant in the city.",
] * 30


@pytest.fixture(scope="module")
def sp_tok(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("sp")
    corpus = workdir / "sp_corpus.txt"
    corpus.write_text("\n".join(SENTENCES), encoding="utf-8")
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(workdir / "sp_akk"),
        vocab_size=70,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
    )
    return mod.SentencePieceSeq2SeqTokenizer(workdir / "sp_akk.model")


def test_vocab_actually_loaded(sp_tok):
    # the transformers-v5 failure mode was len(tok) == 4 (specials only)
    assert len(sp_tok) == 70


def test_encoding_is_not_degenerate(sp_tok):
    text = SENTENCES[1]
    ids = sp_tok(text)["input_ids"]
    unk_frac = sum(1 for i in ids if i == sp_tok.unk_token_id) / len(ids)
    assert unk_frac < 0.1
    assert ids[-1] == sp_tok.eos_token_id


def test_target_roundtrip(sp_tok):
    import torch

    text = SENTENCES[2]
    lab = sp_tok(text_target=text)["input_ids"]
    decoded = sp_tok.batch_decode(torch.tensor([lab]))
    assert decoded[0].strip() == text


def test_batch_encode_pads_and_masks(sp_tok):
    enc = sp_tok(SENTENCES[:3], return_tensors="pt", padding=True, truncation=True, max_length=64)
    assert enc["input_ids"].shape == enc["attention_mask"].shape
    assert enc["attention_mask"].max() == 1
    lengths = enc["attention_mask"].sum(dim=1)
    assert lengths.min() > 2

def test_truncation_respects_max_length(sp_tok):
    ids = sp_tok(" ".join(SENTENCES), max_length=16)["input_ids"]
    assert len(ids) == 16
    assert ids[-1] == sp_tok.eos_token_id
