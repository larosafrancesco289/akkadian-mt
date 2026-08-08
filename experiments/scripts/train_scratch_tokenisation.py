"""Matched-architecture tokenisation contrast, trained from scratch.

Trains the same randomly initialised T5 architecture on the gold corpus twice:
once with an in-domain SentencePiece vocabulary learned from the training split,
once directly on UTF-8 bytes. Because the two conditions share architecture,
data, and training protocol, the contrast isolates the tokenisation axis from
pretraining and vocabulary-fit confounds that affect the mT5/ByT5 comparison.

Writes per-run artifacts in the same layout as run_eval.py
(experiments/results/artifacts/<run_id>/{summary.json,*_metrics.json,*_preds.csv}).

Usage:
  uv run python experiments/scripts/train_scratch_tokenisation.py \
      --tokenisation sp --seed 42 --run-id scratch_sp8k_seed42
  uv run python experiments/scripts/train_scratch_tokenisation.py \
      --tokenisation byte --seed 42 --run-id scratch_byte_seed42 --smoke
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")
from akkadian_mt.data.preprocessing import (  # noqa: E402
    META_RE,
    clean_translation,
    normalize_transliteration,
    postprocess_translation,
)

REPO = Path(".")
CORPUS = REPO / "data/processed/golden_corpus_v2_prefilt.csv"
HOLDOUT = REPO / "data/processed/coursework_holdout_doc_ids.json"
TESTS = {
    "independent": REPO / "data/processed/independent_test_set_clean.csv",
    "newtest": REPO / "data/processed/new_test_set.csv",
}
SOURCE_PREFIX = "translate Akkadian to English: "


def norm_src(text: str) -> str:
    return normalize_transliteration(text, do_strip_homophone_subscripts=True)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quality_ok(row) -> bool:
    src, tgt = row["transliteration"], row["translation"]
    if len(tgt) < 10:
        return False
    a, b = len(src), len(tgt)
    if max(a, b) / max(min(a, b), 1) > 10.0:
        return False
    if META_RE.search(tgt):
        return False
    return True


def build_splits(seed=42, val_frac=0.1):
    holdout = set(json.loads(HOLDOUT.read_text()))
    rows = [r for r in load_rows(CORPUS) if quality_ok(r) and r["oare_id"] not in holdout]
    docs = sorted({r["oare_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_frac))
    val_docs = set(docs[:n_val])
    train = [r for r in rows if r["oare_id"] not in val_docs]
    val = [r for r in rows if r["oare_id"] in val_docs]
    return train, val


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_src, max_tgt):
        self.items = []
        for r in rows:
            src = SOURCE_PREFIX + norm_src(r["transliteration"])
            tgt = clean_translation(r["translation"])
            self.items.append((src, tgt))
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, tgt = self.items[i]
        enc = self.tok(src, max_length=self.max_src, truncation=True)
        lab = self.tok(text_target=tgt, max_length=self.max_tgt, truncation=True)
        return {"input_ids": enc["input_ids"], "labels": lab["input_ids"]}


def collate(batch, pad_id):
    def pad(seqs, value):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (m - len(s)) for s in seqs], dtype=torch.long)

    input_ids = pad([b["input_ids"] for b in batch], pad_id)
    labels = pad([b["labels"] for b in batch], -100)
    attn = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class SentencePieceSeq2SeqTokenizer:
    """Minimal seq2seq tokenizer over a raw SentencePiece model.

    transformers v5 removed the slow ``T5Tokenizer(vocab_file=...)`` path this
    script originally used; constructing it that way silently yields a
    specials-only vocabulary (every word -> <unk>), which collapses training.
    This wrapper drives sentencepiece directly and exposes only the interface
    the script needs (callable encode, batch_decode, pad/eos ids, len).
    """

    def __init__(self, model_file):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def __len__(self):
        return self.sp.get_piece_size()

    def _encode_one(self, text, max_length):
        ids = self.sp.encode(text)
        if max_length is not None:
            ids = ids[: max_length - 1]
        return ids + [self.eos_token_id]

    def __call__(self, text=None, text_target=None, max_length=None, truncation=True,
                 padding=False, return_tensors=None):
        if text_target is not None:
            return {"input_ids": self._encode_one(text_target, max_length)}
        if isinstance(text, str):
            return {"input_ids": self._encode_one(text, max_length)}
        from transformers import BatchEncoding

        seqs = [self._encode_one(t, max_length) for t in text]
        m = max(len(s) for s in seqs)
        input_ids = [s + [self.pad_token_id] * (m - len(s)) for s in seqs]
        attn = [[1] * len(s) + [0] * (m - len(s)) for s in seqs]
        return BatchEncoding(
            {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attn)}
        )

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.sp.decode([i for i in seq if i > self.unk_token_id]) for seq in ids.tolist()]


def make_tokenizer(kind, train_rows, vocab_size, workdir):
    if kind == "byte":
        from transformers import ByT5Tokenizer

        return ByT5Tokenizer()
    import sentencepiece as spm

    corpus_txt = workdir / "sp_corpus.txt"
    with open(corpus_txt, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(SOURCE_PREFIX + norm_src(r["transliteration"]) + "\n")
            f.write(clean_translation(r["translation"]) + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus_txt),
        model_prefix=str(workdir / "sp_akk"),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
    )
    tok = SentencePieceSeq2SeqTokenizer(workdir / "sp_akk.model")
    assert len(tok) == vocab_size, f"SP vocab loaded {len(tok)} pieces, expected {vocab_size}"
    return tok


@torch.no_grad()
def generate(model, tok, rows, device, max_src, num_beams, length_penalty, batch_size=32):
    model.eval()
    preds = []
    srcs = [SOURCE_PREFIX + norm_src(r["transliteration"]) for r in rows]
    for i in range(0, len(srcs), batch_size):
        enc = tok(srcs[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_src).to(device)
        out = model.generate(
            **enc, max_new_tokens=384, num_beams=num_beams, length_penalty=length_penalty
        )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    return preds


def combined_score(preds, refs):
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    return bleu, chrf, math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenisation", choices=["sp", "byte"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 200 rows, 2 epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = REPO / "experiments/results/artifacts" / args.run_id
    art.mkdir(parents=True, exist_ok=True)
    workdir = REPO / "outputs/scratch" / args.run_id
    workdir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = build_splits()
    if args.smoke:
        train_rows, val_rows = train_rows[:200], val_rows[:40]
        args.vocab_size = min(args.vocab_size, 500)

    tok = make_tokenizer("byte" if args.tokenisation == "byte" else "sp", train_rows, args.vocab_size, workdir)

    from transformers import T5Config, T5ForConditionalGeneration

    dim = 64 if args.smoke else 512
    layers = 2 if args.smoke else 4
    config = T5Config(
        vocab_size=max(len(tok), 384),
        d_model=dim,
        d_ff=dim * 4,
        d_kv=dim // 8 if dim >= 64 else 8,
        num_layers=layers,
        num_decoder_layers=layers,
        num_heads=8,
        dropout_rate=0.1,
        decoder_start_token_id=tok.pad_token_id,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        feed_forward_proj="gated-gelu",
    )
    model = T5ForConditionalGeneration(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_nonemb = n_params - model.get_input_embeddings().weight.numel()

    max_src, max_tgt = (512, 384) if args.tokenisation == "byte" else (256, 192)
    ds = PairDataset(train_rows, tok, max_src, max_tgt)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        generator=torch.Generator().manual_seed(args.seed),
    )

    from transformers.optimization import Adafactor

    opt = Adafactor(model.parameters(), lr=args.lr, relative_step=False, scale_parameter=False, warmup_init=False)
    epochs = 2 if args.smoke else args.epochs

    best_val = -1.0
    best_epoch = -1
    bad = 0
    history = []
    val_refs = [clean_translation(r["translation"]) for r in val_rows]
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total += loss.item()
        val_preds = [postprocess_translation(p) for p in generate(model, tok, val_rows, device, max_src, 1, 1.0)]
        _, _, val_comb = combined_score(val_preds, val_refs)
        history.append({"epoch": epoch, "train_loss": total / max(1, len(dl)), "val_combined": val_comb})
        print(f"epoch {epoch}: loss {total/max(1,len(dl)):.3f} val_combined {val_comb:.2f}", flush=True)
        if val_comb > best_val:
            best_val, best_epoch, bad = val_comb, epoch, 0
            torch.save(model.state_dict(), workdir / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(workdir / "best_model.pt", map_location=device))
    summary = {
        "run_id": args.run_id,
        "tokenisation": args.tokenisation,
        "seed": args.seed,
        "vocab_size": len(tok),
        "params_total": n_params,
        "params_non_embedding": n_nonemb,
        "best_epoch": best_epoch,
        "val_combined": best_val,
        "history": history,
    }
    for name, path in TESTS.items():
        rows = load_rows(path)
        if args.smoke:
            rows = rows[:40]
        refs = [clean_translation(r["translation"]) for r in rows]
        preds = [postprocess_translation(p) for p in generate(model, tok, rows, device, max_src, 8, 1.3)]
        bleu, chrf, comb = combined_score(preds, refs)
        summary[name] = {"bleu": bleu, "chrf": chrf, "combined": comb}
        (art / f"{'independent' if name == 'independent' else 'newtest'}_metrics.json").write_text(
            json.dumps({"bleu": bleu, "chrf": chrf, "combined": comb})
        )
        with open(art / f"{'independent' if name == 'independent' else 'newtest'}_preds.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["transliteration", "translation", "genre", "prediction"])
            for r, p in zip(rows, preds):
                w.writerow([r["transliteration"], r["translation"], r.get("genre", ""), p])
        print(f"{name}: BLEU {bleu:.2f} chrF++ {chrf:.2f} combined {comb:.2f}", flush=True)
    (art / "summary.json").write_text(json.dumps(summary, indent=2))
    print("artifacts written to", art, flush=True)


if __name__ == "__main__":
    main()
