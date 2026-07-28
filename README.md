# Byte-Level Machine Translation for Old Assyrian Akkadian

Neural machine translation from romanised Akkadian cuneiform transliterations to
English, in an extreme low-resource setting (~6.7K parallel sentence pairs, no
pretrained Akkadian language model). This repository accompanies the study
*"How does model capacity interact with domain-specific preprocessing in
low-resource ancient-language machine translation?"* ([paper](paper/paper.pdf)).

## TL;DR — what the study found

Across a 2×2 tokenisation × capacity design plus seven preprocessing
interventions, three results stand out (metric: `combined = √(BLEU × chrF++)`,
the [Deep Past Initiative](https://www.kaggle.com/competitions/deep-past-initiative)
competition score, on the held-out *independent* test set):

1. **Tokenisation dominates capacity.** Byte-level ByT5 beats subword mT5 by
   **+19** combined points averaged across sizes — more than double the **+8**
   from doubling byte-level capacity. A TF-IDF retrieval baseline (18.2) even
   beats the 300M subword Transformer (11.6).
2. **Capacity absorbs preprocessing.** Six of seven preprocessing interventions,
   from two methodologically independent families, help the 300M model but
   hurt or do nothing for the 580M model (one-sided sign test *p* ≈ 0.016),
   replicated on a second held-out test set.
3. **Effect hierarchy:** tokenisation (**+19**) ≈ data quantity (**+18**) ≫
   capacity (**+8**) ≫ preprocessing (**< 1** for the larger model). Practical
   takeaway: at this scale, spend effort on data, not hand-crafted preprocessing.

### Tokenisation × capacity (independent test set)

| Model | Tokenisation | Params | BLEU | chrF++ | Combined |
|---|---|---|---|---|---|
| TF-IDF 1-NN | — | — | 12.20 | 27.02 | 18.16 |
| mT5-small | SentencePiece | 300M | 6.93 | 19.32 | 11.58 |
| mT5-base | SentencePiece | 580M | 9.52 | 24.69 | 15.33 |
| ByT5-small | Byte-level | 300M | 20.66 | 39.82 | 28.69 |
| **ByT5-base** | Byte-level | 580M | **27.56** | **48.54** | **36.58** |

## Repository layout

```
akkadian_mt/            # installable package
  data/                 # preprocessing pipeline, dataset construction, dictionary, interventions
  models/               # T5-family seq2seq wrappers + checkpoint compatibility
  training/             # training loop (Adafactor, early stopping, wandb logging)
  evaluation/           # BLEU / chrF++ / combined-score metrics + held-out eval
  train.py, evaluate.py, report.py, cli.py
experiments/
  configs/              # one YAML per experimental condition (31 configs)
  scripts/              # dataset build + evaluation + results-collection orchestration
  results/              # per-run metric artifacts (the evidence behind every table)
  report/               # figure generator + paper figures
paper/                  # the paper (PDF + markdown source)
tests/                  # regression + parity tests for the package
```

## Install

Requires Python ≥ 3.11. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run akkadian-mt --help
```

Or with pip:

```bash
pip install -e ".[dev]"
akkadian-mt --help
```

## Quick start

The training data is **not redistributed** here (see [DATA_ACCESS.md](DATA_ACCESS.md)
for how to obtain the Deep Past Initiative corpus and the optional ORACC
external mix). Once the raw files are in place:

```bash
# 1. build the document-level holdout union and the preprocessing variants
uv run python experiments/scripts/build_holdout_union.py
uv run python experiments/scripts/build_variants.py

# 2. train a model from any experiment config
uv run akkadian-mt train seq2seq --config experiments/configs/byt5_base_baseline.yaml

# 3. evaluate on the held-out test set
uv run akkadian-mt eval lb --config experiments/configs/byt5_base_baseline.yaml \
  --checkpoint outputs/coursework/byt5_base_baseline/best_model.pt
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full report-to-artifact map
and the exact commands behind every table and figure.

## What makes the source format hard

Romanised cuneiform transliteration carries notation that has no counterpart in
the web text used to pretrain multilingual models: diacritics (š, ṭ),
unpronounced determinatives (`{d}`, `{ki}`), Sumerographic logograms
(`KÙ.BABBAR` = silver), homophone subscripts, and scribal-damage brackets. The
deterministic preprocessing pipeline in `akkadian_mt/data/preprocessing.py`
normalises these before tokenisation; the seven interventions in
`akkadian_mt/data/interventions.py` probe whether encoding them explicitly helps.

## Citation

If you use this work, please cite the paper. See [CITATION.cff](CITATION.cff).

## License

Code is released under the [MIT License](LICENSE). The underlying corpora are
governed by their own licenses — see [DATA_ACCESS.md](DATA_ACCESS.md).
