# Data access

This repository redistributes no corpus data. The texts are governed by the
licences of their original providers, so you download them yourself. The code
reads raw files from `data/raw/` and writes derived files to `data/processed/`.
Both directories are ignored by git, and `data/raw/` can be a symlink to wherever
you keep the downloads.

## Primary corpus, Deep Past Initiative on Kaggle

Old Assyrian commercial texts of roughly 1900 to 1700 BCE, mostly from the
Kültepe-Kaneš merchant archives, released with document-level transliteration and
translation pairs.

Competition page:
<https://www.kaggle.com/competitions/deep-past-initiative-machine-translation>

Download these four files into `data/raw/`:

| File | Role |
|---|---|
| `train.csv` | Document-level parallel pairs, the core corpus |
| `Sentences_Oare_FirstWord_LinNum.csv` | Sentence-segmentation anchors |
| `OA_Lexicon_eBL.csv` | Old Assyrian lexicon, used by the dictionary-gloss intervention |
| `eBL_Dictionary.csv` | Concise Dictionary of Akkadian glosses, used by the same intervention |

The file names are the defaults in `akkadian_mt/config.py` and can be changed
there or in a config's `data` section. Use of the Kaggle data is subject to the
competition rules and the providers' terms. Read them before redistributing the
data or using it commercially.

## Derived files the experiments read

Every experiment config in this repository uses `data_mode: prepared`, which
means it reads a materialised CSV rather than assembling the corpus at run time.
Five derived files sit at the bottom of that chain. The corpus assembly step that
produced them is not part of this release, so reconstruct them from the raw files
before running anything. Their shapes are below.

| File | Rows | Columns |
|---|---|---|
| `data/processed/golden_corpus_v2_prefilt.csv` | 6,746 | `transliteration`, `translation`, `oare_id`, `genre`, `source`, `alignment_method` |
| `data/processed/independent_test_set_clean.csv` | 951 | `transliteration`, `translation`, `doc_id`, `genre`, `sentence_idx`, `anchor_valid`, `word_start`, `word_end` |
| `data/processed/new_test_set.csv` | 661 | `transliteration`, `translation`, `oare_id`, `genre`, `source` |
| `data/processed/holdout_doc_ids.json` | 152 | JSON list of document identifiers |
| `data/processed/new_holdout_doc_ids.json` | 130 | JSON list of document identifiers |

The gold corpus is `train.csv` segmented into sentences using the anchors in
`Sentences_Oare_FirstWord_LinNum.csv`, with quality filtering applied later by
the training pipeline. The two test sets are genre-stratified document samples
drawn from the same release and segmented separately. The two JSON lists name the
documents behind those test sets, and their union is excluded from every training
split.

Source-side normalisation is deterministic and lives in
`akkadian_mt/data/preprocessing.py`, so the training pipeline reproduces the same
model inputs from any correctly shaped gold corpus.

## Optional external corpus

The external-data condition mixes the gold corpus with about 29,000 aligned pairs
spanning several Akkadian dialects. The publicly available part derives from this
Hugging Face dataset:

<https://huggingface.co/datasets/phucthaiv02/akkadian_english_sentences_alignment_2>

The `*_external_mix` configs read a prepared CSV at
`data/processed/external_alignment_phucthaiv_translation_oa_mix_v1.csv`. The
extraction and cleaning pipeline that built that file is not part of this
release, because it depended on scanned publications that cannot be redistributed.
To rerun this secondary condition, supply a CSV with `transliteration` and
`translation` columns built from the dataset above. Those transliterations come
from the Open Richly Annotated Cuneiform Corpus
(<http://oracc.museum.upenn.edu/>), whose attribution and licensing terms apply.

## After downloading

```bash
# holdout document-id union, written to data/processed/
uv run python experiments/scripts/build_holdout_union.py

# preprocessing-variant datasets, one subdirectory per intervention under
# data/processed/, each holding a train file and the two test files
uv run python experiments/scripts/build_variants.py

# 25%, 50% and 75% document-level training subsets for the scaling experiments
uv run python experiments/scripts/build_scaling_subsets.py
```

Each script takes command-line flags for its input and output paths. Run any of
them with `--help` to see the defaults.
