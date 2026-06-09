# Data access

This repository does **not** redistribute any corpus. The training and
evaluation data are governed by the licenses of their original providers; you
must obtain them yourself from the sources below. The code expects the raw files
under `data/raw/` (you can symlink this to wherever you keep local-only data).

## Primary corpus — Deep Past Initiative (Kaggle)

Old Assyrian commercial texts (~1900–1700 BCE), primarily from the Kültepe-Kaneš
merchant archives, with document-level transliteration–translation pairs.

- Competition: <https://www.kaggle.com/competitions/deep-past-initiative-machine-translation>
- Dataset mirror: <https://www.kaggle.com/datasets/shelterwf/deep-past-initiative-machine-translation>

Download and place these files in `data/raw/`:

| File | Role |
|---|---|
| `train.csv` | Document-level parallel pairs |
| `test.csv` | Competition test inputs |
| `sample_submission.csv` | Submission format reference |
| `published_texts.csv` | Published transliterations (corpus assembly) |
| `publications.csv` | Publication metadata (used by the external mix) |
| `Sentences_Oare_FirstWord_LinNum.csv` | Sentence-segmentation anchors |
| `OA_Lexicon_eBL.csv` | Old Assyrian lexicon (dictionary-gloss intervention) |
| `eBL_Dictionary.csv` | Concise Dictionary of Akkadian glosses |

Usage of the Kaggle data is subject to the competition rules and the data
providers' terms; review them before redistribution or commercial use.

## Optional external corpus — ORACC alignment mix

The data-scaling / external-data experiments add ~29K multi-dialect aligned
pairs derived from a public Hugging Face dataset:

- <https://huggingface.co/datasets/phucthaiv02/akkadian_english_sentences_alignment_2>

Build the Old-Assyrian-focused mix with:

```bash
uv run akkadian-mt data build-external-mix \
  --output-csv data/processed/external_alignment_phucthaiv_translation_oa_mix_v1.csv
```

The underlying transliterations originate from the
[Open Richly Annotated Cuneiform Corpus (ORACC)](http://oracc.museum.upenn.edu/);
respect ORACC's attribution and licensing terms.

## After downloading

```bash
# build the document-level holdout union and the preprocessing variants
uv run python coursework/scripts/build_coursework_holdout_union.py
uv run python coursework/scripts/build_coursework_variants.py
```

Derived datasets are written under `data/processed/`. Both `data/raw/` and
`data/processed/` are git-ignored.
