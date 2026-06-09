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
| `train.csv` | Document-level parallel pairs (the core corpus) |
| `Sentences_Oare_FirstWord_LinNum.csv` | Sentence-segmentation anchors |
| `OA_Lexicon_eBL.csv` | Old Assyrian lexicon (dictionary-gloss intervention) |
| `eBL_Dictionary.csv` | Concise Dictionary of Akkadian glosses |

Usage of the Kaggle data is subject to the competition rules and the data
providers' terms; review them before redistribution or commercial use.

## Optional external corpus — ORACC alignment mix

The data-scaling / external-data experiments add ~29K multi-dialect aligned
pairs derived from a public Hugging Face dataset:

- <https://huggingface.co/datasets/phucthaiv02/akkadian_english_sentences_alignment_2>

These experiments read a prepared CSV at
`data/processed/external_alignment_phucthaiv_translation_oa_mix_v1.csv` (the path
referenced by the `*_external_mix` configs). The bespoke extraction/cleaning
pipeline that produced this file is **not** part of this release — it depended on
OCR'd publication scans and was orthogonal to the paper's core findings. To
reproduce these (secondary) experiments, supply a CSV with `transliteration` and
`translation` columns built from the dataset above. The transliterations
originate from the
[Open Richly Annotated Cuneiform Corpus (ORACC)](http://oracc.museum.upenn.edu/);
respect ORACC's attribution and licensing terms.

## After downloading

```bash
# build the document-level holdout union and the preprocessing variants
uv run python experiments/scripts/build_holdout_union.py
uv run python experiments/scripts/build_variants.py
```

Derived datasets are written under `data/processed/`. Both `data/raw/` and
`data/processed/` are git-ignored.
