# Akkadian to English Machine Translation (Old Assyrian)

Code, experiment configurations, and per-run result artifacts for the
accompanying paper submission.

The task is sentence-level translation from romanised cuneiform transliteration
into English. The setting is extremely low-resource. The aligned corpus holds
6,746 sentence pairs, the row count of the gold corpus file described in
[DATA_ACCESS.md](DATA_ACCESS.md), and no pretrained Akkadian language model
exists. The study compares the choices a system builder faces in that setting.
Those choices are the pretrained model family (byte-level against subword), the
model size, preprocessing schemes drawn from Assyriological practice, the
quantity of training data, and the learning rate.

All scores are `combined = sqrt(BLEU * chrF++)`, the score used by the Deep Past
Initiative Kaggle competition that released the corpus. Higher is better. The
implementation is in `akkadian_mt/evaluation/metrics.py`. Results are reported on
two held-out sets whose documents are excluded from every training split.

## What the study found

Scores below are on the independent test set and come from the metric files named
beside them. Multi-seed conditions are means over seeds 42, 52 and 62. The
shared-recipe mT5 runs and the TF-IDF baseline ran once each. `M/` abbreviates
`experiments/results/matrix_h100/artifacts/` and `A/` abbreviates
`experiments/results/artifacts/`.

- **Training data is the largest lever.** ByT5-base scores 19.03 on a quarter of
  the corpus and 36.46 on all of it, a gain of 17.4 points, and the ordering
  holds inside every individual seed. Source: `M/byt5_base_scale_{25,50,75}pct_*`
  and `M/byt5_base_baseline_seed*`.
- **Model size is second.** Going from 300M to 580M parameters adds 7.3 points,
  from 29.11 to 36.46. An exploratory 1.2B arm reaches 38.55, on a two-value
  learning-rate probe rather than the tuning the other sizes received. Source:
  `M/byt5_small_baseline_*`, `M/byt5_base_baseline_seed*`, `M/byt5_large_baseline_*`.
- **The byte-level lead over subword is largely a tuning artifact.** Under one
  shared fine-tuning recipe, mT5-base scores 15.33 against ByT5-base's 36.46.
  With the learning rate selected on validation data for each configuration,
  mT5-base reaches 35.47, within one point of that ByT5-base reference and 1.75
  below the ByT5-base baseline retuned to 1e-4 (37.22). Source:
  `A/mt5_base_baseline/`, `M/mt5_base_lr1e3_*`, `M/byt5_base_baseline_lr1e4_*`.
- **Preprocessing effects are small and depend on the same tuning.** Under the
  shared recipe, all six interventions lower the 580M model, by 0.66 combined
  points on average, while the 300M model averages a gain of 0.29. Two of them
  were then retrained at 1e-4 alongside the baseline. Genre conditioning's
  deficit disappears (37.48 over three seeds against the retuned baseline's
  37.22) and retrieval augmentation keeps its own (36.51, a single run). Source:
  `experiments/results/matrix_h100/interaction_check.py`,
  `M/byt5_base_genre_lr1e4`, `M/byt5_base_retrieval_lr1e4`,
  `M/byt5_base_baseline_lr1e4*`.

### Model family and size, independent test set

| Model | BLEU | chrF++ | Combined |
|---|---|---|---|
| TF-IDF 1-nearest-neighbour | 12.20 | 27.02 | 18.16 |
| *Shared recipe* | | | |
| mT5-small (300M) | 6.93 | 19.32 | 11.58 |
| mT5-base (580M) | 9.52 | 24.69 | 15.33 |
| ByT5-small (300M) | 21.07 | 40.23 | 29.11 |
| ByT5-base (580M) | 27.51 | 48.32 | 36.46 |
| *Learning rate tuned per configuration* | | | |
| mT5-small at 1e-3 | 26.10 | 46.03 | 34.66 |
| mT5-base at 1e-3 | 26.73 | 47.08 | 35.47 |

ByT5 rows and tuned mT5 rows are three-seed means. The shared-recipe mT5 rows and
the TF-IDF row are single runs. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) names
the artifact bundle behind every row.

## Setup

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run akkadian-mt --help
```

With pip:

```bash
pip install -e ".[dev]"
akkadian-mt --help
```

Checks that need no corpus and run on CPU in under a minute:

```bash
uv run ruff check akkadian_mt experiments/scripts tests
uv run pytest -q
```

The suite has 38 tests. They use dummy models and tokenizers, so they verify
wiring and invariants rather than real training quality.

## Quickstart

No corpus is redistributed here. Read [DATA_ACCESS.md](DATA_ACCESS.md) first and
put the raw files under `data/raw/`. Then build the derived datasets, train, and
evaluate:

```bash
# 1. holdout document-id union and the preprocessing-variant datasets
uv run python experiments/scripts/build_holdout_union.py
uv run python experiments/scripts/build_variants.py

# 2. fine-tune one condition (each YAML in experiments/configs/ is one condition)
uv run akkadian-mt train seq2seq --config experiments/configs/byt5_base_baseline.yaml

# 3. score the checkpoint on both held-out sets and write a run artifact
uv run python experiments/scripts/run_eval.py \
  --run-id my_byt5_base_baseline \
  --config experiments/configs/byt5_base_baseline.yaml \
  --checkpoint <train.output_dir from the config>/best_model.pt
```

Any config value can be overridden on the command line as `--section.key value`,
for example `--train.epochs 1` or `--train.batch_size 8`. Training logs to
Weights & Biases by default. Set `WANDB_MODE=offline` to avoid network calls.

## Repository layout

```
akkadian_mt/            installable package
  data/                 source normalisation, dataset construction, interventions
  models/               T5-family seq2seq wrappers, checkpoint compatibility
  training/             training loop with early stopping
  evaluation/           BLEU, chrF++, combined score, held-out evaluation
  cli.py                the akkadian-mt entrypoint
experiments/
  configs/              91 YAML files, one per run condition
  scripts/              dataset builds, evaluation, baselines, bootstrap tests
  results/artifacts/    32 per-run metric bundles from the earlier environment
  results/matrix_h100/  88 per-run metric bundles from the later environment,
                        plus the analysis and figure scripts
tests/                  unit and regression tests
```

Metric JSON files and `config_snapshot.yaml` files under `experiments/results/`
are frozen provenance. They record what was actually run and are not edited.

## Why the source format is hard

Romanised transliteration carries notation with no counterpart in the web text
that multilingual models are pretrained on. Diacritics distinguish cuneiform
signs that share a Latin syllable (s against š). Determinatives are unpronounced
class markers written as `{d}` or `{ki}`. Sumerographic logograms are Sumerian
shorthand inside Akkadian text, such as `KÙ.BABBAR` for silver. Subscript digits
on sign readings encode sign identity rather than pronunciation. The
deterministic pipeline in `akkadian_mt/data/preprocessing.py` normalises these
before tokenisation, and the interventions in
`akkadian_mt/data/interventions.py` test whether encoding them explicitly helps.

## Further documentation

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps every reported result to its
  config file and its stored artifact, and lists the commands that produce them.
- [DATA_ACCESS.md](DATA_ACCESS.md) explains how to obtain the corpus and what the
  build scripts expect on disk.

## License

Code is released under the [MIT License](LICENSE). The corpora keep the licences
of their original providers, described in [DATA_ACCESS.md](DATA_ACCESS.md).
