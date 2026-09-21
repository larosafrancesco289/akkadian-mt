# Reproducibility guide

Every result in the accompanying paper comes from a configuration file in this
repository and is stored as a per-run artifact bundle. This document names both
for each result, then lists the commands that regenerate them.

## What a run artifact contains

Each run directory holds `summary.json` (the source of truth for reported
metrics), `independent_metrics.json` and `newtest_metrics.json` (the two held-out
sets), and, for configured runs, `config_snapshot.yaml` (the configuration
actually used). `A/tfidf_baseline` stores its snapshot as JSON and the
`M/scratch_*` directories have none. Six runs in the later pool also carry
`training_history.csv`. Model checkpoints are archived outside the repository
because of their size. Prediction CSVs are present for every run in the earlier
pool and for 46 runs in the later pool.

These files are frozen provenance. They record what was run and are never edited.

## The two artifact pools

Runs were executed in two software and hardware environments, and the paper
keeps every fine-grained comparison inside one of them. The two contrasts that
cross environments, first-round mT5 and TF-IDF against later-pool ByT5, involve
gaps an order of magnitude larger than the measured environment noise.

- `experiments/results/artifacts/` holds 32 runs from the earlier environment.
  `experiments/results/results.csv` is the flat registry for this pool, with one
  row per run and both test sets' scores.
- `experiments/results/matrix_h100/artifacts/` holds 88 runs from the later
  environment, including the full multi-seed intervention matrix, the tuned mT5
  grid, the data-scaling grid, the 1.2B arm, and the from-scratch pair.

Run identifiers in the later pool follow several conventions. A `_seed42r` suffix
marks a seed-42 rerun of the original config in the later environment. A `_r2`
suffix does the same for the data-scaling subsets. `byt5_small_baseline_regression42`
is the seed-42 row for the ByT5-small baseline.

Below, `M/` abbreviates `experiments/results/matrix_h100/artifacts/` and `A/`
abbreviates `experiments/results/artifacts/`. `C/` abbreviates
`experiments/configs/`.

## Prerequisites

```bash
uv sync --extra dev
```

Obtain the corpus as described in [DATA_ACCESS.md](DATA_ACCESS.md), place the raw
files under `data/raw/`, and build the derived datasets:

```bash
uv run python experiments/scripts/build_holdout_union.py
uv run python experiments/scripts/build_variants.py
uv run python experiments/scripts/build_scaling_subsets.py
```

The external-data condition additionally needs a prepared external mix CSV, which
[DATA_ACCESS.md](DATA_ACCESS.md) describes.

## Result-to-artifact map

### Data scaling

| Condition | Configs | Artifacts |
|---|---|---|
| 25% of documents | `C/byt5_base_scale_25pct{,_seed52,_seed62}.yaml` | `M/byt5_base_scale_25pct_r2`, `M/byt5_base_scale_25pct_seed{52,62}` |
| 50% | `C/byt5_base_scale_50pct{,_seed52,_seed62}.yaml` | `M/byt5_base_scale_50pct_r2`, `M/byt5_base_scale_50pct_seed{52,62}` |
| 75% | `C/byt5_base_scale_75pct{,_seed52,_seed62}.yaml` | `M/byt5_base_scale_75pct_r2`, `M/byt5_base_scale_75pct_seed{52,62}` |
| 100% | `C/byt5_base_baseline{,_seed52,_seed62}.yaml` | `M/byt5_base_baseline_seed42r`, `M/byt5_base_baseline_seed{52,62}` |
| External mix, 580M | `C/byt5_base_external_mix{,_seed52,_seed62}.yaml` | `A/byt5_base_external_mix{,_seed52,_seed62}` |
| External mix, 300M | `C/byt5_small_external_mix.yaml` | `A/byt5_small_external_mix` |

### Model family and size

| Condition | Configs | Artifacts |
|---|---|---|
| TF-IDF retrieval baseline | none, see the script below | `A/tfidf_baseline` |
| mT5, shared recipe | `C/mt5_small_baseline.yaml`, `C/mt5_base_baseline.yaml` | `A/mt5_small_baseline`, `A/mt5_base_baseline` |
| ByT5-small, shared recipe | `C/byt5_small_baseline{,_seed52,_seed62}.yaml` | `M/byt5_small_baseline_regression42`, `M/byt5_small_baseline_seed{52,62}` |
| ByT5-base, shared recipe | `C/byt5_base_baseline{,_seed52,_seed62}.yaml` | `M/byt5_base_baseline_seed42r`, `M/byt5_base_baseline_seed{52,62}` |
| mT5 learning-rate grid | `C/mt5_small_lr3e4.yaml`, `C/mt5_small_lr1e3{,_seed52,_seed62}.yaml`, `C/mt5_base_lr3e4{,_seed52,_seed62}.yaml`, `C/mt5_base_lr1e3{,_seed52,_seed62}.yaml` | `M/` directories of the same names |
| From-scratch tokenisation contrast | none, see the script below | `M/scratch_byte_seed{42,52,62}`, `M/scratch_sp4k_fixed_seed{42,52,62}` |

The `M/scratch_sp4k_seed{42,52,62}` directories are a first attempt whose
SentencePiece setup was faulty. They score zero and are not reported. The
`_fixed` runs replace them.

### Preprocessing interventions

Six interventions run at both model sizes with seeds 42, 52 and 62. Substitute
`<i>` and `<size>` in the two patterns below.

- configs: `C/byt5_<size>_<i>.yaml`, `C/byt5_<size>_<i>_seed52.yaml`, `C/byt5_<size>_<i>_seed62.yaml`
- artifacts: `M/byt5_<size>_<i>_seed42r`, `M/byt5_<size>_<i>_seed52`, `M/byt5_<size>_<i>_seed62`

with `<size>` in {`small`, `base`} and `<i>` one of `rq1_no_determinatives`,
`rq2_tag_sumerograms`, `rq3_dictionary_gloss`, `cw4_genre_conditioned`,
`cw4_numeral_normalized`, `cw4_retrieval_augmented`. Baselines are the ByT5 rows
of the previous table.

A seventh candidate, gap-marker removal, is a no-op on this corpus release and is
excluded from the paper. Its single-seed runs remain at
`A/byt5_{small,base}_rq4_remove_gaps` from `C/byt5_{small,base}_rq4_remove_gaps.yaml`.

### Retuning the harmful interventions at 580M

| Condition | Configs | Artifacts |
|---|---|---|
| Baseline at 1e-4 | `C/byt5_base_baseline_lr1e4{,_seed52,_seed62}.yaml` | `M/byt5_base_baseline_lr1e4{,_seed52,_seed62}` |
| Genre conditioning at 1e-4 | `C/byt5_base_cw4_genre_conditioned_lr1e4{,_seed52,_seed62}.yaml` | `M/byt5_base_genre_lr1e4`, `M/byt5_base_cw4_genre_conditioned_lr1e4_seed{52,62}` |
| Genre conditioning at 5e-5 | `C/byt5_base_cw4_genre_conditioned_lr5e5.yaml` | `M/byt5_base_genre_lr5e5` |
| Retrieval augmentation at 1e-4 | `C/byt5_base_cw4_retrieval_augmented_lr1e4.yaml` | `M/byt5_base_retrieval_lr1e4` |
| Retrieval augmentation at 5e-5 | `C/byt5_base_cw4_retrieval_augmented_lr5e5.yaml` | `M/byt5_base_retrieval_lr5e5` |

The seed-42 run identifiers for the retuned interventions are shortened to
`byt5_base_genre_lr1e4` and `byt5_base_retrieval_lr1e4`. The
`config_snapshot.yaml` in each directory names the config it used.

### 1.2B parameter arm

| Condition | Configs | Artifacts |
|---|---|---|
| ByT5-large at 5e-5 | `C/byt5_large_baseline_lr5e5.yaml`, `C/byt5_large_baseline_seed{52,62}.yaml` | `M/byt5_large_baseline_lr5e5`, `M/byt5_large_baseline_seed{52,62}` |
| ByT5-large at 7e-5, single run | `C/byt5_large_baseline_lr7e5.yaml` | `M/byt5_large_baseline_lr7e5` |
| Dictionary glosses at 1.2B | `C/byt5_large_rq3_dictionary_gloss_seed{42,52,62}.yaml` | `M/byt5_large_rq3_dictionary_gloss_seed{42,52,62}` |

### Retrieval format ablation

Single-seed runs in the earlier environment, reported in the appendix.

- configs: `C/byt5_{small,base}_cw4_retrieval_{augmented,compact,source_only}.yaml`
- artifacts: `A/` directories of the same names, compared against
  `A/byt5_{small,base}_baseline`

### Environment comparison and sentence-level bootstrap

The paper compares the two environments over fourteen seed-matched pairs, the
baseline and the six interventions at both model sizes. The `M/*_seed42r` runs
and `M/byt5_small_baseline_regression42` are one side of each pair. The
directories in `A/` named without the `_seed42r` suffix are the other, with
`A/byt5_small_baseline` paired against `M/byt5_small_baseline_regression42`.

`M/byt5_base_baseline_envr2` and `M/byt5_small_baseline_envr2` are further
seed-42 reruns of the two ByT5 baseline configs in the later environment. The
580M one is the shared-recipe reference for two of the bootstrap comparisons
below.

`experiments/scripts/paired_bootstrap.py` resamples independent-test sentences
for eight run pairs and prints a 95% interval on the combined-score difference.
The pairs it uses are listed at the bottom of the script. It reads the
`independent_preds.csv` files stored beside the metrics and takes the reference
translations, row-aligned, from the locally rebuilt
`data/processed/independent_test_set_clean.csv`, because the prediction files
carry no corpus text.

### Qualitative examples

`experiments/results/qualitative_examples.csv`, regenerated with
`experiments/scripts/select_qualitative_examples.py`.

## Commands

```bash
# Fine-tune one condition
uv run akkadian-mt train seq2seq --config experiments/configs/<config>.yaml

# Score a checkpoint on both held-out sets and write a run artifact bundle
uv run python experiments/scripts/run_eval.py \
  --run-id <run_id> \
  --config experiments/configs/<config>.yaml \
  --checkpoint <train.output_dir from the config>/best_model.pt

# Score a checkpoint on one file without writing an artifact
uv run akkadian-mt eval lb \
  --checkpoint <path>/best_model.pt \
  --test-file data/processed/independent_test_set_clean.csv

# TF-IDF retrieval baseline
uv run python experiments/scripts/run_tfidf_baseline.py

# From-scratch tokenisation contrast (one run per invocation).
# The script writes its artifact under experiments/results/artifacts/;
# the stored scratch_* runs were relocated to the later pool.
uv run python experiments/scripts/train_scratch_tokenisation.py \
  --tokenisation sp --seed 42 --run-id scratch_sp4k_fixed_seed42

# Append a run to the earlier-pool registry
uv run python experiments/scripts/collect_results.py \
  --run-id <run_id> --model-family seq2seq --model-size <size> \
  --seed <seed> --ablation <ablation> \
  --config-path experiments/configs/<config>.yaml \
  --checkpoint-path <train.output_dir from the config>/best_model.pt
```

## Analysis and figures

These read the stored artifacts and need no GPU and no corpus.

```bash
# Seed-averaged numbers quoted in the paper, plus LaTeX table rows
uv run python experiments/results/matrix_h100/make_paper_numbers.py

# Capacity by preprocessing interaction, per metric and per seed
uv run python experiments/results/matrix_h100/interaction_check.py

# Paired bootstrap over test sentences. 2,000 resamples per pair, so it is slow.
uv run python experiments/scripts/paired_bootstrap.py

# Figures, written to paper/figures/ (created if missing).
uv run python experiments/results/matrix_h100/make_figures.py
```

## Cost

The full multi-seed matrix completed in under 24 hours on one 80 GB GPU with BF16
precision and gradient checkpointing.

## What this release cannot rebuild

The build scripts above start from three corpus CSVs that a separate assembly
step produced, and that step is not part of this release. Those files are
`data/processed/golden_corpus_v2_prefilt.csv`,
`data/processed/independent_test_set_clean.csv`, and
`data/processed/new_test_set.csv`. [DATA_ACCESS.md](DATA_ACCESS.md) describes
their contents and shape so they can be reconstructed from the raw Kaggle files.
The two holdout document-ID lists they depend on are included under
`data/processed/`.
