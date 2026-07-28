# Reproducibility guide

This document maps each section of the [paper](paper/paper.pdf) to its tracked
experiment artifacts and lists the commands to rerun every experiment.

## Prerequisites

```bash
uv sync --extra dev
```

Obtain the corpora as described in [DATA_ACCESS.md](DATA_ACCESS.md) and place the
raw files under `data/raw/`. Then build the derived datasets:

```bash
uv run python experiments/scripts/build_holdout_union.py
uv run python experiments/scripts/build_variants.py
```

The secondary external-data experiments additionally require a prepared external
mix CSV — see [DATA_ACCESS.md](DATA_ACCESS.md).

## Tracked artifacts

Per-run artifacts live under `experiments/results/artifacts/<run_id>/` and include
`summary.json`, `independent_metrics.json`, `newtest_metrics.json`, and
`config_snapshot.yaml`. Aggregate tables are in `experiments/results/`:
`results.csv`, `experiment_matrix.csv`, `qualitative_examples.csv`.

## Report-to-artifact map

Each subsection lists the run IDs whose metrics appear in the corresponding
report table or figure.

### Tokenisation × capacity factorial
`tfidf_baseline`, `mt5_small_baseline`, `mt5_base_baseline`,
`byt5_small_baseline`, `byt5_base_baseline`

### Preprocessing intervention table
`byt5_small_rq1_no_determinatives`, `byt5_small_rq2_tag_sumerograms`,
`byt5_small_rq3_dictionary_gloss`, `byt5_small_rq4_remove_gaps`,
`byt5_base_rq1_no_determinatives`, `byt5_base_rq2_tag_sumerograms`,
`byt5_base_rq3_dictionary_gloss`, `byt5_base_rq4_remove_gaps`,
`byt5_small_cw4_genre_conditioned`, `byt5_base_cw4_genre_conditioned`,
`byt5_small_cw4_numeral_normalized`, `byt5_base_cw4_numeral_normalized`,
`byt5_small_cw4_retrieval_augmented`, `byt5_base_cw4_retrieval_augmented`

### Retrieval format ablation
`byt5_small_baseline`, `byt5_base_baseline`,
`byt5_small_cw4_retrieval_augmented`, `byt5_base_cw4_retrieval_augmented`,
`byt5_small_cw4_retrieval_compact`, `byt5_base_cw4_retrieval_compact`,
`byt5_small_cw4_retrieval_source_only`, `byt5_base_cw4_retrieval_source_only`

### Data scaling and external-data checks
`byt5_base_scale_25pct`, `byt5_base_scale_50pct`, `byt5_base_scale_75pct`,
`byt5_base_baseline`, `byt5_base_external_mix`, `byt5_small_external_mix`

### Variance estimation
`byt5_base_baseline`, `byt5_base_seed52`, `byt5_base_seed62`,
`byt5_base_external_mix`, `byt5_base_external_mix_seed52`,
`byt5_base_external_mix_seed62`

### Qualitative analysis
Source table: `experiments/results/qualitative_examples.csv`
(regenerate with `experiments/scripts/select_qualitative_examples.py`).

## Command reference

```bash
# Train (one YAML per condition under experiments/configs/)
uv run akkadian-mt train seq2seq --config experiments/configs/<config>.yaml

# Evaluate on the held-out test set
uv run akkadian-mt eval lb \
  --config experiments/configs/<config>.yaml \
  --checkpoint outputs/coursework/<run_id>/best_model.pt

# TF-IDF retrieval baseline
uv run python experiments/scripts/run_tfidf_baseline.py

# Collect per-run results into the registry
uv run python experiments/scripts/collect_results.py \
  --run-id <run_id> --model-family seq2seq --model-size <size> \
  --seed <seed> --ablation <ablation> \
  --config-path experiments/configs/<config>.yaml \
  --checkpoint-path outputs/coursework/<run_id>/best_model.pt

# Regenerate paper figures
uv run python experiments/report/make_figures.py

# Regenerate grouped training-loss curves from rerun histories
uv run akkadian-mt report plot-loss-curves
```

## A note on variance

Both ByT5 baselines and all six non-trivial preprocessing interventions were
run with seeds {42, 52, 62} at both model sizes in a single training
environment (`experiments/results/matrix_h100/`), as was the external-data
condition. The paper's central claim rests on the unanimous *direction* of the
capacity × preprocessing effect across the six seed-averaged deltas (sign
test, *p* ≈ 0.016), with the direction replicating for five of six
interventions on the second test set. The mT5 comparison, TF-IDF baseline,
data-scaling, and retrieval-format conditions remain single runs. Rerun
`experiments/results/matrix_h100/interaction_check.py` to reproduce the
seed-averaged analysis.
