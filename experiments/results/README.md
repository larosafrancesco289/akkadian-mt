# Results

Tracked experiment results used by the paper and the reproducibility guide.

## Files

- `results.csv`: flat run registry (all reported runs)
- `experiment_matrix.csv`: the experiment matrix as planned before the runs;
  its status column reflects planning time, `results.csv` records what ran
- `loss_curve_runs.csv`: curated manifest for training-loss figures
- `qualitative_examples.csv`: qualitative example table used in the paper
- `loss_histories/`: per-epoch training and validation curves for the runs in
  `loss_curve_runs.csv`

## Per-Run Artifacts

Each run directory under `artifacts/<run_id>/` contains:

- `summary.json`: source of truth for reported metrics
- `independent_metrics.json`: primary holdout metrics
- `newtest_metrics.json`: secondary holdout metrics
- `config_snapshot.*`: exact config used for evaluation (absent for the six
  from-scratch `scratch_*` runs)
- `independent_preds.csv`, `newtest_preds.csv`: one row per test sentence with
  row metadata and the system prediction. Source and reference text are not
  included, since the corpus is not redistributed; rows are aligned with the
  locally rebuilt test sets (see `DATA_ACCESS.md`)
