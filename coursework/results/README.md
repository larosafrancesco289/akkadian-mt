# Coursework Results

Tracked experiment results used by the report and the reproducibility guide.

## Files

- `results.csv`: flat run registry (all reported runs)
- `experiment_matrix.csv`: planned experiment matrix and run notes
- `loss_curve_runs.csv`: curated manifest for training-loss figures
- `qualitative_examples.csv`: qualitative example table used in the report

## Per-Run Artifacts

Each run directory under `artifacts/<run_id>/` contains:

- `summary.json`: source of truth for reported metrics
- `independent_metrics.json`: primary holdout metrics
- `newtest_metrics.json`: secondary holdout metrics
- `config_snapshot.*`: exact config used for evaluation
