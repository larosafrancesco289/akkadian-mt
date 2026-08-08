# Env-consistent multi-seed matrix (H100, July 2026)

Results of the full intervention matrix rerun in a single environment: 14
configs (baseline + 6 interventions, ByT5-small and ByT5-base) × 3 seeds
(42, 52, 62), trained to convergence on a spot H100 (torch/CUDA environment
of July 2026) and evaluated on the frozen held-out splits.

Run-id conventions:

- `*_seed52` / `*_seed62` — the multi-seed configs in `experiments/configs/`.
- `*_seed42r` — seed-42 rerun of the original config in the new
  environment, so every number in this table comes from one environment. The
  original single-seed results in `../artifacts/` are unchanged provenance.
- `byt5_small_baseline_regression42` — the seed-42 regression run of the small
  baseline; serves as that config's seed-42 row.

Each run directory contains `summary.json`, `independent_metrics.json`,
`newtest_metrics.json`, `training_history.csv`, and `config_snapshot.yaml`.
Prediction CSVs and model checkpoints are archived externally and omitted here
for size. `interaction_check.py` reproduces the seed-averaged
capacity × preprocessing interaction analysis from these files.

Like `../artifacts/`, metric JSON and `config_snapshot.yaml` here are frozen
provenance — do not edit.
