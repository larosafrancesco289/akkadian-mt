# Env-consistent multi-seed matrix (H100, July 2026)

The 88 runs of the later environment: the full intervention matrix (baseline +
6 interventions, ByT5-small and ByT5-base, seeds 42, 52, 62), the tuned mT5
grid, the 1e-4 retuning arm, the data-scaling grid, the 1.2B arm, the
environment-check reruns, and the from-scratch tokenisation pair. All trained
on a spot H100 (torch/CUDA environment of July 2026) and evaluated on the
frozen held-out splits.

Run-id conventions:

- `*_seed52` / `*_seed62` — the multi-seed configs in `experiments/configs/`.
- `*_seed42r` — seed-42 rerun of the original config in the new
  environment, so every number in this table comes from one environment. The
  original single-seed results in `../artifacts/` are unchanged provenance.
- `byt5_small_baseline_regression42` — the seed-42 regression run of the small
  baseline; serves as that config's seed-42 row.

Each run directory contains `summary.json`, `independent_metrics.json`, and
`newtest_metrics.json`; configured runs also carry `config_snapshot.yaml`, six
carry `training_history.csv`, and 46 carry prediction CSVs. Model checkpoints
are archived externally for size. `interaction_check.py` reproduces the
seed-averaged capacity × preprocessing interaction analysis from these files.

Like `../artifacts/`, metric JSON and `config_snapshot.yaml` here are frozen
provenance — do not edit.
