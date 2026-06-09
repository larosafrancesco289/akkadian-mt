from __future__ import annotations

from pathlib import Path

import pytest

from akkadian_mt.config import Config, apply_cli_overrides, load_config


def test_apply_cli_overrides_handles_optional_none_fields() -> None:
    cfg = Config()

    apply_cli_overrides(
        cfg,
        [
            "--train.wandb_run_name",
            "exp-1",
            "--data.prepared_file",
            "data/processed/all_pairs.csv",
        ],
    )

    assert cfg.train.wandb_run_name == "exp-1"
    assert cfg.data.prepared_file == "data/processed/all_pairs.csv"


def test_apply_cli_overrides_supports_explicit_null() -> None:
    cfg = Config()
    cfg.train.wandb_run_name = "debug-run"

    apply_cli_overrides(cfg, ["--train.wandb_run_name", "null"])

    assert cfg.train.wandb_run_name is None


def test_load_config_rejects_non_mapping_payload(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level must be a mapping"):
        load_config(bad_config)
