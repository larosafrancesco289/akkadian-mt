"""Run pure-model evaluation on the two independent test files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from akkadian_mt.config import DataConfig, load_config
from akkadian_mt.data.preprocessing import normalize_inference_transliterations
from akkadian_mt.evaluation.evaluate_lb import (
    SentenceDataset,
    evaluate_model,
    load_checkpoint_model,
    load_exported_model,
)


def _load_model(
    *,
    checkpoint: str | None,
    model_dir: str | None,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], str]:
    if checkpoint:
        model, tokenizer, meta = load_checkpoint_model(checkpoint, device)
        return model, tokenizer, meta, checkpoint
    if model_dir:
        model, tokenizer, meta = load_exported_model(model_dir, device)
        return model, tokenizer, meta, model_dir
    raise ValueError("Provide either --checkpoint or --model-dir")


def _resolve_preprocessing_config(
    *,
    config_path: str | None,
    meta: dict[str, Any],
) -> DataConfig:
    if config_path:
        return load_config(config_path).data

    return DataConfig(
        data_dir=str(meta.get("data_dir", "data/raw")),
        source_prefix=str(meta.get("source_prefix", "")),
        use_dictionary_gloss=bool(meta.get("use_dictionary_gloss", False)),
        max_glosses=int(meta.get("max_glosses", 8)),
        strip_determinatives=bool(meta.get("strip_determinatives", False)),
        tag_sumerograms=bool(meta.get("tag_sumerograms", False)),
        remove_gaps=bool(meta.get("remove_gaps", False)),
        strip_homophone_subscripts=bool(meta.get("strip_homophone_subscripts", True)),
        skip_source_normalization=bool(meta.get("skip_source_normalization", False)),
    )


def _resolve_test_file(
    cli_path: str | None,
    config_path: str | None,
    fallback: str,
) -> Path:
    if cli_path:
        return Path(cli_path)
    if config_path:
        return Path(config_path)
    return Path(fallback)


def _evaluate_split(
    *,
    test_file: Path,
    model: Any,
    tokenizer: Any,
    data_config: DataConfig,
    device: torch.device,
    num_beams: int,
    length_penalty: float,
    batch_size: int,
    no_repeat: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    test_df = pd.read_csv(test_file)
    if data_config.skip_source_normalization:
        texts = test_df["transliteration"].astype(str).tolist()
    else:
        texts = normalize_inference_transliterations(
            test_df["transliteration"].astype(str).tolist(),
            data_dir=data_config.data_dir,
            remove_gaps=data_config.remove_gaps,
            strip_determinatives=data_config.strip_determinatives,
            tag_sumerograms=data_config.tag_sumerograms,
            strip_homophone_subscripts=data_config.strip_homophone_subscripts,
            use_dictionary_gloss=data_config.use_dictionary_gloss,
            max_glosses=data_config.max_glosses,
        )

    dataset = SentenceDataset(
        texts,
        test_df["translation"].astype(str).tolist(),
        prefix=data_config.source_prefix,
    )
    result = evaluate_model(
        model,
        tokenizer,
        dataset,
        device,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram=no_repeat,
        batch_size=batch_size,
        use_postprocessing=True,
    )
    pred_df = test_df.copy()
    pred_df["prediction"] = result["predictions"]
    metrics = {
        "bleu": float(result["bleu"]),
        "chrf": float(result["chrf"]),
        "combined": float(result["combined"]),
    }
    return metrics, pred_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument("--checkpoint", default=None, help="Training checkpoint to evaluate.")
    parser.add_argument("--model-dir", default=None, help="Exported model directory to evaluate.")
    parser.add_argument("--config", default=None, help="Optional config file to snapshot.")
    parser.add_argument(
        "--primary-test-file",
        default=None,
        help="Primary independent test file. Defaults to config override, then the standard held-out split.",
    )
    parser.add_argument(
        "--secondary-test-file",
        default=None,
        help="Secondary independent test file. Defaults to config override, then the standard held-out split.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/artifacts",
        help="Artifact root directory.",
    )
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=1.3)
    parser.add_argument("--no-repeat", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    artifact_dir = Path(args.output_dir) / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, meta, artifact_source = _load_model(
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        device=device,
    )
    data_config = _resolve_preprocessing_config(config_path=args.config, meta=meta)
    primary_test_file = _resolve_test_file(
        args.primary_test_file,
        data_config.eval_primary_test_file,
        "data/processed/independent_test_set_clean.csv",
    )
    secondary_test_file = _resolve_test_file(
        args.secondary_test_file,
        data_config.eval_secondary_test_file,
        "data/processed/new_test_set.csv",
    )

    if args.config:
        shutil.copyfile(args.config, artifact_dir / "config_snapshot.yaml")

    independent_metrics, independent_df = _evaluate_split(
        test_file=primary_test_file,
        model=model,
        tokenizer=tokenizer,
        data_config=data_config,
        device=device,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        batch_size=args.batch_size,
        no_repeat=args.no_repeat,
    )
    independent_df.to_csv(artifact_dir / "independent_preds.csv", index=False)

    newtest_metrics, newtest_df = _evaluate_split(
        test_file=secondary_test_file,
        model=model,
        tokenizer=tokenizer,
        data_config=data_config,
        device=device,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        batch_size=args.batch_size,
        no_repeat=args.no_repeat,
    )
    newtest_df.to_csv(artifact_dir / "newtest_preds.csv", index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "run_id": args.run_id,
        "artifact_source": artifact_source,
        "config_path": args.config,
        "primary_test_file": str(primary_test_file),
        "secondary_test_file": str(secondary_test_file),
        "source_prefix": data_config.source_prefix,
        "remove_gaps": data_config.remove_gaps,
        "strip_determinatives": data_config.strip_determinatives,
        "tag_sumerograms": data_config.tag_sumerograms,
        "strip_homophone_subscripts": data_config.strip_homophone_subscripts,
        "skip_source_normalization": data_config.skip_source_normalization,
        "use_dictionary_gloss": data_config.use_dictionary_gloss,
        "max_glosses": data_config.max_glosses,
        "independent": independent_metrics,
        "newtest": newtest_metrics,
    }
    (artifact_dir / "independent_metrics.json").write_text(
        json.dumps(independent_metrics, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "newtest_metrics.json").write_text(
        json.dumps(newtest_metrics, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
