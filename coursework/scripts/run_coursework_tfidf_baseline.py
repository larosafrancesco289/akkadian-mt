"""Run a char n-gram TF-IDF retrieval baseline for the coursework lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from akkadian_mt.config import DataConfig
from akkadian_mt.data.dataset import load_train_data
from akkadian_mt.data.preprocessing import normalize_inference_transliterations
from akkadian_mt.evaluation.metrics import compute_combined_score


def _predict(
    train_matrix,
    train_translations: list[str],
    eval_texts: list[str],
    vectorizer: TfidfVectorizer,
) -> list[str]:
    eval_matrix = vectorizer.transform(eval_texts)
    similarities = eval_matrix @ train_matrix.T
    best_indices = similarities.argmax(axis=1).A1
    return [train_translations[idx] for idx in best_indices]


def _run_one_split(
    train_matrix,
    train_translations: list[str],
    eval_path: Path,
    output_path: Path,
    vectorizer: TfidfVectorizer,
    data_config: DataConfig,
) -> dict[str, float]:
    eval_df = pd.read_csv(eval_path)
    eval_texts = normalize_inference_transliterations(
        eval_df["transliteration"].astype(str).tolist(),
        data_dir=data_config.data_dir,
        remove_gaps=data_config.remove_gaps,
        strip_determinatives=data_config.strip_determinatives,
        tag_sumerograms=data_config.tag_sumerograms,
        strip_homophone_subscripts=data_config.strip_homophone_subscripts,
        use_dictionary_gloss=data_config.use_dictionary_gloss,
        max_glosses=data_config.max_glosses,
    )
    predictions = _predict(train_matrix, train_translations, eval_texts, vectorizer)
    metrics = compute_combined_score(predictions, eval_df["translation"].astype(str).tolist())

    pred_df = eval_df.copy()
    pred_df["prediction"] = predictions
    pred_df.to_csv(output_path, index=False)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="tfidf_baseline")
    parser.add_argument(
        "--prepared-file",
        default="data/processed/golden_corpus_v2_prefilt.csv",
        help="Prepared training CSV.",
    )
    parser.add_argument(
        "--holdout-file",
        default="data/processed/coursework_holdout_doc_ids.json",
        help="Union holdout JSON file.",
    )
    parser.add_argument(
        "--primary-test-file",
        default="data/processed/independent_test_set_clean.csv",
        help="Primary external test file.",
    )
    parser.add_argument(
        "--secondary-test-file",
        default="data/processed/new_test_set.csv",
        help="Secondary external test file.",
    )
    parser.add_argument(
        "--output-dir",
        default="coursework/results/artifacts",
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum character n-gram size.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=5,
        help="Maximum character n-gram size.",
    )
    parser.add_argument(
        "--analyzer",
        choices=["char", "char_wb"],
        default="char",
        help="TF-IDF analyzer type.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        data_mode="prepared",
        prepared_file=args.prepared_file,
        val_split=0.1,
        val_split_strategy="doc_id",
        val_group_column="oare_id",
        max_source_length=512,
        max_target_length=384,
        source_prefix="translate Akkadian to English: ",
        filter_quality=True,
        min_translation_chars=10,
        max_length_ratio=10.0,
        strip_homophone_subscripts=True,
        holdout_doc_ids_file=args.holdout_file,
    )
    train_df, val_df = load_train_data(data_config)
    retrieval_df = train_df.reset_index(drop=True)
    retrieval_path = output_dir / "retrieval_corpus.csv"
    retrieval_df.to_csv(retrieval_path, index=False)

    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=False,
    )
    train_texts = retrieval_df["transliteration"].astype(str).tolist()
    train_translations = retrieval_df["translation"].astype(str).tolist()
    train_matrix = vectorizer.fit_transform(train_texts)

    independent_metrics = _run_one_split(
        train_matrix,
        train_translations,
        Path(args.primary_test_file),
        output_dir / "independent_preds.csv",
        vectorizer,
        data_config,
    )
    newtest_metrics = _run_one_split(
        train_matrix,
        train_translations,
        Path(args.secondary_test_file),
        output_dir / "newtest_preds.csv",
        vectorizer,
        data_config,
    )

    config_snapshot = {
        "run_id": args.run_id,
        "prepared_file": args.prepared_file,
        "holdout_file": args.holdout_file,
        "primary_test_file": args.primary_test_file,
        "secondary_test_file": args.secondary_test_file,
        "tfidf": {
            "analyzer": args.analyzer,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "lowercase": False,
        },
        "train_rows": int(len(retrieval_df)),
        "validation_rows": int(len(val_df)),
    }
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )
    summary = {
        "run_id": args.run_id,
        "artifact_type": "tfidf_retrieval",
        "prepared_file": args.prepared_file,
        "holdout_file": args.holdout_file,
        "retrieval_rows": int(len(retrieval_df)),
        "validation_rows": int(len(val_df)),
        "tfidf": {
            "analyzer": args.analyzer,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
        },
        "independent": independent_metrics,
        "newtest": newtest_metrics,
    }
    (output_dir / "independent_metrics.json").write_text(
        json.dumps(independent_metrics, indent=2),
        encoding="utf-8",
    )
    (output_dir / "newtest_metrics.json").write_text(
        json.dumps(newtest_metrics, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
