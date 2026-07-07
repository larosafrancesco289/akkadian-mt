"""Seed-averaged capacity x preprocessing interaction check.

For each intervention and model size, compute delta BLEU vs the same-size
baseline, per seed (paired) and seed-averaged. The coursework claim (sign
test p~=0.016 at seed 42 only): interventions help small, hurt base, i.e.
interaction = delta_small - delta_base > 0 for all 6 interventions.
"""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent / "artifacts"

INTERVENTIONS = [
    "rq1_no_determinatives",
    "rq2_tag_sumerograms",
    "rq3_dictionary_gloss",
    "cw4_genre_conditioned",
    "cw4_numeral_normalized",
    "cw4_retrieval_augmented",
]
SIZES = ["small", "base"]
SEEDS = ["42", "52", "62"]


def run_id(size, config, seed):
    base = f"byt5_{size}_{config}"
    if seed == "42":
        # seed-42 rows: reruns are *_seed42r; small baseline's is regression42
        if size == "small" and config == "baseline":
            return f"{base}_regression42"
        return f"{base}_seed42r"
    return f"{base}_seed{seed}"


def metric(size, config, seed, key="bleu", split="independent"):
    with open(ROOT / run_id(size, config, seed) / f"{split}_metrics.json") as f:
        return json.load(f)[key]


for key in ["bleu", "chrf"]:
    print(f"\n{'='*78}\n{key.upper()} on independent test set\n{'='*78}")
    base_means = {}
    print("\nBaselines (per seed / mean +- sd):")
    for size in SIZES:
        vals = [metric(size, "baseline", s, key) for s in SEEDS]
        base_means[size] = mean(vals)
        print(f"  {size:5s}: " + "  ".join(f"s{s}={v:6.2f}" for s, v in zip(SEEDS, vals))
              + f"   mean={mean(vals):6.2f} +- {stdev(vals):.2f}")

    print(f"\n{'intervention':26s} {'d_small(42/52/62)':>22s} {'avg':>7s} "
          f"{'d_base(42/52/62)':>22s} {'avg':>7s} {'inter':>7s}")
    n_pos_interaction = 0
    n_small_pos = 0
    n_base_neg = 0
    for cfg in INTERVENTIONS:
        deltas = {}
        for size in SIZES:
            per_seed = [metric(size, cfg, s, key) - metric(size, "baseline", s, key)
                        for s in SEEDS]
            deltas[size] = per_seed
        avg_s = mean(deltas["small"])
        avg_b = mean(deltas["base"])
        inter = avg_s - avg_b
        n_pos_interaction += inter > 0
        n_small_pos += avg_s > 0
        n_base_neg += avg_b < 0
        def fmt(xs):
            return "/".join(f"{x:+5.2f}" for x in xs)
        print(f"{cfg:26s} {fmt(deltas['small']):>22s} {avg_s:+7.2f} "
              f"{fmt(deltas['base']):>22s} {avg_b:+7.2f} {inter:+7.2f}")

    print(f"\nSeed-averaged sign summary ({key}):")
    print(f"  interaction (d_small - d_base) > 0 : {n_pos_interaction}/6")
    print(f"  d_small > 0 (helps small)          : {n_small_pos}/6")
    print(f"  d_base  < 0 (hurts base)           : {n_base_neg}/6")
    p_one = 0.5 ** 6
    if n_pos_interaction == 6:
        print(f"  unanimous -> one-sided sign test p = {p_one:.4f}")
