"""Emit every seed-averaged number the paper quotes, plus LaTeX table rows.

Companion to interaction_check.py; reads the same artifact JSONs. Values that
stay in the earlier environment (mT5, TF-IDF, scaling, external
mix, retrieval-format ablation) are not computed here.
"""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent / "artifacts"

LABELS = {
    "rq1_no_determinatives": "1.\\ Strip determinatives",
    "rq2_tag_sumerograms": "2.\\ Tag Sumerograms",
    "rq3_dictionary_gloss": "3.\\ Dictionary glosses",
    "cw4_genre_conditioned": "5.\\ Genre conditioning",
    "cw4_numeral_normalized": "6.\\ Numeral normalisation",
    "cw4_retrieval_augmented": "7.\\ Retrieval augmentation",
}
SEEDS = ["42", "52", "62"]


def rid(size, cfg, seed):
    b = f"byt5_{size}_{cfg}"
    if seed == "42":
        return f"{b}_regression42" if (size, cfg) == ("small", "baseline") else f"{b}_seed42r"
    return f"{b}_seed{seed}"


def metric(size, cfg, seed, key, split):
    with open(ROOT / rid(size, cfg, seed) / f"{split}_metrics.json") as f:
        return json.load(f)[key]


def series(size, cfg, key="combined", split="independent"):
    return [metric(size, cfg, s, key, split) for s in SEEDS]


def deltas(size, cfg, key="combined", split="independent"):
    return [metric(size, cfg, s, key, split) - metric(size, "baseline", s, key, split)
            for s in SEEDS]


print("== Baselines (mean +- sd over seeds 42/52/62) ==")
for split in ["independent", "newtest"]:
    for size in ["small", "base"]:
        parts = [f"{k}={mean(series(size, 'baseline', k, split)):.2f}+-{stdev(series(size, 'baseline', k, split)):.2f}"
                 for k in ["bleu", "chrf", "combined"]]
        print(f"  {split:11s} {size:5s}: " + "  ".join(parts))

print("\n== Table 4 rows (combined; mean +- sd; deltas paired per seed) ==")
b_s = series("small", "baseline")
b_b = series("base", "baseline")
print(f"Baseline & ${mean(b_s):.2f} \\pm {stdev(b_s):.2f}$ & --- & "
      f"${mean(b_b):.2f} \\pm {stdev(b_b):.2f}$ & --- & --- & --- \\\\")
sum_ds, sum_db = [], []
for cfg, label in LABELS.items():
    s, b = series("small", cfg), series("base", cfg)
    ds, db = deltas("small", cfg), deltas("base", cfg)
    ns, nb = deltas("small", cfg, split="newtest"), deltas("base", cfg, split="newtest")
    sum_ds.append(mean(ds))
    sum_db.append(mean(db))
    print(f"{label} & ${mean(s):.2f} \\pm {stdev(s):.2f}$ & ${mean(ds):+.2f}$ & "
          f"${mean(b):.2f} \\pm {stdev(b):.2f}$ & ${mean(db):+.2f}$ & "
          f"${mean(ns):+.2f}$ & ${mean(nb):+.2f}$ \\\\")
print(f"Mean (excl. gaps) & & ${mean(sum_ds):+.2f}$ & & ${mean(sum_db):+.2f}$ & & \\\\")

print("\n== Table 6 rows (per-seed combined) ==")
for size in ["small", "base"]:
    ind, new = series(size, "baseline"), series(size, "baseline", split="newtest")
    for i, s in enumerate(SEEDS):
        print(f"  {size} seed {s}: {ind[i]:.2f} & {new[i]:.2f}")
    print(f"  {size} mu+-sd: {mean(ind):.2f}+-{stdev(ind):.2f} & {mean(new):.2f}+-{stdev(new):.2f}")

print("\n== Prose numbers ==")
print(f"capacity gap byte-level (base-small, combined): {mean(b_b) - mean(b_s):.2f}")
print(f"tokenisation gap 300M (vs frozen mT5-small 11.58): {mean(b_s) - 11.58:.2f}")
print(f"tokenisation gap 580M (vs frozen mT5-base 15.33): {mean(b_b) - 15.33:.2f}")
print(f"tokenisation gap avg: {((mean(b_s) - 11.58) + (mean(b_b) - 15.33)) / 2:.2f}")
print(f"mean |base delta| (hierarchy preprocessing bar): {mean(abs(d) for d in sum_db):.2f}")
print("BLEU/chrF for Table 3 ByT5 rows:")
for size in ["small", "base"]:
    print(f"  {size}: BLEU {mean(series(size,'baseline','bleu')):.2f} "
          f"chrF {mean(series(size,'baseline','chrf')):.2f} "
          f"combined {mean(series(size,'baseline','combined')):.2f}")

print("\n== Sign counts (seed-averaged) ==")
for split in ["independent", "newtest"]:
    inter = [mean(deltas("small", c, split=split)) - mean(deltas("base", c, split=split))
             for c in LABELS]
    ds = [mean(deltas("small", c, split=split)) for c in LABELS]
    db = [mean(deltas("base", c, split=split)) for c in LABELS]
    print(f"  {split}: inter>0 {sum(i > 0 for i in inter)}/6, small>0 {sum(d > 0 for d in ds)}/6, "
          f"base<0 {sum(d < 0 for d in db)}/6, min|inter| {min(abs(i) for i in inter):.2f}")

print("\n== Figure data (independent, combined) ==")
for cfg, label in LABELS.items():
    ds, db = deltas("small", cfg), deltas("base", cfg)
    print(f"  {label.split(chr(92))[0]:3s}{cfg:26s} small {mean(ds):+.2f} sd {stdev(ds):.2f}   "
          f"base {mean(db):+.2f} sd {stdev(db):.2f}")
