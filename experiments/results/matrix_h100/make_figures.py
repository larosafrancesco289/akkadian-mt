"""Regenerate paper/figures/{interaction,hierarchy}.pdf from the seed-averaged
matrix results. Values come from make_paper_numbers.py output; style matches
the original coursework figures (serif, red/blue dumbbell, blue bars).
"""
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "artifacts"
FIGDIR = HERE.parents[2] / "paper" / "figures"

BLUE = "#20659c"   # ByT5-small
RED = "#b03a2e"    # ByT5-base
SEEDS = ["42", "52", "62"]
ROWS = [  # (config, two-line label)
    ("rq1_no_determinatives", "Determinative\nstripping"),
    ("rq2_tag_sumerograms", "Sumerogram\ntagging"),
    ("rq3_dictionary_gloss", "Dictionary\nglosses"),
    ("cw4_genre_conditioned", "Genre\nconditioning"),
    ("cw4_numeral_normalized", "Numeral\nnormalisation"),
    ("cw4_retrieval_augmented", "Retrieval\naugmentation"),
]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "axes.edgecolor": "#888888",
    "font.size": 13,
})


def combined(size, cfg, seed):
    run = f"byt5_{size}_{cfg}"
    if seed == "42":
        run = f"{run}_regression42" if (size, cfg) == ("small", "baseline") else f"{run}_seed42r"
    else:
        run = f"{run}_seed{seed}"
    with open(ROOT / run / "independent_metrics.json") as f:
        return json.load(f)["combined"]


def delta_stats(size, cfg):
    ds = [combined(size, cfg, s) - combined(size, "baseline", s) for s in SEEDS]
    return mean(ds), stdev(ds)


def fmt(v):
    return "0.00" if abs(round(v, 2)) < 0.005 else f"{v:+.2f}"


def interaction_figure():
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ys = range(len(ROWS), 0, -1)
    for y, (cfg, _) in zip(ys, ROWS):
        (ms, ss), (mb, sb) = delta_stats("small", cfg), delta_stats("base", cfg)
        ax.plot([mb, ms], [y, y], color="#999999", lw=1.6, zorder=1)
        ax.errorbar([mb], [y], xerr=[sb], color=RED, fmt="o", ms=9, capsize=3,
                    elinewidth=1.2, zorder=3)
        ax.errorbar([ms], [y], xerr=[ss], color=BLUE, fmt="o", ms=9, capsize=3,
                    elinewidth=1.2, zorder=3)
        ax.annotate(fmt(mb), (mb, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", color=RED, fontsize=12)
        ax.annotate(fmt(ms), (ms, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", color=BLUE, fontsize=12)
    ax.axvline(0, color="#555555", lw=1.0, zorder=2)
    ax.set_yticks(list(ys), [label for _, label in ROWS])
    ax.set_xlim(-2.4, 1.8)
    ax.set_ylim(0.4, len(ROWS) + 0.9)
    ax.set_xlabel(r"$\Delta$ combined score vs. baseline")
    ax.text(-2.3, len(ROWS) + 0.62, "ByT5-base (580M)", color=RED, fontsize=13)
    ax.text(1.7, len(ROWS) + 0.62, "ByT5-small (300M)", color=BLUE, fontsize=13,
            ha="right")
    ax.grid(axis="x", color="#dddddd", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "interaction.pdf")
    plt.close(fig)


def hierarchy_figure():
    labels = ["Tokenisation", "Data quantity", "Model capacity", "Preprocessing"]
    values = [19.3, 18.1, 7.3, 0.7]
    colors = ["#25608e", "#25608e", "#7d9fbe", "#b9cddd"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ys = range(len(labels), 0, -1)
    ax.barh(list(ys), values, height=0.62, color=colors, zorder=2)
    for y, v in zip(ys, values):
        inside = v > 3
        ax.annotate(f"{v:.1f}", (v, y), xytext=(-8 if inside else 8, 0),
                    textcoords="offset points", va="center",
                    ha="right" if inside else "left",
                    color="white" if inside else "#333333", fontsize=13)
    ax.set_yticks(list(ys), labels)
    ax.set_xlim(0, 22)
    ax.set_xlabel("Effect size (combined score points)")
    ax.grid(axis="x", color="#dddddd", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hierarchy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    interaction_figure()
    hierarchy_figure()
    print(f"wrote {FIGDIR}/interaction.pdf and hierarchy.pdf")
