"""Regenerate paper/figures/{interaction,hierarchy,scaling,tuning}.pdf from
the seed-averaged matrix results. Values come from make_paper_numbers.py
output; style matches the first-round figures (serif, red/blue dumbbell,
blue bars).
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
    "font.size": 14,
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
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
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
    ax.axvspan(-2.4, 0, color="#f2f2f2", zorder=0)
    # dotted rule between the philological (rows 1-3) and error-driven (4-6) families
    ax.axhline(len(ROWS) - 2.5, color="#aaaaaa", lw=0.9, ls=":", zorder=1)
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


def scale_runs(pct):
    """Same-environment multi-seed subset runs (wave 3, r2 rebuild)."""
    runs = [f"byt5_base_scale_{pct}pct_r2", f"byt5_base_scale_{pct}pct_seed52",
            f"byt5_base_scale_{pct}pct_seed62"]
    vals = []
    for run in runs:
        with open(ROOT / run / "independent_metrics.json") as f:
            vals.append(json.load(f)["combined"])
    return mean(vals), stdev(vals)


def scaling_figure():
    pcts = [25, 50, 75, 100]
    stats = [scale_runs(p) for p in (25, 50, 75)]
    stats.append((mean([combined("base", "baseline", s) for s in SEEDS]),
                  stdev([combined("base", "baseline", s) for s in SEEDS])))
    scores = [m for m, _ in stats]
    sds = [s for _, s in stats]
    ext_mean, ext_sd = 35.82, 1.62  # over seeds {42, 52, 62}, Table tab:variance
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.errorbar(pcts, scores, yerr=sds, color=BLUE, marker="o", ms=7, lw=1.8,
                capsize=4, elinewidth=1.2, zorder=3)
    for x, v in zip(pcts, scores):
        dy = -16 if x == 100 else 9
        ax.annotate(f"{v:.1f}", (x, v), xytext=(0, dy), textcoords="offset points",
                    ha="center", color=BLUE, fontsize=12)
    # External mix occupies its own separated slot: it is out-of-domain data,
    # not a point on the gold-corpus percentage axis.
    ax.axvline(106, color="#aaaaaa", lw=1.0, ls="--", zorder=1)
    ax.errorbar([112], [ext_mean], yerr=[ext_sd], color=RED, fmt="D", ms=7,
                capsize=4, elinewidth=1.2, zorder=3)
    ax.annotate(f"{ext_mean:.1f}", (112, ext_mean), xytext=(0, 10),
                textcoords="offset points", ha="center", color=RED, fontsize=12)
    ax.set_xticks(pcts + [112], [f"{p}%" for p in pcts] + ["+ext.\nmix"])
    ax.set_xlim(15, 120)
    ax.set_ylim(14, 42)
    ax.set_xlabel("Training data (% of gold corpus; external mix separate)")
    ax.set_ylabel("Combined score")
    ax.grid(axis="y", color="#dddddd", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "scaling.pdf")
    plt.close(fig)


EARLIER = HERE.parent / "artifacts"  # first-round single runs (mT5, TF-IDF)
LIGHT = "#93b5d2"  # 5e-5 probe points


def run_combined(run, root=ROOT):
    with open(root / run / "independent_metrics.json") as f:
        return json.load(f)["combined"]


def run_stats(runs, root=ROOT):
    vals = [run_combined(r, root) for r in runs]
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def baseline_stats(size):
    vals = [combined(size, "baseline", s) for s in SEEDS]
    return mean(vals), stdev(vals)


def shared_stats(size, cfg):
    vals = [combined(size, cfg, s) for s in SEEDS]
    return mean(vals), stdev(vals)


def tuning_figure():
    """Two-panel dumbbell: (a) family/size scores shared-recipe vs own rate,
    (b) the 580M retuning probe at 5e-5 / 7e-5 / 1e-4."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.0, 6.6), gridspec_kw={"height_ratios": [4.0, 3.2]})

    def point(ax, x, y, sd, style):
        kw = dict(fmt="o", ms=8, capsize=3, elinewidth=1.2, zorder=3)
        if style == "open":
            ax.errorbar([x], [y], xerr=[sd], color=BLUE, mfc="white", **kw)
        elif style == "filled":
            ax.errorbar([x], [y], xerr=[sd], color=BLUE, **kw)
        else:  # probe (5e-5)
            ax.errorbar([x], [y], xerr=[sd], color=LIGHT, **kw)

    def label(ax, x, y, dx=0, ha="center"):
        ax.annotate(f"{x:.2f}", (x, y), xytext=(dx, 11),
                    textcoords="offset points", ha=ha, color="#333333",
                    fontsize=11)

    # -- panel (a): shared recipe vs per-configuration rate ---------------
    rows_a = [  # (label, shared (m, sd), tuned (m, sd) or None)
        ("ByT5-base\n(580M)", baseline_stats("base"),
         run_stats(["byt5_base_baseline_lr1e4", "byt5_base_baseline_lr1e4_seed52",
                    "byt5_base_baseline_lr1e4_seed62"])),
        ("ByT5-small\n(300M)", baseline_stats("small"), None),
        ("mT5-base\n(580M)", (run_combined("mt5_base_baseline", EARLIER), 0.0),
         run_stats(["mt5_base_lr1e3", "mt5_base_lr1e3_seed52", "mt5_base_lr1e3_seed62"])),
        ("mT5-small\n(300M)", (run_combined("mt5_small_baseline", EARLIER), 0.0),
         run_stats(["mt5_small_lr1e3", "mt5_small_lr1e3_seed52", "mt5_small_lr1e3_seed62"])),
    ]
    ys = range(len(rows_a), 0, -1)
    for y, (_, (ms, ss), tuned) in zip(ys, rows_a):
        if tuned is not None:
            mt, st = tuned
            if mt - ms > 2:  # arrow only when there is room for one
                ax1.annotate("", (mt, y), (ms, y), zorder=1,
                             arrowprops=dict(arrowstyle="-|>", color="#999999",
                                             lw=1.4, shrinkA=6, shrinkB=6))
                label(ax1, ms, y)
                label(ax1, mt, y)
            else:
                label(ax1, ms, y, dx=-4, ha="right")
                label(ax1, mt, y, dx=4, ha="left")
            point(ax1, mt, y, st, "filled")
        else:
            label(ax1, ms, y)
        point(ax1, ms, y, ss, "open")
    tfidf = run_combined("tfidf_baseline", EARLIER)
    ax1.axvline(tfidf, color="#aaaaaa", lw=1.0, ls="--", zorder=1)
    ax1.text(tfidf, 0.52, f"TF-IDF 1-NN ({tfidf:.2f})", ha="center",
             color="#777777", fontsize=10)
    ax1.set_yticks(list(ys), [r[0] for r in rows_a])
    ax1.set_xlim(9, 40.5)
    ax1.set_ylim(0.35, len(rows_a) + 0.75)
    ax1.set_title("(a) Model family and size", loc="left", fontsize=13)
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", mfc="white", ls="",
                   ms=8, label="shared recipe"),
        plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=8,
                   label="own tuned rate"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=11, frameon=True,
               framealpha=1.0, edgecolor="#cccccc")

    # -- panel (b): the 580M retuning probe -------------------------------
    rows_b = [  # (label, {rate: (m, sd)})
        ("Baseline", {
            "7e-5": baseline_stats("base"),
            "1e-4": run_stats(["byt5_base_baseline_lr1e4",
                               "byt5_base_baseline_lr1e4_seed52",
                               "byt5_base_baseline_lr1e4_seed62"])}),
        ("Genre\nconditioning", {
            "5e-5": run_stats(["byt5_base_genre_lr5e5"]),
            "7e-5": shared_stats("base", "cw4_genre_conditioned"),
            "1e-4": run_stats(["byt5_base_genre_lr1e4",
                               "byt5_base_cw4_genre_conditioned_lr1e4_seed52",
                               "byt5_base_cw4_genre_conditioned_lr1e4_seed62"])}),
        ("Retrieval\naugmentation", {
            "5e-5": run_stats(["byt5_base_retrieval_lr5e5"]),
            "7e-5": shared_stats("base", "cw4_retrieval_augmented"),
            "1e-4": run_stats(["byt5_base_retrieval_lr1e4"])}),
    ]
    styles = {"5e-5": "probe", "7e-5": "open", "1e-4": "filled"}
    ys = range(len(rows_b), 0, -1)
    for y, (_, rates) in zip(ys, rows_b):
        xs = sorted(m for m, _ in rates.values())
        ax2.plot(xs, [y] * len(xs), color="#999999", lw=1.4, zorder=1)
        for rate, (m, sd) in rates.items():
            point(ax2, m, y, sd, styles[rate])
            label(ax2, m, y)
    ax2.set_yticks(list(ys), [r[0] for r in rows_b])
    ax2.set_xlim(32.6, 38.9)
    ax2.set_ylim(0.5, len(rows_b) + 0.7)
    ax2.set_xlabel("Combined score (independent test)")
    ax2.set_title("(b) Retuning at 580M", loc="left", fontsize=13)
    handles = [
        plt.Line2D([], [], color=LIGHT, marker="o", ls="", ms=8, label="5e-5"),
        plt.Line2D([], [], color=BLUE, marker="o", mfc="white", ls="", ms=8,
                   label="7e-5 (shared)"),
        plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=8, label="1e-4"),
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=11, frameon=True,
               framealpha=1.0, edgecolor="#cccccc")

    for ax in (ax1, ax2):
        ax.grid(axis="x", color="#dddddd", lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "tuning.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # hierarchy_figure() retired Aug 3: Figure 3 cut from the paper with the
    # optimisation-sensitivity reframe (council round 2).
    interaction_figure()
    scaling_figure()
    tuning_figure()
    print(f"wrote {FIGDIR}/interaction.pdf, scaling.pdf, and tuning.pdf")
