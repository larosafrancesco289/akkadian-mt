"""
Generate publication-quality figures for CW4 report.

Design language: Tufte-inspired editorial with a high data-ink ratio,
STIX Two Text for LaTeX harmony, restrained two-hue palette,
direct annotation over legends, no chartjunk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np

# Global style

# Palette: desaturated academic blue/red, tuned to print well in grayscale too
BLUE = "#2d5f8a"
RED = "#b5382a"
GREY = "#8c8c8c"
LIGHT_GREY = "#e8e8e8"
BG = "#fafafa"
INK = "#1a1a1a"

COLWIDTH = 3.35  # inches, ICML single-column width

plt.rcParams.update({
    # Typography: STIX Two matches the LaTeX body text
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Ink economy
    "axes.linewidth": 0.4,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # Clean background
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "figure.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
})

OUTDIR = "figures"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# Data

interventions = [
    "Determinative\nstripping",
    "Sumerogram\ntagging",
    "Dictionary\nglosses",
    "Genre\nconditioning",
    "Numeral\nnormalisation",
    "Retrieval\naugmentation",
]
delta_small = np.array([+0.60, -0.03, +0.96, +0.84, +0.50, +1.31])
delta_base = np.array([-0.30, -0.84, -1.10, -1.42, -0.68, -1.30])

scaling_pct = [25, 50, 75, 100]
scaling_score = [18.48, 28.23, 34.00, 36.58]
baseline_seeds = [36.58, 36.27, 36.57]
ext_seeds = [37.69, 34.90, 34.87]
ext_mean = np.mean(ext_seeds)
ext_std = np.std(ext_seeds, ddof=1)
baseline_std = np.std(baseline_seeds, ddof=1)

hierarchy_labels = ["Tokenisation", "Data quantity", "Model capacity", "Preprocessing"]
hierarchy_vals = [19.2, 18.1, 7.9, 0.9]


# Figure 1: Dumbbell / connected dot interaction plot

fig, ax = plt.subplots(figsize=(COLWIDTH, 3.0))

y = np.arange(len(interventions))
NOISE_BAND = 0.54  # 3σ

# Noise band
ax.axvspan(-NOISE_BAND, NOISE_BAND, color=LIGHT_GREY, zorder=0,
           linewidth=0)

# Zero line
ax.axvline(0, color=GREY, linewidth=0.5, zorder=1, linestyle="-")

# Connecting lines (dumbbell stems)
for i in range(len(interventions)):
    ax.plot([delta_base[i], delta_small[i]], [y[i], y[i]],
            color=GREY, linewidth=0.8, zorder=2, solid_capstyle="round")

# Dots
ax.scatter(delta_small, y, s=28, color=BLUE, zorder=4, edgecolors="white",
           linewidths=0.3)
ax.scatter(delta_base, y, s=28, color=RED, zorder=4, edgecolors="white",
           linewidths=0.3)

# Direct value labels on dots
outline = [pe.withStroke(linewidth=2, foreground="white")]
for i in range(len(interventions)):
    # Small model label (right of dot if positive, left if negative)
    s_val = delta_small[i]
    s_ha = "left" if s_val >= 0 else "right"
    s_off = 0.08 if s_val >= 0 else -0.08
    ax.text(s_val + s_off, y[i] + 0.02, f"{s_val:+.2f}",
            color=BLUE, fontsize=9, ha=s_ha, va="center",
            fontweight="medium", path_effects=outline)

    # Base model label: push further from the dot to avoid legend overlap
    b_val = delta_base[i]
    b_ha = "right" if b_val <= 0 else "left"
    b_off = -0.12 if b_val <= 0 else 0.12
    ax.text(b_val + b_off, y[i] + 0.02, f"{b_val:+.2f}",
            color=RED, fontsize=9, ha=b_ha, va="center",
            fontweight="medium", path_effects=outline)

ax.set_yticks(y)
ax.set_yticklabels(interventions, linespacing=0.85)
ax.set_xlabel(r"$\Delta$ combined score vs. baseline")
ax.invert_yaxis()
ax.set_xlim(-2.1, 2.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

# Minimal grid: x only, very faint
ax.grid(axis="x", linewidth=0.2, alpha=0.4, color=GREY)
ax.set_axisbelow(True)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Direct series labels on the first row instead of a legend box
ax.text(delta_small[0] + 0.12, y[0] - 0.38, "ByT5-small (300M)",
        color=BLUE, fontsize=9, va="center", ha="left",
        fontweight="medium", path_effects=outline)
ax.text(delta_base[0] - 0.12, y[0] - 0.38, "ByT5-base (580M)",
        color=RED, fontsize=9, va="center", ha="right",
        fontweight="medium", path_effects=outline)

# Noise band annotation
ax.text(NOISE_BAND + 0.03, len(interventions) - 0.65,
        r"$3\sigma$", color=GREY, fontsize=9, va="center",
        fontstyle="italic")

fig.savefig(f"{OUTDIR}/interaction.pdf")
plt.close(fig)
print("✓ interaction.pdf")


# Figure 2: Data scaling curve

fig, ax = plt.subplots(figsize=(COLWIDTH, 2.5))

# Main curve
ax.plot(scaling_pct, scaling_score, "-", color=BLUE, linewidth=1.0,
        zorder=3, solid_capstyle="round")
ax.scatter(scaling_pct, scaling_score, s=22, color=BLUE, zorder=4,
           edgecolors="white", linewidths=0.4)

# Baseline 100% error bar (multi-seed)
ax.errorbar(100, np.mean(baseline_seeds), yerr=baseline_std,
            fmt="none", ecolor=BLUE, capsize=2, linewidth=0.6,
            zorder=5, capthick=0.6)

# External data point
ax.errorbar(108, ext_mean, yerr=ext_std, fmt="none", ecolor=RED,
            capsize=2, linewidth=0.6, zorder=5, capthick=0.6)
ax.scatter([108], [ext_mean], s=28, color=RED, zorder=5,
           edgecolors="white", linewidths=0.4, marker="D")

# Direct annotations instead of legend
ax.annotate("+ external mix", xy=(108, ext_mean),
            xytext=(10, -2), textcoords="offset points",
            fontsize=9, color=RED, va="center",
            arrowprops=dict(arrowstyle="-", color=RED,
                            linewidth=0.4, shrinkA=0, shrinkB=3))

# Score labels on each point
outline = [pe.withStroke(linewidth=2, foreground="white")]
for pct, score in zip(scaling_pct, scaling_score):
    ax.text(pct, score + 1.5, f"{score:.1f}", ha="center",
            fontsize=9, color=BLUE, path_effects=outline)

ax.text(108, ext_mean + 1.7, f"{ext_mean:.1f}", ha="center",
        fontsize=9, color=RED, path_effects=outline)

ax.set_xlabel("Training data (% of gold corpus)")
ax.set_ylabel("Combined score")
ax.set_xticks([25, 50, 75, 100])
ax.set_xticklabels(["25%", "50%", "75%", "100%"])
ax.set_xlim(15, 130)
ax.set_ylim(14, 43)

ax.grid(linewidth=0.2, alpha=0.4, color=GREY)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(f"{OUTDIR}/scaling.pdf")
plt.close(fig)
print("✓ scaling.pdf")


# Figure 3: Effect hierarchy

fig, ax = plt.subplots(figsize=(COLWIDTH, 1.65))

y = np.arange(len(hierarchy_labels))
# Monochromatic gradient: darkest for the largest effect
alphas = np.array(hierarchy_vals) / max(hierarchy_vals)
colors = [(*plt.matplotlib.colors.to_rgb(BLUE), 0.35 + 0.65 * a) for a in alphas]

bars = ax.barh(y, hierarchy_vals, height=0.55, color=colors,
               edgecolor="none", zorder=3)

# Direct value labels: inside bar for large values, outside for small values
for bar, val in zip(bars, hierarchy_vals):
    if val > 5:
        ax.text(bar.get_width() - 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="right", fontsize=9,
                color="white", fontweight="bold")
    else:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left", fontsize=9,
                color=INK)

ax.set_yticks(y)
ax.set_yticklabels(hierarchy_labels)
ax.set_xlabel("Effect size (combined score points)")
ax.invert_yaxis()
ax.set_xlim(0, 22)

ax.grid(axis="x", linewidth=0.2, alpha=0.4, color=GREY)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)  # no y tick marks

fig.savefig(f"{OUTDIR}/hierarchy.pdf")
plt.close(fig)
print("✓ hierarchy.pdf")
