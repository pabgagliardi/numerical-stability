"""
Stage 5 — visualise.py
Produces all required plots for the numerical stability exercise:

1. Distribution of intra/inter distances per precision (KDE + histogram)
2. Intra vs inter distances across precision levels (bar plot)
3. Ratio (inter/intra) across precision levels
4. Per-word breakdown of distances
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

DISTANCES_CSV = "results/distances.csv"
OUTPUT_DIR    = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load distances ─────────────────────────────────────────────────
df = pd.read_csv(DISTANCES_CSV)

PRECISIONS = ["float64", "float32", "float16", "int8"]
COLORS     = {
    "intra": "#2196F3",   # blue
    "inter": "#F44336",   # red
}

# ══════════════════════════════════════════════════════════════════
# Plot 1 — KDE of distance distributions per precision
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
axes = axes.flatten()

for ax, precision in zip(axes, PRECISIONS):
    sub_intra = df[(df["precision"] == precision) & 
                   (df["type"] == "intra")]["distance"].dropna()
    sub_inter = df[(df["precision"] == precision) & 
                   (df["type"] == "inter")]["distance"].dropna()

    for vals, label, color in [
        (sub_intra, "intra-speaker", COLORS["intra"]),
        (sub_inter, "inter-speaker", COLORS["inter"]),
    ]:
        # histogram
        ax.hist(vals, bins=40, alpha=0.3, color=color, density=True)
        # KDE
        kde  = gaussian_kde(vals)
        x    = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x, kde(x), color=color, linewidth=2, label=label)
        # vertical mean line
        ax.axvline(vals.mean(), color=color, linewidth=1.5,
                   linestyle="--", alpha=0.8)

    ax.set_title(f"{precision}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Distribution of intra- and inter-speaker cosine distances\n"
    "across precision levels",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot1_distributions.png", dpi=150)
plt.close()
print("Saved plot1_distributions.png")

# ══════════════════════════════════════════════════════════════════
# Plot 2 — Bar plot: mean intra vs inter per precision
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

x      = np.arange(len(PRECISIONS))
width  = 0.35
means  = {}

for dtype in ["intra", "inter"]:
    means[dtype] = [
        df[(df["precision"] == p) & 
           (df["type"] == dtype)]["distance"].mean()
        for p in PRECISIONS
    ]
    stds = [
        df[(df["precision"] == p) & 
           (df["type"] == dtype)]["distance"].std()
        for p in PRECISIONS
    ]
    offset = -width/2 if dtype == "intra" else width/2
    ax.bar(x + offset, means[dtype], width,
           label=f"{dtype}-speaker",
           color=COLORS[dtype], alpha=0.8,
           yerr=stds, capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(PRECISIONS, fontsize=11)
ax.set_ylabel("Mean cosine distance")
ax.set_title("Mean intra- vs inter-speaker distances\nacross precision levels",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(means["inter"]), max(means["intra"])) * 1.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot2_mean_distances.png", dpi=150)
plt.close()
print("Saved plot2_mean_distances.png")

# ══════════════════════════════════════════════════════════════════
# Plot 3 — Ratio (inter/intra) across precision levels
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

ratios = [
    df[(df["precision"] == p) & (df["type"] == "inter")]["distance"].mean() /
    df[(df["precision"] == p) & (df["type"] == "intra")]["distance"].mean()
    for p in PRECISIONS
]

bars = ax.bar(PRECISIONS, ratios, color="#9C27B0", alpha=0.8)
ax.axhline(1.0, color="black", linewidth=1.2,
           linestyle="--", label="ratio = 1 (no separation)")

# annotate bars
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0005,
            f"{ratio:.4f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Inter / Intra distance ratio")
ax.set_title("Speaker separability ratio across precision levels",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.99, max(ratios) * 1.02)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot3_ratio.png", dpi=150)
plt.close()
print("Saved plot3_ratio.png")

# ══════════════════════════════════════════════════════════════════
# Plot 4 — Per-word ratio across precision levels
# ══════════════════════════════════════════════════════════════════
words = sorted(df["word"].unique())
fig, ax = plt.subplots(figsize=(13, 5))

x      = np.arange(len(words))
width  = 0.2
prec_colors = {
    "float64": "#1565C0",
    "float32": "#42A5F5",
    "float16": "#FF7043",
    "int8":    "#8D6E63",
}

for i, precision in enumerate(PRECISIONS):
    ratios_per_word = []
    for word in words:
        intra = df[(df["precision"] == precision) &
                   (df["type"] == "intra") &
                   (df["word"] == word)]["distance"].mean()
        inter = df[(df["precision"] == precision) &
                   (df["type"] == "inter") &
                   (df["word"] == word)]["distance"].mean()
        ratios_per_word.append(inter / intra if intra > 0 else np.nan)

    offset = (i - 1.5) * width
    ax.bar(x + offset, ratios_per_word, width,
           label=precision,
           color=prec_colors[precision], alpha=0.85)

ax.axhline(1.0, color="black", linewidth=1.2,
           linestyle="--", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(words, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Inter / Intra distance ratio")
ax.set_title("Per-word speaker separability ratio across precision levels",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot4_per_word_ratio.png", dpi=150)
plt.close()
print("Saved plot4_per_word_ratio.png")

print("\nAll plots saved to results/")