"""Descriptive lighting figure for EXP-000 (no thresholds drawn on purpose)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
d = pd.read_csv("/Users/atruong/thesis-data/ucf-crime/lighting_test.csv")
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": "#c9c8c2", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), facecolor=SURFACE)
for a in ax:
    a.set_facecolor(SURFACE); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
ax[0].hist(d.mean_luma, bins=range(30, 176, 5), color=BLUE, edgecolor=SURFACE, linewidth=2)
ax[0].set(xlabel="mean luma per video (Y, 0–255)", ylabel="videos", title="Mean luma (1 frame/s)")
ax[1].hist(d.mean_chroma, bins=[x * 1.0 for x in range(0, 48, 1)], color=BLUE, edgecolor=SURFACE, linewidth=1)
ax[1].set(xlabel="mean |Cr−128|+|Cb−128| per video", ylabel="videos", title="Colourfulness (≈0 = greyscale/IR)")
ax[2].scatter(d.mean_luma, d.greyscale_frac, s=22, color=BLUE, edgecolor=SURFACE, linewidth=1, alpha=0.85)
ax[2].set(xlabel="mean luma per video", ylabel="fraction of near-grey frames", title="Luma vs greyscale fraction")
for a in ax: a.title.set_fontsize(10); a.title.set_ha("left"); a.title.set_x(0)
fig.suptitle(f"UCF-Crime test set, n = {len(d)} videos, {d.n_sampled.sum():,} sampled frames", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150, facecolor=SURFACE)
