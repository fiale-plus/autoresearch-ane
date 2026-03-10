#!/usr/bin/env python3
"""Generate progress.png for ANE experiments from results.tsv"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {"keep": "#2ecc71", "discard": "#e67e22", "crash": "#e74c3c"}
MARKERS = {"keep": "o", "discard": "s", "crash": "x"}

rows = []
with open("results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

xs, ys, statuses, descs = [], [], [], []
for i, r in enumerate(rows):
    v = float(r["val_loss"])
    s = r["status"]
    if s == "crash" or v >= 100:
        continue
    xs.append(i + 1)
    ys.append(v)
    statuses.append(s)
    descs.append(r["description"])

# Best-so-far line (keeps only)
best_x, best_y = [], []
best = float("inf")
for i, r in enumerate(rows):
    v = float(r["val_loss"])
    s = r["status"]
    if s == "keep" and v < 100:
        if v < best:
            best = v
        best_x.append(i + 1)
        best_y.append(best)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

# Best-so-far line
if best_x:
    ax.step(best_x, best_y, where="post", color="#58a6ff", linewidth=1.5,
            linestyle="--", alpha=0.6, label="best so far")

# Scatter points
for x, y, s, d in zip(xs, ys, statuses, descs):
    ax.scatter(x, y, color=COLORS[s], marker=MARKERS[s], s=80, zorder=5,
               edgecolors="white", linewidths=0.4)

# Annotate keeps
for x, y, s, d in zip(xs, ys, statuses, descs):
    if s == "keep":
        short = d.split("(")[0].strip()[:30]
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=6.5, color="#cdd9e5")

ax.set_xlabel("Experiment #", color="#8b949e", fontsize=11)
ax.set_ylabel("val_loss (↓ better)", color="#8b949e", fontsize=11)
ax.set_title("autoresearch-ane — mar10 run", color="#e6edf3", fontsize=13, pad=12)
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.grid(axis="y", color="#21262d", linewidth=0.7)

legend_patches = [
    mpatches.Patch(color=COLORS["keep"], label="keep"),
    mpatches.Patch(color=COLORS["discard"], label="discard"),
    mpatches.Patch(color=COLORS["crash"], label="crash"),
]
ax.legend(handles=legend_patches, facecolor="#161b22", edgecolor="#30363d",
          labelcolor="#cdd9e5", fontsize=9, loc="upper right")

best_val = min(ys) if ys else None
if best_val:
    ax.axhline(best_val, color="#58a6ff", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.text(0.01, best_val, f" best={best_val:.4f}", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#58a6ff", va="bottom")

plt.tight_layout()
plt.savefig("progress_ane.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved progress_ane.png")
