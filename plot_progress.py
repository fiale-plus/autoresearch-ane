#!/usr/bin/env python3
"""Generate progress.png from results.tsv"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

rows = []
with open('results.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        v = float(r['val_loss'])
        s = r['status'].strip()
        if v <= 0 or v >= 10:
            continue
        rows.append({
            'val_loss': v,
            'status': s,
            'desc': r['description'].strip(),
        })

# Indices of val-split calibration rows (each data expansion changes the
# evaluation segment, so mark every boundary rather than comparing raw values)
split_markers = []
for i, r in enumerate(rows):
    desc = r['desc'].lower()
    if '[aug21:calib]' in desc:
        split_markers.append((i, '8-shard val split →'))
    elif '[aug21:calib50]' in desc:
        split_markers.append((i, '50-shard val split →'))

n = len(rows)
xs = list(range(len(rows)))
ys = [r['val_loss'] for r in rows]
retained_pairs = [(i, r['val_loss']) for i, r in enumerate(rows) if r['status'] == 'keep']
retained_best_i, retained_best = min(retained_pairs, key=lambda pair: pair[1]) if retained_pairs else (None, float('nan'))
observed_best = min(ys) if ys else float('nan')

# Running best (keeps only) — step line connecting kept improvements
best_x, best_y = [], []
best = float('inf')
for i, r in enumerate(rows):
    if r['status'] == 'keep' and r['val_loss'] < best:
        best = r['val_loss']
    if r['status'] == 'keep':
        best_x.append(i)
        best_y.append(best)

# Light gray background style
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Grid — very light, y-axis only
ax.grid(axis='y', color='#e0e0e0', linewidth=0.5)
ax.grid(axis='x', visible=False)

# Running best step line
if best_x:
    ax.step(best_x, best_y, where='post', color='#2ecc71', linewidth=1.8,
            alpha=0.7, label='Running best', zorder=3)

# Scatter: green keeps (larger), gray discards (smaller, lighter)
for x, y, r in zip(xs, ys, rows):
    if r['status'] == 'keep':
        ax.scatter(x, y, color='#2ecc71', s=70, zorder=5,
                   edgecolors='white', linewidths=0.5)
    else:
        ax.scatter(x, y, color='#cccccc', s=30, zorder=4,
                   edgecolors='#bbbbbb', linewidths=0.3, alpha=0.7)

# Annotate only running-best keeps (new records)
prev_best = float('inf')
for x, y, r in zip(xs, ys, rows):
    if r['status'] == 'keep' and r['val_loss'] < prev_best:
        prev_best = r['val_loss']
        short = r['desc'].split('(')[0].strip()
        if len(short) > 45:
            short = short[:42] + '...'
        ax.annotate(short, (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, color='#888888',
                    ha='left', va='bottom', rotation=45)
if retained_best_i is not None:
    ax.scatter([retained_best_i], [retained_best], color='#1abc9c', marker='*',
               s=180, zorder=6, edgecolors='white', linewidths=0.8)
    ax.annotate(f'Retained best {retained_best:.6f}', (retained_best_i, retained_best),
                textcoords='offset points', xytext=(-100, -18), fontsize=8,
                color='#168f80', ha='left', va='top', fontweight='bold')

n_kept = sum(1 for r in rows if r['status'] == 'keep')
ax.set_xlabel('Experiment #', fontsize=12, color='#555555')
ax.set_ylabel('Validation Loss (lower is better)', fontsize=12, color='#555555')
title = f'ANE Autoresearch Progress: {n} Experiments, {n_kept} Kept Improvements'
if retained_best_i is not None:
    suffix = ' (not retained)' if observed_best < retained_best else ''
    title += f'\nRetained best: {retained_best:.6f} | observed minimum: {observed_best:.6f}{suffix}'
ax.set_title(title, fontsize=14, fontweight='bold', color='#333333', pad=15)

# Clean spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#cccccc')

ax.tick_params(colors='#888888', labelsize=10)

# Dashed markers where the val split changed (Aug 21 data expansions)
if split_markers:
    y_bottom, y_top = ax.get_ylim()
    y_span = y_top - y_bottom
    for marker_i, (split_x, label) in enumerate(split_markers):
        ax.axvline(split_x - 0.5, color='#e67e22', linewidth=1.2,
                   linestyle='--', alpha=0.8, zorder=2)
        ax.text(split_x - 0.5, y_top - marker_i * 0.08 * y_span, label,
                fontsize=9, color='#e67e22', ha='left', va='top')

legend = [
    mpatches.Patch(color='#cccccc', label='Discarded'),
    mpatches.Patch(color='#2ecc71', label='Kept'),
    plt.Line2D([0], [0], color='#2ecc71', linewidth=1.8, alpha=0.7, label='Running best'),
]
ax.legend(handles=legend, fontsize=10, loc='upper right',
          framealpha=0.9, edgecolor='#dddddd')

# Tight y-axis: show range of actual progress with some padding
valid_ys = [y for y in ys if y < 8]
if valid_ys:
    ymin, ymax = min(valid_ys), max(valid_ys)
    ax.set_ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.35)

plt.tight_layout()
plt.savefig('progress.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved progress.png ({n} experiments, retained_best={retained_best:.6f}, observed_min={observed_best:.6f})')
