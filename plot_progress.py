#!/usr/bin/env python3
"""Generate progress.png from results.tsv — matches original chart style"""
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
        if s == 'crash' or v >= 10:
            continue
        rows.append({
            'val_loss': v,
            'status': s,
            'desc': r['description'].strip(),
        })

n = len(rows)
xs = list(range(1, n + 1))
ys = [r['val_loss'] for r in rows]

# Running best (keeps only)
best_x, best_y = [], []
best = float('inf')
for i, r in enumerate(rows):
    if r['status'] == 'keep' and r['val_loss'] < best:
        best = r['val_loss']
    if r['status'] == 'keep':
        best_x.append(i + 1)
        best_y.append(best)

fig, ax = plt.subplots(figsize=(10, 5))

# Running best step line
if best_x:
    ax.step(best_x, best_y, where='post', color='#2ecc71', linewidth=2,
            alpha=0.8, label='Running best')

# Scatter: green keeps, gray discards
for x, y, r in zip(xs, ys, rows):
    color = '#2ecc71' if r['status'] == 'keep' else '#cccccc'
    edge = 'white' if r['status'] == 'keep' else '#999999'
    size = 80 if r['status'] == 'keep' else 40
    ax.scatter(x, y, color=color, s=size, zorder=5, edgecolors=edge, linewidths=0.5)

# Annotate running-best keeps only
prev_best = float('inf')
for x, y, r in zip(xs, ys, rows):
    if r['status'] == 'keep' and r['val_loss'] < prev_best:
        prev_best = r['val_loss']
        short = r['desc'].split('(')[0].strip()
        if len(short) > 40:
            short = short[:37] + '...'
        ax.annotate(short, (x, y), textcoords='offset points',
                    xytext=(6, 6), fontsize=6.5, color='#555555',
                    ha='left', va='bottom')

# Best line
best_val = min(ys)
ax.axhline(best_val, color='#2ecc71', linewidth=0.6, linestyle=':', alpha=0.5)
ax.text(n + 0.5, best_val, f' {best_val:.3f}', fontsize=8, color='#2ecc71', va='center')

n_kept = sum(1 for r in rows if r['status'] == 'keep')
ax.set_xlabel('Experiment #', fontsize=11)
ax.set_ylabel('Validation Loss (lower is better)', fontsize=11)
ax.set_title(f'ANE Autoresearch Progress: {n} Experiments, {n_kept} Kept Improvements',
             fontsize=13, pad=12)
ax.grid(axis='y', alpha=0.3)

legend = [
    mpatches.Patch(color='#cccccc', label='Discarded'),
    mpatches.Patch(color='#2ecc71', label='Kept'),
    plt.Line2D([0], [0], color='#2ecc71', linewidth=2, label='Running best'),
]
ax.legend(handles=legend, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('progress.png', dpi=200, bbox_inches='tight')
print(f'Saved progress.png ({n} experiments, best={best_val:.4f})')
