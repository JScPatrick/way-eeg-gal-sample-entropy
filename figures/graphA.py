"""Graph A - onset-locked two-hemisphere SampEn time course, matched 3-ch ROIs."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _paths import OUT                                  # repo-relative
S = json.load(open(os.path.join(OUT, 'graphA_stats.json')))
V = json.load(open(os.path.join(OUT, 'verified_stats.json')))   # slopes, events
c = np.array(S['centres']); TW = np.array(S['term_idx'])
HL, HH = 250.0, 248.0
COV_LO, COV_HI = c[TW].min() - HL, c[TW].max() + HH

BLUE, VERM, PURPLE = '#0072B2', '#D55E00', '#7B3294'
GREEN, ORANGE, LGREY = '#009E73', '#E69F00', '#EDEDED'
# SCALE multiplies every font size in this figure. One knob, fully
# reversible; it is a rendering parameter only and touches no data.
SCALE = 1.30
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11 * SCALE,
    'axes.labelsize': 12.5 * SCALE, 'axes.titlesize': 13.5 * SCALE,
    'xtick.labelsize': 11 * SCALE, 'ytick.labelsize': 11 * SCALE,
    'legend.fontsize': 9.5 * 1.12,
    'axes.linewidth': 1.0, 'axes.edgecolor': '#333333',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'figure.dpi': 110, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.25})
EV = V['events']
sL, sR = V['LEFT 3ch (FC1,C3,CP1)'], V['RIGHT 3ch (FC2,C4,CP2)']
pr = V['paired_LR_3ch']

mL, eL = np.array(S['traj_L']['mean']), np.array(S['traj_L']['sem'])
mR, eR = np.array(S['traj_R']['mean']), np.array(S['traj_R']['sem'])

fig, ax = plt.subplots(figsize=(13.6, 7.6))
lo = min((mL - eL).min(), (mR - eR).min())
hi = max((mL + eL).max(), (mR + eR).max())
rng = hi - lo
y0, y1 = lo - 0.56 * rng, hi + 0.24 * rng
ax.set_ylim(y0, y1); H = y1 - y0

ax.axvspan(c[0] - HL, 0, color=LGREY, alpha=0.55, zorder=0, lw=0)
ax.axhline(0, color='#888888', lw=0.9, zorder=1)
ax.axvspan(EV['led_mean'] - EV['led_sd'], EV['led_mean'] + EV['led_sd'],
           color=ORANGE, alpha=0.13, zorder=0, lw=0)
for x, col, ls, lw in ((EV['led_mean'], ORANGE, '--', 1.6),
                       (EV['digit'], GREEN, '--', 1.6),
                       (EV['liftoff'], '#CC79A7', ':', 1.8)):
    ax.axvline(x, color=col, ls=ls, lw=lw, zorder=3)
ax.axvline(0, color='black', lw=2.0, zorder=4)

ax.fill_between(c, mL - eL, mL + eL, color=BLUE, alpha=0.16, lw=0, zorder=2)
ax.fill_between(c, mR - eR, mR + eR, color=VERM, alpha=0.16, lw=0, zorder=2)
ax.plot(c, mL, color=BLUE, lw=2.7, zorder=4)
ax.plot(c, mR, color=VERM, lw=2.7, zorder=4)

# terminal-window indicator (ticks = centres, bar = sample coverage)
yb = y1 - 0.045 * H
ax.plot([COV_LO, COV_HI], [yb, yb], color='#444444', lw=7,
        solid_capstyle='butt', alpha=0.35, zorder=5)
for k in TW:
    ax.plot([c[k], c[k]], [yb - 0.018 * H, yb + 0.018 * H], color='#444444',
            lw=2.2, zorder=6)
ax.text(COV_LO - 60, yb, 'terminal-slope windows:\nticks = centres, '
        'bar = coverage', ha='right', va='center', fontsize=8.6 * SCALE,
        color='#444444', zorder=6, linespacing=1.35)

# ---- cluster rows: paired contrast first, then each vs its own baseline ----
def pf(p):
    """Per-cluster p. Not every cluster is p<0.001 - the right-hemisphere
    pre-onset cluster is p=0.002 - so each is reported individually."""
    return 'p<0.001' if p < 0.001 else f'p={p:.3f}'


# Row labels are short so their white backing cannot paint over the left end
# of a bar, which previously made clusters look as if they started later.
rows = [
    # the cluster tests the L-R LEVEL difference per window, not the slope
    ('L − R paired', PURPLE,
     [(d['start'], d['end'], f"L < R,  {pf(d['p'])}")
      for d in S['cluster_LminusR']]),
    ('Left vs base', BLUE,
     [(d['start'], d['end'],
       f"L {'>' if d['mean_t'] > 0 else '<'} baseline,  {pf(d['p'])}")
      for d in S['cluster_L_vs_base']]),
    ('Right vs base', VERM,
     [(d['start'], d['end'],
       f"R {'>' if d['mean_t'] > 0 else '<'} baseline,  {pf(d['p'])}")
      for d in S['cluster_R_vs_base']]),
]
ytop = y0 + 0.255 * H
for i, (lab, col, segs) in enumerate(rows):
    yy = ytop - i * 0.070 * H
    # No far-left row caption: its white backing was painting over the left end
    # of the bars. Each segment names its own side, direction and p-value, and
    # the bar colour is identified in the legend.
    for (x0, x1, tag) in segs:
        ax.plot([x0, x1], [yy, yy], color=col, lw=7.5, solid_capstyle='butt',
                zorder=6, alpha=0.95 if i == 0 else 0.75)
        ax.text((x0 + x1) / 2, yy + 0.021 * H, tag, ha='center', va='bottom',
                fontsize=8.0 * SCALE, color=col, zorder=7)

# the critical honesty annotation, placed ON the empty left part of the
# paired-contrast row so it cannot cover the row labels below it
# (pre-onset null callout removed at author's request; the point is carried
#  in the figure caption instead)

# Terminal-slope statistics removed from the plot at the author's request and
# exported separately as graphA_terminal_stats_panel.* for optional overlay.

ax.set_xlim(c[0] - HL, c[-1] + HH)
ax.set_xlabel('Time relative to EMG-defined movement onset (ms)')
ax.set_ylabel('Baseline-normalized Sample Entropy', fontweight='bold')
ax.set_title('Sensorimotor EEG Entropy Before and After Movement Onset',
             fontweight='bold', pad=16 * SCALE)

h = [Line2D([], [], color=BLUE, lw=2.7, label='Left sensorimotor (FC1, C3, CP1)'),
     Line2D([], [], color=VERM, lw=2.7, label='Right sensorimotor (FC2, C4, CP2)'),
     Patch(facecolor=BLUE, alpha=0.16, label='±1 SEM across participants'),
     Line2D([], [], color=PURPLE, lw=7.5, label='Paired L − R cluster (centres)'),
     Line2D([], [], color=BLUE, lw=7.5, alpha=0.75,
            label='L vs own baseline cluster'),
     Line2D([], [], color=VERM, lw=7.5, alpha=0.75,
            label='R vs own baseline cluster'),
     Line2D([], [], color='#444444', lw=7, alpha=0.35,
            label='Terminal coverage (−884 to −14 ms)'),
     Patch(facecolor=LGREY, label='Pre-onset region'),
     Line2D([], [], color='black', lw=2.0, label='EMG onset (0 ms)'),
     Line2D([], [], color=ORANGE, ls='--', lw=1.6,
            label='Mean LED cue ({} ms)'.format(
                format(EV['led_mean'], '.0f').replace('-', '−'))),
     Patch(facecolor=ORANGE, alpha=0.13,
           label=f"LED cue ±1 SD ({EV['led_sd']:.0f} ms)"),
     Line2D([], [], color=GREEN, ls='--', lw=1.6,
            label=f"Mean digit contact (+{EV['digit']:.0f} ms)"),
     Line2D([], [], color='#CC79A7', ls=':', lw=1.8,
            label=f"Mean lift-off (+{EV['liftoff']:.0f} ms)")]
_leg = ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, -0.135),
          ncol=3, frameon=True, framealpha=0.95, borderpad=0.7,
          columnspacing=1.1, handlelength=1.7)

fig.subplots_adjust(bottom=0.30)

# Methods note, mirroring the dashed box used on the poster. Positioned from
# the legend's measured extent: savefig bbox='tight' crops any margin we try
# to reserve, so a fixed offset cannot be relied on.
fig.canvas.draw()
_lb = _leg.get_window_extent().transformed(fig.transFigure.inverted())
# Shrink the note until it clears the legend. The legend's width depends on
# the font scale, so no fixed size can be guaranteed to fit.
_fs = 8.3 * SCALE
while True:
    _t = fig.text(0.012, _lb.y1,
                  'Baseline = Mean of first two windows\n'
                  '(\u22122000 ms to \u22121378 ms)\n'
                  'SampEn: m = 2, r = 0.2 \u00d7 SD,\n'
                  'held fixed across windows',
                  ha='left', va='top', fontsize=_fs, color='0.35',
                  style='italic', linespacing=1.55,
                  bbox=dict(boxstyle='square,pad=0.6', facecolor='white',
                            edgecolor='0.55', linestyle=(0, (2, 2)),
                            linewidth=0.9))
    fig.canvas.draw()
    _tb = _t.get_window_extent().transformed(fig.transFigure.inverted())
    if _tb.x1 < _lb.x0 - 0.008 or _fs < 5.0:
        break
    _t.remove()
    _fs -= 0.25

# The dashed frame sits above the text by its padding, so aligning the text
# to the legend top leaves the frame proud of it. Measure the drawn patch and
# drop the box by the difference.
fig.canvas.draw()
_pb = _t.get_bbox_patch().get_window_extent().transformed(fig.transFigure.inverted())
_t.set_y(_t.get_position()[1] - (_pb.y1 - _lb.y1))

for ext, kw in (('png', dict(dpi=400)), ('pdf', {}), ('svg', {})):
    fig.savefig(os.path.join(OUT, f'graphA_hemispheres_onset_locked.{ext}'), **kw)
plt.close(fig)
print('wrote graphA_hemispheres_onset_locked.png/.pdf/.svg')
