"""
Graph B - cue-locked (LED-referenced) two-hemisphere SampEn time course.

Re-expresses the existing per-epoch sliding-window entropy on a cue-locked
axis: a window's time relative to the LED cue is its onset-relative centre
plus that trial's reaction time. Validated in chunk1_stats.py (gate PASS:
exact reproduction under constant RT; 0.30% max deviation under real
RT-mod-124 jitter). No SampEn is recomputed.
"""
import os, json
import numpy as np, pandas as pd, scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _paths import (DERIVED as PE, ONSET, OUT, require,   # repo-relative
                    sliding_window_file)
N_P, STEP, MIN_COV = 12, 124.0, 0.95

CH = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7',
      'C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3',
      'Pz','P4','P8','PO9','O1','Oz','O2','PO10']
I = {c: i for i, c in enumerate(CH)}
LH3 = [I[c] for c in ['FC1', 'C3', 'CP1']]
RH3 = [I[c] for c in ['FC2', 'C4', 'CP2']]
KEY = ['participant', 'series', 'trial_in_series']

BLUE, VERM, GREEN, ORANGE, PINK = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7'
LGREY = '#EDEDED'
# SCALE multiplies every font size in this figure. See graphA.py.
SCALE = 1.30
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11 * SCALE,
    'axes.labelsize': 12.5 * SCALE, 'axes.titlesize': 13.5 * SCALE,
    'xtick.labelsize': 11 * SCALE, 'ytick.labelsize': 11 * SCALE,
    'legend.fontsize': 9.5 * 1.12,
    'axes.linewidth': 1.0, 'axes.edgecolor': '#333333',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'figure.dpi': 110, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.25})

# ------------------------------------------------------------------ load
need = sorted(set(LH3 + RH3))
cols = KEY + ['window_centre_ms'] + [f'sampen_norm_ch{c}' for c in need]
sw = pd.concat([pd.read_csv(sliding_window_file(p, PE),
                            usecols=lambda c: c in cols)
                for p in range(1, N_P + 1)], ignore_index=True)
sw['L'] = sw[[f'sampen_norm_ch{c}' for c in LH3]].mean(axis=1)
sw['R'] = sw[[f'sampen_norm_ch{c}' for c in RH3]].mean(axis=1)
centres = np.sort(sw['window_centre_ms'].unique())

od = pd.read_csv(ONSET)
od = od[~od['excluded'].astype(str).str.lower().isin(['true', '1'])].copy()
od['RT'] = pd.to_numeric(od['emg_onset_ms_post_led'], errors='coerce')
eo = pd.to_numeric(od['emg_onset_s'], errors='coerce')
od['DIG'] = (pd.to_numeric(od['tFirstDigitTouch_s'], errors='coerce')
             - (eo - od['RT'] / 1000.0)) * 1000.0
od['LIF'] = (pd.to_numeric(od['tLiftOff_s'], errors='coerce')
             - (eo - od['RT'] / 1000.0)) * 1000.0
sw = sw.merge(od[KEY + ['RT']], on=KEY, how='left', validate='many_to_one')
assert sw['RT'].notna().all(), 'join incomplete'

print(f'{len(sw):,} window rows, {sw[KEY].drop_duplicates().shape[0]:,} trials')

# -------------------------------------------------- cue-locked re-expression
sw['cue_t'] = sw['window_centre_ms'] + sw['RT']
k0 = int(np.floor((sw['cue_t'].min()) / STEP))
k1 = int(np.ceil((sw['cue_t'].max()) / STEP))
bins = np.arange(k0, k1 + 1) * STEP                 # bin centres, anchored on 0
edges = np.r_[bins - STEP / 2, bins[-1] + STEP / 2]
sw['b'] = np.digitize(sw['cue_t'].to_numpy(), edges) - 1
sw = sw[(sw['b'] >= 0) & (sw['b'] < len(bins))]

n_trials = sw[KEY].drop_duplicates().shape[0]
# coverage = fraction of trials contributing at least one window to the bin
tb = sw.groupby('b')[KEY].nunique() if False else \
     sw.drop_duplicates(KEY + ['b']).groupby('b').size()
cov = (tb.reindex(range(len(bins))).fillna(0) / n_trials).to_numpy()

per = {}
for side in ('L', 'R'):
    m = sw.groupby(['participant', 'b'])[side].mean().unstack()
    per[side] = m.reindex(index=range(1, N_P + 1),
                          columns=range(len(bins))).to_numpy()
# every plotted bin must contain all 12 participants
n_part = np.sum(~np.isnan(per['L']), axis=0)

keep = (cov >= MIN_COV) & (n_part == N_P)
# keep the single longest contiguous run
idx = np.where(keep)[0]
splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
run = max(splits, key=len)
print(f'\nbins {len(bins)} | plotted {len(run)}  '
      f'({bins[run[0]]:+.0f} to {bins[run[-1]]:+.0f} ms)')
print(f'coverage in plotted range: {100*cov[run].min():.1f}% to '
      f'{100*cov[run].max():.1f}%  |  participants per bin: '
      f'{n_part[run].min()}-{n_part[run].max()} of {N_P}')

t = bins[run]
mL = np.nanmean(per['L'][:, run], axis=0); eL = scipy.stats.sem(per['L'][:, run], axis=0)
mR = np.nanmean(per['R'][:, run], axis=0); eR = scipy.stats.sem(per['R'][:, run], axis=0)

# ---------------------------------- descriptive alignment comparison
A = json.load(open(os.path.join(OUT, 'graphA_stats.json')))
oL = np.array(A['traj_L']['mean'])
exc_onset = oL.max() - oL.min()
exc_cue = mL.max() - mL.min()
print(f'\npeak-to-trough excursion, left ROI:')
print(f'  movement-locked : {exc_onset:.5f}')
print(f'  cue-locked      : {exc_cue:.5f}  ({100*exc_cue/exc_onset:.1f}% of movement-locked)')

# Colours kept consistent with Graph A: black = EMG onset, orange = LED cue.
# Neither may reuse a curve colour (blue / vermillion).
EVENTS = [('EMG onset', od['RT'], 'black', '--'),
          ('first digit contact', od['DIG'].dropna(), GREEN, '--'),
          ('lift-off', od['LIF'].dropna(), PINK, ':')]
for nm, v, _, _ in EVENTS:
    print(f'  {nm:<20} {v.mean():+8.1f} ± {v.std(ddof=1):6.1f} ms (SD)')

# ================================================================== figure
fig, ax = plt.subplots(figsize=(13.6, 7.0))
lo = min((mL - eL).min(), (mR - eR).min()); hi = max((mL + eL).max(), (mR + eR).max())
rng = hi - lo
y0, y1 = lo - 0.30 * rng, hi + 0.40 * rng
ax.set_ylim(y0, y1); H = y1 - y0
XL, XR = -1450, 2780
ax.set_xlim(XL, XR)

ax.axvspan(XL, 0, color=LGREY, alpha=0.55, zorder=0, lw=0)
ax.axhline(0, color='#888888', lw=0.9, zorder=1)
ax.axvline(0, color=ORANGE, lw=2.6, zorder=5)

ax.fill_between(t, mL - eL, mL + eL, color=BLUE, alpha=0.16, lw=0, zorder=2)
ax.fill_between(t, mR - eR, mR + eR, color=VERM, alpha=0.16, lw=0, zorder=2)
ax.plot(t, mL, color=BLUE, lw=2.7, zorder=4)
ax.plot(t, mR, color=VERM, lw=2.7, zorder=4)

# ---- event markers: mean line + horizontal +/-1 SD bar, stacked near top ----
ytop = y1 - 0.075 * H
for i, (nm, v, col, ls) in enumerate(EVENTS):
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    yy = ytop - i * 0.082 * H
    ax.axvline(mu, color=col, ls=ls, lw=1.6, alpha=0.85, zorder=3)
    ax.errorbar(mu, yy, xerr=sd, fmt='o', ms=7, color=col, capsize=5,
                lw=2.2, zorder=7, markeredgecolor='white', markeredgewidth=1.0)
    # label centred above its own marker: cannot overflow the right spine
    ax.text(mu, yy + 0.028 * H, f'{nm}  {mu:+.0f} ± {sd:.0f} ms',
            ha='center', va='bottom', fontsize=9 * SCALE, color=col, zorder=7,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.85))

# bottom-left: clear of the EMG-onset label at top and of the coverage note
# at bottom-right
# small unboxed label beside its own line, set below the EMG-onset marker row
# so the two annotations stay clearly separate
ax.text(-55, ytop - 0.105 * H, 'LED cue (0 ms)', ha='right', va='center',
        fontsize=9.5 * SCALE, color='#B07A00', zorder=8)
_rng = (f'{t[0]:+.0f} to {t[-1]:+.0f}').replace('-', '−')
_cmin = int(np.floor(100 * min(cov[run])))
# stated as the positive fact about what IS plotted, so it cannot be misread
# as "the plotted curve rests on under 95% of trials"
ax.text(t[-1], y0 + 0.045 * H,
        f'plotted {_rng} ms, where ≥{_cmin}% of trials '
        f'and all 12 participants contribute',
        ha='right', va='bottom', fontsize=8.4 * SCALE, color='#555555', zorder=7,
        bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#AAAAAA',
                  lw=0.9, alpha=0.92))

ax.set_xlabel('Time relative to LED cue (ms)')
ax.set_ylabel('Baseline-normalized Sample Entropy', fontweight='bold')
ax.set_title('Sensorimotor EEG Entropy Relative to the Visual Cue',
             fontweight='bold', pad=26 * SCALE)
# Subtitle as a real text object, not mathtext: $\it{}$ renders the
# hyphen as a minus sign and the apostrophe as a prime.
ax.text(0.5, 1.015,
        "Movement-locked windows realigned by each trial's own reaction time",
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=13.5 * SCALE * 0.84, style='italic', color='0.12')

h = [Line2D([], [], color=BLUE, lw=2.7, label='Left sensorimotor (FC1, C3, CP1)'),
     Line2D([], [], color=VERM, lw=2.7, label='Right sensorimotor (FC2, C4, CP2)'),
     Patch(facecolor=BLUE, alpha=0.16, label='±1 SEM across participants'),
     Patch(facecolor=LGREY, label='Pre-cue region'),
     Line2D([], [], color=ORANGE, lw=2.6, label='LED cue (0 ms)'),
     Line2D([], [], color='black', marker='o', ls='--', lw=1.6,
            label='Mean EMG onset ±1 SD'),
     Line2D([], [], color=GREEN, marker='o', ls='--', lw=1.6,
            label='Mean digit contact ±1 SD'),
     Line2D([], [], color=PINK, marker='o', ls=':', lw=1.6,
            label='Mean lift-off ±1 SD')]
_leg = ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, -0.135),
          ncol=3, frameon=True, framealpha=0.95, borderpad=0.7,
          columnspacing=1.5, handlelength=2.2)

fig.subplots_adjust(bottom=0.27)

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
                  '(before the cue)\n'
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

# Sit the box just left of the legend rather than at the far edge.
fig.canvas.draw()
_pb = _t.get_bbox_patch().get_window_extent().transformed(fig.transFigure.inverted())
_t.set_x(max(0.012, _lb.x0 - 0.035 - _pb.width))

for ext, kw in (('png', dict(dpi=400)), ('pdf', {}), ('svg', {})):
    fig.savefig(os.path.join(OUT, f'graphB_hemispheres_cue_locked.{ext}'), **kw)
plt.close(fig)

json.dump(dict(t=t.tolist(), L=mL.tolist(), R=mR.tolist(),
               semL=eL.tolist(), semR=eR.tolist(),
               coverage=cov[run].tolist(), n_trials=int(n_trials),
               excursion_onset=float(exc_onset), excursion_cue=float(exc_cue),
               events={nm: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
                       for nm, v, _, _ in EVENTS}),
          open(os.path.join(OUT, 'graphB_stats.json'), 'w'), indent=1)
print('\nwrote graphB_hemispheres_cue_locked.png/.pdf/.svg + graphB_stats.json')
