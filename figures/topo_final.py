"""
Poster figure: scalp distribution of terminal SampEn slopes.

DATA / STATS ARE NOT TOUCHED HERE. Everything is read from
output/topo_verified.json, written by verify_topo.py from the per-participant
entropy tables in results/ (CF1-corrected, per-epoch tolerance).

----------------------------------------------------------------------------
GEOMETRY FIX (the vertical offset)
----------------------------------------------------------------------------
MNE draws three things in two different frames:

  * mne/viz/topomap.py  _plot_topomap, line ~114:
        outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    -> clip_origin is HARD-CODED to (0, 0). _setup_interp anchors the
       interpolation grid and the extrapolation circle there.

  * _make_head_outlines draws the head circle, nose and ears at sphere[:2].

  * mne/channels/layout.py _find_topomap_coords ends with
        out += sphere[:2]
    so electrode coordinates are also expressed about sphere[:2].

With an auto-fitted sphere the origin is (-0.9, +12.9, +41.3) mm, so the head
outline sits 12.9 mm above the frame the fill is clipped and extrapolated in:
a white crescent at the top of the scalp and overspill at the bottom.

Fix, without distorting the projection: the azimuthal projection is computed
about the ANATOMICALLY FITTED origin (verified below to be independent of the
sphere radius), the resulting 2-D coordinates are then recentred by
subtracting sphere[:2], and the figure is drawn with sphere=(0, 0, 0, R).
Outline frame, clip_origin and electrode frame then coincide exactly.

Rejected: sphere=<scalar>, which resets the origin to (0,0,0) and displaces
electrodes by up to 37.7 mm, moving the positive region from right-parietal
to left-frontal.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import mne
from mne.channels.layout import _find_topomap_coords
import warnings
warnings.filterwarnings('ignore')

from _paths import OUT                                  # repo-relative
S = json.load(open(f'{OUT}/topo_verified.json'))
CH = S['names']
sl = np.array(S['slope'])
qv = np.array(S['p_fdr'])
sig = np.array(S['sig'], bool)

info = mne.create_info(CH, 500., 'eeg')
info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                 match_case=False)
_, org, _ = mne.bem.fit_sphere_to_headshape(info, units='m')

# project about the fitted origin, then recentre into a single frame
POS = _find_topomap_coords(info, picks='eeg',
                           sphere=(org[0], org[1], org[2], 0.132))
POS = POS - np.array([org[0], org[1]])
RMAX = np.linalg.norm(POS, axis=1).max()
R = 0.135
assert RMAX < R, (RMAX, R)
assert max(1.0, RMAX * 1.01 / R) == 1.0        # mask_scale must stay 1.0
SPHERE = (0.0, 0.0, 0.0, R)

v = np.abs(sl).max() * 1e5                      # +/-2.559, unchanged
# SCALE multiplies every font size here. Rendering only; touches no data.
SCALE = 1.30
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig = plt.figure(figsize=(7.0, 8.7))
ax = fig.add_axes([0.06, 0.375, 0.88, 0.550])

im, _ = mne.viz.plot_topomap(
    sl * 1e5, POS, axes=ax, show=False, cmap='RdBu_r',
    vlim=(-v, v), sensors=False, contours=5, outlines='head',
    sphere=SPHERE, extrapolate='head',
    mask=sig,
    mask_params=dict(marker='o', markerfacecolor='none', markeredgecolor='k',
                     linewidth=0, markersize=10, markeredgewidth=1.9))

# all 32 electrodes as dots: shows what is measured vs interpolated
ax.scatter(POS[:, 0], POS[:, 1], s=9, c='0.30', zorder=4, linewidths=0)

# Limits are set before labels are placed so the label geometry is computed in
# the final data-to-inch scale. Horizontal padding leaves room for L/R; the
# vertical padding only has to clear the nose at 1.15 R.
# Vertical padding is asymmetric: the nose reaches 1.15 R at the top, but
# nothing extends below the head circle, so reserving nose-height at the
# bottom too would only create dead space above the colour bar.
PADX, PADY_T, PADY_B = R * 1.30, R * 1.17, R * 1.05
ax.set_xlim(-PADX, PADX)
ax.set_ylim(-PADY_B, PADY_T)

# ---- labels for every BH-FDR significant electrode ------------------------
# Offsets are searched, not hand-placed. A label must not overlap another
# label, must not land on ANY other electrode (a label sitting on an
# unlabelled dot would read as labelling that dot), and must stay inside the
# head. Among the admissible positions the closest to its own electrode wins.
FS = 11.5 * SCALE
UPI = (2 * PADX) / (fig.get_size_inches()[0] * 0.88)  # data units per inch
HALF_W = lambda s: (0.62 * FS / 72.0 * UPI) * len(s) / 2 + 0.0030
HALF_H = (FS / 72.0 * UPI) / 2 + 0.0026
CLEAR = 0.0045                                        # dot clearance

SIGI = [i for i in np.where(sig)[0]]
SIGI.sort(key=lambda i: -np.hypot(*POS[i]))           # outermost placed first
placed, OFF = [], {}
for i in SIGI:
    name, (x, y) = CH[i], POS[i]
    hw, hh = HALF_W(name), HALF_H
    best = None
    for rad in (0.017, 0.022, 0.027, 0.033):
        for ang in np.arange(0, 360, 15):
            dx = rad * np.cos(np.radians(ang))
            dy = rad * np.sin(np.radians(ang))
            lx, ly = x + dx, y + dy
            if np.hypot(lx, ly) + max(hw, hh) > R * 0.97:
                continue                              # keep inside the head
            bad = False
            for j in range(len(CH)):                  # every electrode, not
                if j == i:                            # only the labelled ones
                    continue
                if (abs(POS[j, 0] - lx) < hw + CLEAR
                        and abs(POS[j, 1] - ly) < hh + CLEAR):
                    bad = True
                    break
            if bad:
                continue
            for (px, py, phw, phh) in placed:
                if abs(px - lx) < hw + phw + 0.002 and \
                   abs(py - ly) < hh + phh + 0.002:
                    bad = True
                    break
            if bad:
                continue
            # tie-break: prefer pointing away from the array centre
            out = (dx * x + dy * y) / (rad * max(np.hypot(x, y), 1e-9))
            score = rad - 0.004 * out
            if best is None or score < best[0]:
                best = (score, lx, ly, dx, dy)
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f'no admissible label position for {name}')
    _, lx, ly, dx, dy = best
    placed.append((lx, ly, hw, hh))
    OFF[name] = (dx, dy)

HALO = [pe.withStroke(linewidth=2.8, foreground='white')]
for i in SIGI:
    name = CH[i]
    dx, dy = OFF[name]
    x, y = POS[i]
    if np.hypot(dx, dy) > 0.0165:                     # leader line if needed
        f0, f1 = 0.34, 0.66
        ax.plot([x + dx * f0, x + dx * f1], [y + dy * f0, y + dy * f1],
                lw=1.0, color='0.12', zorder=5, solid_capstyle='round',
                path_effects=[pe.withStroke(linewidth=2.6,
                                            foreground='white')])
    ax.text(x + dx, y + dy, name, ha='center', va='center', fontsize=FS,
            fontweight='bold', color='black', zorder=6, path_effects=HALO)

# hemisphere labels
ax.text(-R * 1.19, 0, 'L', ha='center', va='center',
        fontsize=19 * SCALE, fontweight='bold', color='0.25')
ax.text(R * 1.19, 0, 'R', ha='center', va='center',
        fontsize=19 * SCALE, fontweight='bold', color='0.25')

ax.set_title('Scalp Distribution of Late Pre-Movement Entropy Changes',
             fontsize=15.5 * SCALE, fontweight='bold', pad=12)

# ---- colour bar -----------------------------------------------------------
cax = fig.add_axes([0.28, 0.333, 0.44, 0.020])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Terminal slope (SampEn/ms × 10⁻⁵)',
             fontsize=11.5 * SCALE, labelpad=5)
cb.ax.tick_params(labelsize=10 * SCALE)
cb.outline.set_linewidth(0.6)

# Plain-language direction markers either side of the scale.
cax.text(-0.03, 0.5, 'Less entropy →', transform=cax.transAxes,
         ha='right', va='center', fontsize=9.8 * SCALE, color='0.25',
         fontweight='bold')
cax.text(1.03, 0.5, '← More entropy', transform=cax.transAxes,
         ha='left', va='center', fontsize=9.8 * SCALE, color='0.25',
         fontweight='bold')

key = fig.text(0.5, 0.203, '○  q < .05, BH-FDR across 32 channels',
               ha='center', fontsize=10.5 * SCALE, color='0.25')

# The colour-bar label's final height is only known after a draw (it depends
# on the tick labels), so the key and the table are positioned from measured
# bounding boxes rather than from guessed offsets.
TBH = 0.185
def _fy(artist):
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(fig.transFigure.inverted())

fig.canvas.draw()
lab_y0 = _fy(cb.ax.xaxis.label).y0
key.set_y(lab_y0 - 0.013 - _fy(key).height)
fig.canvas.draw()
tb_top = _fy(key).y0 - 0.022
TB_BOTTOM = tb_top - TBH
assert TB_BOTTOM > 0.005, f'table runs off the canvas: {TB_BOTTOM}'

# ---- table: five steepest declines ----------------------------------------
tb = fig.add_axes([0.10, TB_BOTTOM, 0.80, TBH])
tb.set_xlim(0, 1)
tb.set_ylim(0, 1)
tb.set_autoscale_on(False)   # otherwise the rule below rescales the axes and
tb.axis('off')               # the unclipped text is thrown off-canvas
order = np.argsort(sl)[:5]
COLX = (0.05, 0.66, 0.97)
tb.text(0.5, 0.97, 'Five steepest declines', ha='center', va='top',
        fontsize=11.5 * SCALE, fontweight='bold')
hdr_y = 0.74
for x, h, al in zip(COLX, ('Channel', 'Slope (×10⁻⁵)', 'q'),
                    ('left', 'right', 'right')):
    tb.text(x, hdr_y, h, ha=al, va='center', fontsize=10.2 * SCALE, color='0.30')
tb.plot([0.02, 0.99], [hdr_y - 0.085] * 2, lw=0.9, color='0.45')
for r, i in enumerate(order):
    y = hdr_y - 0.165 - r * 0.125
    tb.text(COLX[0], y, CH[i], ha='left', va='center', fontsize=10.8 * SCALE,
            fontweight='bold')
    tb.text(COLX[1], y, f'{sl[i]*1e5:.3f}'.replace('-', '−'),
            ha='right', va='center', fontsize=10.8 * SCALE)
    tb.text(COLX[2], y, f'{qv[i]:.3f}'.lstrip('0'),
            ha='right', va='center', fontsize=10.8 * SCALE)

# Guard against reading rank as significance. Ranks 1-6 happen to be the
# significant channels, but ranks 7-9 and 11 (Fz, F3, O1, FC1) are steeper
# than C4, which IS significant: consistency across participants, not
# magnitude, drives the test.
tb.text(0.5, -0.02, 'Rank is by magnitude, not significance: Fz, F3 and O1 '
        'are steeper than C4 but q > .05.',
        ha='center', va='top', fontsize=8.8 * SCALE, color='0.40', style='italic')

for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{OUT}/fig3_topography.{ext}',
                **({'dpi': 400} if ext == 'png' else {}),
                facecolor='white', bbox_inches='tight', pad_inches=0.12)

print(f'R={R}  max|pos|={RMAX:.5f}  margin {(R-RMAX)*1000:.2f} mm')
print(f'colour limit +/-{v:.3f} x1e-5   significant {sig.sum()}/32')
print('labelled:', [CH[i] for i in np.where(sig)[0]])
print('table   :', [(CH[i], round(sl[i]*1e5, 3), round(qv[i], 4)) for i in order])
