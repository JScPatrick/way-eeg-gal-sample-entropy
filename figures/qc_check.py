"""Automated QC of the rendered figure: overlaps, mislabels, data fidelity."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.text as mtext

ns = {'__name__': '__qc__'}
_HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_HERE, 'topo_final.py')).read(), ns)
fig, ax, POS, CH = ns['fig'], ns['ax'], ns['POS'], ns['CH']
sl, qv, sig, R = ns['sl'], ns['qv'], ns['sig'], ns['R']
fig.canvas.draw()          # table axes is added after the last draw
rnd = fig.canvas.get_renderer()
fails = []

def bb(a):
    return a.get_window_extent(rnd)

# ---- A. text vs text --------------------------------------------------------
texts = []
for a in fig.findobj(mtext.Text):
    t = a.get_text().strip()
    if not t or not a.get_visible():
        continue
    texts.append((t, a, bb(a), a.axes))
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        (t1, _, b1, _), (t2, _, b2, _) = texts[i], texts[j]
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ox > 1.0 and oy > 1.0:
            fails.append(f'TEXT OVERLAP  {t1!r} <-> {t2!r}  ({ox:.0f}x{oy:.0f} px)')

# ---- B. no label sits on an electrode that is not its own -------------------
disp = ax.transData.transform(POS)
signames = [CH[i] for i in np.where(sig)[0]]
for t, a, b, axp in texts:
    if t not in signames or axp is not ax:   # table column is not a label
        continue
    own = CH.index(t)
    for k, (px, py) in enumerate(disp):
        if k == own:
            continue
        if b.x0 - 2 <= px <= b.x1 + 2 and b.y0 - 2 <= py <= b.y1 + 2:
            fails.append(f'LABEL {t!r} covers electrode {CH[k]!r}')

# ---- C. every significant channel labelled exactly once ---------------------
drawn = [t for t, _, _, axp in texts if t in signames and axp is ax]
if sorted(drawn) != sorted(signames):
    fails.append(f'label set mismatch: drawn={sorted(drawn)} expected={sorted(signames)}')
for n in signames:
    if drawn.count(n) != 1:
        fails.append(f'{n} drawn {drawn.count(n)} times')
nonsig = [CH[i] for i in np.where(~sig)[0]]
for t, _, _, axp in texts:
    if t in nonsig and axp is ax:
        fails.append(f'NON-SIGNIFICANT channel {t!r} appears as a label')

# ---- D. data fidelity against the json -------------------------------------
S = json.load(open(os.path.join(ns['OUT'], 'topo_verified.json')))
if CH != S['names']:
    fails.append('channel order differs from source json')
if not np.allclose(sl, S['slope']):
    fails.append('slopes differ from source json')
if not np.array_equal(sig, np.array(S['sig'], bool)):
    fails.append('significance mask differs from source json')
exp = [CH[i] for i in np.argsort(S['slope'])[:5]]
if exp != ['P3', 'C3', 'CP5', 'CP1', 'CP2']:
    fails.append(f'table order unexpected: {exp}')

# ---- E. orientation and geometry -------------------------------------------
L = ['F3', 'FC1', 'C3', 'CP1', 'P3', 'O1', 'T7', 'FC5', 'CP5', 'F7', 'P7']
Rt = ['F4', 'FC2', 'C4', 'CP2', 'P4', 'O2', 'T8', 'FC6', 'CP6', 'F8', 'P8']
lx = np.mean([POS[CH.index(c), 0] for c in L])
rx = np.mean([POS[CH.index(c), 0] for c in Rt])
if not (lx < 0 < rx):
    fails.append(f'ORIENTATION REVERSED  left={lx:+.4f} right={rx:+.4f}')
rmax = np.linalg.norm(POS, axis=1).max()
if rmax >= R:
    fails.append(f'electrode outside head circle: {rmax:.5f} >= {R}')

print('channels labelled :', sorted(drawn))
print('orientation       : left %+.4f  right %+.4f  -> %s'
      % (lx, rx, 'OK' if lx < 0 < rx else 'REVERSED'))
print('max |pos| %.5f  head R %.3f  margin %.2f mm' % (rmax, R, (R - rmax) * 1000))
print('colour limit      : +/-%.3f x1e-5' % (np.abs(sl).max() * 1e5))
print()
print('FAILURES:' if fails else 'ALL CHECKS PASSED')
for f in fails:
    print('  -', f)
