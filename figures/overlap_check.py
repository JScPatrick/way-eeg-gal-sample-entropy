"""Overlap check for the two time-course figures (qc_check.py covers Fig 3)."""
import sys, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.text as mtext

fails = []
for fn, name in (('graphA.py', 'Figure 1'), ('graphB.py', 'Figure 2')):
    src = re.sub(r"\nfor ext, kw in.*", "\n", open(fn).read(), flags=re.S)
    ns = {'__name__': '__overlap__'}
    exec(compile(src, fn, 'exec'), ns)
    fig = ns['fig']
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    # A tick outside the view limits is culled at draw time even though
    # get_visible() stays True, and tick labels carry no .axes back-reference,
    # so gather the phantoms from each axes directly.
    phantom = set()
    for _ax in fig.axes:
        lo, hi = sorted(_ax.get_ylim())
        for tl in _ax.get_yticklabels():
            if not lo <= tl.get_position()[1] <= hi:
                phantom.add(id(tl))
        lo, hi = sorted(_ax.get_xlim())
        for tl in _ax.get_xticklabels():
            if not lo <= tl.get_position()[0] <= hi:
                phantom.add(id(tl))

    items = []
    for a in fig.findobj(mtext.Text):
        t = a.get_text().strip()
        if not (t and a.get_visible()):
            continue
        bb = a.get_window_extent(r)
        fb = fig.bbox
        if bb.x1 < fb.x0 or bb.x0 > fb.x1 or bb.y1 < fb.y0 or bb.y0 > fb.y1:
            continue          # artist parked off-canvas
        if id(a) in phantom:
            continue
        items.append((t, bb))
    n = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (t1, b1), (t2, b2) = items[i], items[j]
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > 1 and oy > 1:
                fails.append(f'{name}: {t1[:38]!r} <-> {t2[:38]!r} ({ox:.0f}x{oy:.0f}px)')
                n += 1
    print(f'{name}: {len(items)} text objects, {n} overlaps')
print()
print('FAILURES:' if fails else 'NO TEXT OVERLAPS')
for f in fails:
    print(' -', f)
sys.exit(1 if fails else 0)
