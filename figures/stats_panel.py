"""Standalone terminal-slope statistics panel, for optional overlay on Graph A.
Transparent outside the box so it can be dropped straight onto the figure."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _paths import OUT                                  # repo-relative
V = json.load(open(os.path.join(OUT, 'verified_stats.json')))
sL, sR = V['LEFT 3ch (FC1,C3,CP1)'], V['RIGHT 3ch (FC2,C4,CP2)']
pr = V['paired_LR_3ch']

def num(x, fmt='.2e'):
    """Typeset a number with true minus signs, leaving prose hyphens alone."""
    return format(x, fmt).replace('-', '−')


TXT = (
    "Terminal pre-onset slope (SampEn/ms)\n"
    "window centres −634 to −262 ms   "
    "(EEG samples −884 to −14 ms)\n"
    f"Left    {num(sL['slope'])}   p = {sL['p']:.4f}   "
    f"d = {num(sL['d'], '.2f')}   {sL['neg']}/12\n"
    f"Right   {num(sR['slope'])}   p = {sR['p']:.4f}   "
    f"d = {num(sR['d'], '.2f')}   {sR['neg']}/12\n"
    f"L − R   {num(pr['diff'])}   p = {pr['p_unadj']:.3f} unadjusted   "
    f"({pr['p_bonf2']:.3f} Bonferroni, n.s.)"
)

fig = plt.figure(figsize=(7.6, 1.55))
fig.patch.set_alpha(0.0)
fig.text(0.5, 0.5, TXT, ha='center', va='center',
         family='DejaVu Sans Mono', fontsize=10.5, linespacing=1.7,
         bbox=dict(boxstyle='round,pad=0.55', fc='white', ec='#777777',
                   lw=1.2, alpha=1.0))
for ext, kw in (('png', dict(dpi=400, transparent=True)),
                ('pdf', dict(transparent=True)),
                ('svg', dict(transparent=True))):
    # pad_inches must exceed the rounded-box padding or the box consumes the
    # whole margin and no transparent border survives the tight crop
    fig.savefig(os.path.join(OUT, f'graphA_terminal_stats_panel.{ext}'),
                bbox_inches='tight', pad_inches=0.15, **kw)
plt.close(fig)

with open(os.path.join(OUT, 'graphA_terminal_stats.txt'), 'w') as f:
    f.write(TXT + '\n')
print('wrote graphA_terminal_stats_panel.png/.pdf/.svg and .txt')
print()
print(TXT)
