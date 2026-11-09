"""Verify every claim the topography figure makes, from source."""
import os, json
import numpy as np, pandas as pd, scipy.stats
import matplotlib; matplotlib.use('Agg')
import mne
from mne.channels.layout import _find_topomap_coords
from statsmodels.stats.multitest import multipletests

from _paths import (DERIVED as PE, OUT, require,   # repo-relative
                    sliding_window_file)
N_P, N_CH, N_PERM, SEED = 12, 32, 10000, 42
CH = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7',
      'C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3',
      'Pz','P4','P8','PO9','O1','Oz','O2','PO10']
TERM = [-634., -510., -386., -262.]

print('=' * 78)
print('TOPOGRAPHY VERIFICATION')
print('=' * 78)

nc = [f'sampen_norm_ch{c}' for c in range(N_CH)]
pmn, centres = None, None
for p in range(1, N_P + 1):
    d = pd.read_csv(sliding_window_file(p, PE),
                    usecols=lambda c: c in ['window_index','window_centre_ms']+nc)
    g = d.groupby(['window_index','window_centre_ms'], sort=True)[nc].mean()
    if centres is None:
        centres = np.array([c for _, c in g.index], float)
        pmn = np.full((N_P, len(centres), N_CH), np.nan)
    pmn[p-1] = g.to_numpy()
TW = np.where(np.isin(centres, TERM))[0]
print(f'windows={len(centres)}  terminal idx={TW.tolist()} '
      f'centres={centres[TW].astype(int).tolist()}')
print(f'terminal EEG sample coverage: {centres[TW].min()-250:.0f} to '
      f'{centres[TW].max()+248:.0f} ms  (all strictly pre-onset: '
      f'{bool((centres[TW]+248 < 0).all())})')

# ---- per-channel terminal slope + sign-flip permutation + BH-FDR ----
slopes, ps = [], []
for c in range(N_CH):
    X = pmn[:, :, c]
    s = np.array([scipy.stats.linregress(centres[TW], X[p, TW]).slope
                  for p in range(N_P)])
    obs = float(np.mean(s))
    rng = np.random.default_rng(SEED)
    null = np.array([float(np.mean(s * rng.choice([-1.,1.], size=N_P)))
                     for _ in range(N_PERM)])
    ps.append(float((int(np.sum(np.abs(null) >= abs(obs))) + 1) / (N_PERM + 1)))
    slopes.append(obs)
slopes = np.array(slopes); ps = np.array(ps)
rej, pfdr, _, _ = multipletests(ps, method='fdr_bh')

print(f'\n--- per-channel terminal slope, BH-FDR across 32 channels ---')
print(f'significant: {int(rej.sum())}/32  -> '
      f'{[CH[i] for i in np.where(rej)[0]]}')
print(f'slope range: {slopes.min():+.4e} to {slopes.max():+.4e} SampEn/ms')
print(f'channels with POSITIVE slope: {[CH[i] for i in np.where(slopes>0)[0]]}')
print(f'colour limit used by figure (symmetric): +/-{np.abs(slopes).max()*1e5:.3f} x1e-5')

# ---- ORIENTATION TEST: does MNE put left-hemisphere channels on the left? ----
print('\n--- ORIENTATION TEST (empirical, not assumed) ---')
info = mne.create_info(CH, sfreq=500., ch_types='eeg')
info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                 match_case=False)
pos = _find_topomap_coords(info, picks='eeg', sphere='auto')
for name in ['C3','C4','Fz','Oz','T7','T8','CP1','CP2']:
    i = CH.index(name)
    print(f'  {name:<5} plotted x={pos[i,0]:+.4f}  y={pos[i,1]:+.4f}'
          f'   -> {"LEFT" if pos[i,0] < -1e-6 else "RIGHT" if pos[i,0] > 1e-6 else "midline"}')
lefts  = [c for c in ['F3','FC1','C3','CP1','P3','O1','T7','FC5','CP5','F7','P7']]
rights = [c for c in ['F4','FC2','C4','CP2','P4','O2','T8','FC6','CP6','F8','P8']]
lx = np.mean([pos[CH.index(c),0] for c in lefts])
rx = np.mean([pos[CH.index(c),0] for c in rights])
print(f'  mean x of 11 LEFT channels : {lx:+.4f}')
print(f'  mean x of 11 RIGHT channels: {rx:+.4f}')
ok = lx < 0 < rx
print(f'  ORIENTATION: {"CORRECT (left hemisphere on image-left)" if ok else "*** REVERSED ***"}')

# where do the significant channels sit?
sig_x = [pos[i,0] for i in np.where(rej)[0]]
print(f'  significant channels x-positions: '
      f'{[f"{CH[i]}:{pos[i,0]:+.3f}" for i in np.where(rej)[0]]}')
print(f'  -> {sum(1 for x in sig_x if x<0)} left, '
      f'{sum(1 for x in sig_x if x>0)} right, '
      f'{sum(1 for x in sig_x if abs(x)<1e-6)} midline')

# ---- is the most-negative region actually left? ----
print('\n--- is the darkest region genuinely left-sided? ---')
w = -slopes  # magnitude of decline
cx = float(np.sum(w * pos[:,0]) / np.sum(w))
print(f'  decline-weighted centroid x = {cx:+.4f}  '
      f'({"LEFT" if cx<0 else "RIGHT"} of midline)')
print(f'  mean slope over 11 left ch : {np.mean([slopes[CH.index(c)] for c in lefts]):+.4e}')
print(f'  mean slope over 11 right ch: {np.mean([slopes[CH.index(c)] for c in rights]):+.4e}')

json.dump(dict(names=CH, slope=slopes.tolist(), p=ps.tolist(),
               p_fdr=pfdr.tolist(), sig=rej.tolist(),
               pos=pos.tolist(), centres=centres.tolist(),
               term_idx=[int(x) for x in TW]),
          open(os.path.join(OUT, 'topo_verified.json'),'w'), indent=1)
print('\nwrote topo_verified.json')
