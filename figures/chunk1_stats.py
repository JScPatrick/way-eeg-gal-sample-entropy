"""
Chunk 1 steps 1-2:
  (1) paired L-R cluster permutation test, matched 3-channel sensorimotor ROIs
  (2) validation of the cue-locked re-expression machinery (GATE)

Reads only the corrected per-epoch sliding-window CSVs and the onset table.
Writes graphA_stats.json.
"""
import os, json
import numpy as np, pandas as pd, scipy.stats
from mne.stats import permutation_cluster_1samp_test

from _paths import (DERIVED as PE, ONSET, OUT, require,   # repo-relative
                    sliding_window_file)
N_P, N_PERM, SEED, ALPHA = 12, 10000, 42, 0.05
STEP = 124.0                      # native sliding-window step (ms)

CH = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7',
      'C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3',
      'Pz','P4','P8','PO9','O1','Oz','O2','PO10']
I = {c: i for i, c in enumerate(CH)}
LH3 = [I[c] for c in ['FC1', 'C3', 'CP1']]
RH3 = [I[c] for c in ['FC2', 'C4', 'CP2']]

rec = {}
print('=' * 86)
print('CHUNK 1 — STEPS 1 & 2')
print('=' * 86)

# ---------------------------------------------------------------- load trials
KEY = ['participant', 'series', 'trial_in_series']
need_ch = sorted(set(LH3 + RH3))
cols = KEY + ['window_index', 'window_centre_ms'] + \
       [f'sampen_norm_ch{c}' for c in need_ch]
frames = []
for p in range(1, N_P + 1):
    f = sliding_window_file(p, PE)
    frames.append(pd.read_csv(f, usecols=lambda c: c in cols))
sw = pd.concat(frames, ignore_index=True)
sw['L'] = sw[[f'sampen_norm_ch{c}' for c in LH3]].mean(axis=1)
sw['R'] = sw[[f'sampen_norm_ch{c}' for c in RH3]].mean(axis=1)
centres = np.sort(sw['window_centre_ms'].unique())
NW = len(centres)
print(f'loaded {len(sw):,} window rows | {sw[KEY].drop_duplicates().shape[0]:,} '
      f'trials | {NW} windows ({centres[0]:.0f}..{centres[-1]:.0f} ms)')

# ---------------------------------------------------------------- join RT
od = pd.read_csv(ONSET)
od = od[~od['excluded'].astype(str).str.lower().isin(['true', '1'])].copy()
od['RT'] = pd.to_numeric(od['emg_onset_ms_post_led'], errors='coerce')
j = sw.merge(od[KEY + ['RT']], on=KEY, how='left', validate='many_to_one')
n_unmatched = int(j['RT'].isna().sum())
n_trials_sw = sw[KEY].drop_duplicates().shape[0]
n_trials_od = od[KEY].drop_duplicates().shape[0]
print(f'\nJOIN CHECK: unmatched window rows = {n_unmatched}; '
      f'trials in sw = {n_trials_sw:,}; retained trials in onset table = {n_trials_od:,}')
assert n_unmatched == 0, 'join incomplete — Graph B cannot proceed'
assert n_trials_sw == n_trials_od, 'trial sets differ between sw and onset table'
print('JOIN: PASS (every window row matched exactly one onset row)')
sw = j

# ------------------------------------------------- participant_mean_norm (L,R)
pm = sw.groupby(['participant', 'window_centre_ms'])[['L', 'R']].mean()
XL = pm['L'].unstack().reindex(columns=centres).to_numpy()
XR = pm['R'].unstack().reindex(columns=centres).to_numpy()
assert XL.shape == (N_P, NW) and not np.isnan(XL).any()

# =========================================================== 1. paired cluster
print('\n' + '-' * 86)
print('STEP 1 — PAIRED L−R CLUSTER PERMUTATION TEST')
print('-' * 86)
D = XL - XR
print(f'input {D.shape}, MNE permutation_cluster_1samp_test, '
      f'n_perm={N_PERM}, tail=0, threshold=None (t from df={N_P-1}), seed={SEED}')


def clusters_of(M, label):
    T, cl, pv, _ = permutation_cluster_1samp_test(
        M, n_permutations=N_PERM, tail=0, threshold=None, rng=SEED,
        out_type='indices', verbose=False)
    out = []
    print(f'  {label}:')
    any_sig = False
    for c, p in zip(cl, pv):
        e = c[0] if isinstance(c, tuple) else c
        w = np.arange(M.shape[1])[e] if isinstance(e, slice) else np.asarray(e).ravel()
        if p < ALPHA:
            any_sig = True
            direc = 'L more negative than R' if np.mean(T[w]) < 0 else 'R more negative than L'
            out.append(dict(start=float(centres[w[0]]), end=float(centres[w[-1]]),
                            p=float(p), n=len(w), direction=direc,
                            idx=[int(x) for x in w], mean_t=float(np.mean(T[w]))))
            print(f'    {centres[w[0]]:+7.0f} to {centres[w[-1]]:+7.0f} ms  '
                  f'p={p:.5f}  mean t={np.mean(T[w]):+.2f}  {direc}  ({len(w)} windows)')
    if not any_sig:
        print('    (no significant clusters)')
        if len(pv):
            print(f'    smallest cluster p = {min(pv):.4f}')
    return out


rec['cluster_LminusR'] = clusters_of(D, 'L − R difference')
rec['cluster_L_vs_base'] = clusters_of(XL, 'Left vs own baseline')
rec['cluster_R_vs_base'] = clusters_of(XR, 'Right vs own baseline')

# ======================================================= 2. validation (GATE)
print('\n' + '-' * 86)
print('STEP 2 — VALIDATION OF THE CUE-LOCKED RE-EXPRESSION  (GATE)')
print('-' * 86)


def rebin(df, offsets, bin_centres, col):
    """Hierarchical: (trial,window) -> participant x bin -> group.
    offsets: per-row time offset added to window_centre_ms."""
    t = df['window_centre_ms'].to_numpy() + offsets
    edges = np.r_[bin_centres - STEP / 2, bin_centres[-1] + STEP / 2]
    b = np.digitize(t, edges) - 1
    ok = (b >= 0) & (b < len(bin_centres))
    tmp = pd.DataFrame({'p': df['participant'].to_numpy()[ok],
                        'b': b[ok], 'v': df[col].to_numpy()[ok]})
    per = tmp.groupby(['p', 'b'])['v'].mean().unstack().reindex(
        index=range(1, N_P + 1), columns=range(len(bin_centres)))
    cov = tmp.groupby('b')['v'].size().reindex(range(len(bin_centres))).fillna(0)
    return per.to_numpy(), cov.to_numpy()


# --- Check A: zero offset must reproduce the native curve exactly ------------
exactL = np.nanmean(XL, axis=0)
perA, _ = rebin(sw, np.zeros(len(sw)), centres, 'L')
gA = np.nanmean(perA, axis=0)
dA = np.nanmax(np.abs(gA - exactL))
print(f'  Check A1 (zero offset reproduces native grid): max|diff| = {dA:.3e}  '
      f'{"PASS" if dA < 1e-12 else "FAIL"}')

# --- Check A2: constant RT must reproduce the curve shifted by that constant --
C = 283.0
perA2, _ = rebin(sw, np.full(len(sw), C), centres + C, 'L')
gA2 = np.nanmean(perA2, axis=0)
dA2 = np.nanmax(np.abs(gA2 - exactL))
print(f'  Check A2 (constant RT={C:.0f} ms -> pure shift):  max|diff| = {dA2:.3e}  '
      f'{"PASS" if dA2 < 1e-12 else "FAIL"}')

# --- Check B: real grid-offset jitter, zero-mean so only blur remains --------
u = (sw['RT'].to_numpy() % STEP)
u = u - u.mean()                      # zero-mean, range about +/-62 ms
perB, _ = rebin(sw, u, centres, 'L')
gB = np.nanmean(perB, axis=0)
diff = gB - exactL
exc = exactL.max() - exactL.min()
print(f'\n  Check B (real RT-mod-{STEP:.0f} jitter, zero-mean, range '
      f'[{u.min():+.0f},{u.max():+.0f}] ms):')
print(f'    curve peak-to-trough excursion : {exc:.5f} SampEn')
print(f'    max |difference|               : {np.nanmax(np.abs(diff)):.5f} '
      f'({100*np.nanmax(np.abs(diff))/exc:.2f}% of excursion)')
print(f'    RMS  difference                : {np.sqrt(np.nanmean(diff**2)):.5f} '
      f'({100*np.sqrt(np.nanmean(diff**2))/exc:.2f}% of excursion)')
print(f'    excursion after jitter         : {gB.max()-gB.min():.5f} '
      f'({100*(gB.max()-gB.min())/exc:.2f}% of exact — '
      f'predicted ~97% from 3% broadening)')
TW = np.where(np.isin(centres, [-634., -510., -386., -262.]))[0]
sl_e = scipy.stats.linregress(centres[TW], exactL[TW]).slope
sl_b = scipy.stats.linregress(centres[TW], gB[TW]).slope
print(f'    terminal slope exact / jittered: {sl_e:+.4e} / {sl_b:+.4e}  '
      f'({100*abs(sl_b-sl_e)/abs(sl_e):.2f}% change)')

gate = (dA < 1e-12 and dA2 < 1e-12
        and np.nanmax(np.abs(diff)) / exc < 0.05
        and abs(sl_b - sl_e) / abs(sl_e) < 0.10)
print(f'\n  GATE: {"PASS" if gate else "*** FAIL — report before building Graph B ***"}')
rec['validation'] = dict(checkA1=float(dA), checkA2=float(dA2),
                         max_diff=float(np.nanmax(np.abs(diff))),
                         rms_diff=float(np.sqrt(np.nanmean(diff ** 2))),
                         excursion=float(exc),
                         pct_max=float(100 * np.nanmax(np.abs(diff)) / exc),
                         excursion_ratio=float((gB.max() - gB.min()) / exc),
                         slope_exact=float(sl_e), slope_jitter=float(sl_b),
                         gate=bool(gate))

# ---------------------------------------------------------------- trajectories
rec['centres'] = centres.tolist()
rec['term_idx'] = [int(x) for x in TW]
for nm, X in (('L', XL), ('R', XR), ('D', D)):
    rec[f'traj_{nm}'] = dict(
        mean=np.nanmean(X, axis=0).tolist(),
        sem=scipy.stats.sem(X, axis=0, nan_policy='omit').tolist(),
        per_participant=X.tolist())
os.makedirs(OUT, exist_ok=True)
json.dump(rec, open(os.path.join(OUT, 'graphA_stats.json'), 'w'), indent=1)
print(f'\nwrote {OUT}/graphA_stats.json')
