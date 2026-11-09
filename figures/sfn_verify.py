"""Stage B — figure-specific statistical verification, corrected per-epoch data."""
import sys, os, json
import numpy as np, pandas as pd, scipy.stats
import pingouin as pg
from mne.stats import permutation_cluster_1samp_test
from statsmodels.stats.multitest import multipletests

from _paths import (DERIVED as PE, PERWINDOW as PW, ONSET, OUT,  # repo-relative
                    sliding_window_file)
os.makedirs(OUT, exist_ok=True)
N_P, N_CH, N_PERM, ALPHA, SEED = 12, 32, 10000, 0.05, 42

CH = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7',
      'C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3',
      'Pz','P4','P8','PO9','O1','Oz','O2','PO10']
I = {c: i for i, c in enumerate(CH)}
ROI7 = [I[c] for c in ['FC1','FC2','C3','Cz','C4','CP1','CP2']]
LH3  = [I[c] for c in ['FC1','C3','CP1']]
RH3  = [I[c] for c in ['FC2','C4','CP2']]
LH5  = [I[c] for c in ['FC5','FC1','C3','CP5','CP1']]
RH5  = [I[c] for c in ['FC6','FC2','C4','CP6','CP2']]
LOBES = {'frontal': ['F7','F3','Fz','F4','F8','FC5','FC6'],
         'central': ['FC1','FC2','C3','Cz','C4','CP1','CP2'],
         'parietal': ['CP5','CP6','P7','P3','Pz','P4','P8'],
         'occipital': ['PO9','O1','Oz','O2','PO10']}
TERM = [-634., -510., -386., -262.]


def load(d):
    nc = [f'sampen_norm_ch{c}' for c in range(N_CH)]
    need = ['participant', 'window_index', 'window_centre_ms'] + nc
    pmn, centres = None, None
    for p in range(1, N_P + 1):
        f = sliding_window_file(p, d)
        df = pd.read_csv(f, usecols=lambda c: c in need)
        g = df.groupby(['window_index', 'window_centre_ms'], sort=True)[nc].mean()
        if centres is None:
            centres = np.array([c for _, c in g.index], float)
            pmn = np.full((N_P, len(centres), N_CH), np.nan)
        pmn[p - 1] = g.to_numpy()
    return pmn, centres


def slopes(pmn, centres, widx, chidx):
    X = np.nanmean(pmn[:, :, chidx], axis=2) if len(chidx) > 1 else pmn[:, :, chidx[0]]
    c = centres[widx]
    return np.array([scipy.stats.linregress(c, X[p, widx]).slope for p in range(N_P)])


def perm1(v, seed=SEED, n=N_PERM):
    obs = float(np.mean(v))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(v * rng.choice([-1., 1.], size=len(v))))
                     for _ in range(n)])
    p = float((int(np.sum(np.abs(null) >= abs(obs))) + 1) / (n + 1))
    sd = float(np.std(v, ddof=1))
    ci = scipy.stats.t.interval(.95, len(v) - 1, loc=obs, scale=sd / np.sqrt(len(v)))
    return obs, p, obs / sd, ci


pmn, centres = load(PE)
# The per-window dataset is the superseded CF1 comparison and is optional.
if sliding_window_file(1, PW, required=False):
    pmn_pw, _ = load(PW)
else:
    pmn_pw = None
    print('  [note] per-window dataset not present; skipping the CF1 '
          'comparison trajectory (not needed for any poster figure).')
TW = np.where(np.isin(centres, TERM))[0]
rec = {}

print('=' * 88)
print('STAGE B — FIGURE-SPECIFIC VERIFICATION (corrected per-epoch data)')
print('=' * 88)
print(f'windows={len(centres)}  terminal idx={TW.tolist()} '
      f'centres={centres[TW].astype(int).tolist()}')
print(f'terminal sample coverage: {centres[TW].min()-250:.0f} to '
      f'{centres[TW].max()+248:.0f} ms')

# ---- 1. terminal slopes ------------------------------------------------
print('\n--- 1. Terminal slopes (windows 9-12, per-epoch) ---')
print(f"  {'ROI':<28}{'slope':>13}{'p':>9}{'d':>7}{'neg':>7}{'95% CI':>28}")
for nm, idx in [('sensorimotor 7ch (ORIGINAL)', ROI7), ('LEFT 3ch (FC1,C3,CP1)', LH3),
                ('RIGHT 3ch (FC2,C4,CP2)', RH3), ('LEFT 5ch (robustness)', LH5),
                ('RIGHT 5ch (robustness)', RH5)]:
    s = slopes(pmn, centres, TW, idx)
    o, p, d, ci = perm1(s)
    rec[nm] = dict(slope=o, p=p, d=d, ci=list(ci), neg=int(np.sum(s < 0)),
                   per_participant=s.tolist())
    print(f'  {nm:<28}{o:>+13.4e}{p:>9.5f}{d:>7.2f}{int(np.sum(s<0)):>5}/12'
          f'  [{ci[0]:+.3e},{ci[1]:+.3e}]')

# ---- 2. paired L-R + multiplicity -------------------------------------
print('\n--- 2. Paired L-R contrast + multiplicity ---')
L3, R3 = slopes(pmn, centres, TW, LH3), slopes(pmn, centres, TW, RH3)
L5, R5 = slopes(pmn, centres, TW, LH5), slopes(pmn, centres, TW, RH5)
d3, d5 = L3 - R3, L5 - R5
o3, p3, dd3, ci3 = perm1(d3)
o5, p5, dd5, ci5 = perm1(d5)
w3 = scipy.stats.wilcoxon(L3, R3)
t3 = scipy.stats.ttest_rel(L3, R3)
print(f'  3ch PRIMARY : L-R {o3:+.4e}  p_perm={p3:.5f}  d={dd3:+.2f}  '
      f'L steeper {int(np.sum(d3<0))}/12  CI[{ci3[0]:+.2e},{ci3[1]:+.2e}]')
print(f'               cross-check: Wilcoxon p={w3.pvalue:.5f}, '
      f'paired t p={t3.pvalue:.5f}')
print(f'  5ch robust  : L-R {o5:+.4e}  p_perm={p5:.5f}  d={dd5:+.2f}  '
      f'L steeper {int(np.sum(d5<0))}/12')
hol = multipletests([p3, p5], method='holm')
bon = multipletests([p3, p5], method='bonferroni')
print(f'  family of 2 (3ch primary + 5ch robustness):')
print(f'    Bonferroni: 3ch p={bon[1][0]:.4f} ({"sig" if bon[0][0] else "NOT sig"}), '
      f'5ch p={bon[1][1]:.4f} ({"sig" if bon[0][1] else "NOT sig"})')
print(f'    Holm      : 3ch p={hol[1][0]:.4f} ({"sig" if hol[0][0] else "NOT sig"}), '
      f'5ch p={hol[1][1]:.4f} ({"sig" if hol[0][1] else "NOT sig"})')
rec['paired_LR_3ch'] = dict(diff=o3, p_unadj=p3, d=dd3, n_neg=int(np.sum(d3 < 0)),
                            ci=list(ci3), p_bonf2=float(bon[1][0]),
                            p_holm2=float(hol[1][0]),
                            wilcoxon_p=float(w3.pvalue), ttest_p=float(t3.pvalue),
                            L=L3.tolist(), R=R3.tolist())
rec['paired_LR_5ch'] = dict(diff=o5, p_unadj=p5, d=dd5, n_neg=int(np.sum(d5 < 0)))

# ---- 3. matched LH cluster test (same MNE call as pipeline) -----------
print('\n--- 3. Matched cluster permutation, identical MNE call to pipeline ---')
for nm, idx in [('LEFT 3ch', LH3), ('RIGHT 3ch', RH3), ('sensorimotor 7ch', ROI7)]:
    X = np.nan_to_num(np.nanmean(pmn[:, :, idx], axis=2), nan=0.0)
    T, cl, pv, _ = permutation_cluster_1samp_test(
        X, n_permutations=N_PERM, tail=0, threshold=None, rng=SEED,
        out_type='indices', verbose=False)
    sig = [(c, p) for c, p in zip(cl, pv) if p < ALPHA]
    rec.setdefault('clusters', {})[nm] = []
    print(f'  {nm}:')
    for c, p in sig:
        # MNE returns each cluster as a tuple; element 0 is an index array
        # (out_type='indices') or a slice (out_type='mask') over the time axis.
        e = c[0] if isinstance(c, tuple) else c
        w = np.arange(len(centres))[e] if isinstance(e, slice) else np.asarray(e).ravel()
        direc = 'above baseline' if np.mean(T[w]) > 0 else 'below baseline'
        rec['clusters'][nm].append(dict(start=float(centres[w[0]]),
                                        end=float(centres[w[-1]]), p=float(p),
                                        direction=direc, idx=w.tolist()))
        print(f'    {centres[w[0]]:+7.0f} to {centres[w[-1]]:+7.0f} ms  '
              f'p={p:.5f}  {direc}  ({len(w)} windows)')
    if not sig:
        print('    (no significant clusters)')

# ---- 4. regional / lobe ------------------------------------------------
print('\n--- 4. Regional (lobe) verification against pipeline CSV ---')
lt = {}
for lb, names in LOBES.items():
    idx = [I[c] for c in names]
    lt[lb] = np.nanmean(np.nanmean(pmn[:, :, idx], axis=2)[:, TW], axis=1)
    print(f'  {lb:<10} terminal mean {np.mean(lt[lb]):+.6f}  '
          f'SEM {scipy.stats.sem(lt[lb]):.6f}')
long = pd.DataFrame({'subj': np.tile(np.arange(N_P), 4),
                     'lobe': np.repeat(list(LOBES), N_P),
                     'val': np.concatenate([lt[l] for l in LOBES])})
aov = pg.rm_anova(data=long, dv='val', within='lobe', subject='subj',
                  correction=True, detailed=True)
row = aov.iloc[0]
col = lambda *names: next((n for n in names if n in aov.columns), None)
c_gg, c_un = col('p-GG-corr', 'p_GG_corr'), col('p-unc', 'p_unc')
pgg = float(row[c_gg]) if c_gg else np.nan
punc = float(row[c_un]) if c_un else np.nan
print(f"  pingouin {pg.__version__} columns: {list(aov.columns)}")
print(f"  rm_anova  F={float(row['F']):.4f}  p_unc={punc:.3e}  "
      f"p_GG_corr={pgg:.3e}  eps={float(row.get('eps', np.nan)):.4f}  "
      f"sphericity={row.get('sphericity')}")
print('  pipeline CSV reported F=18.619040077787314  p=4.832215929762135e-07')
ph = []
for other in ['frontal', 'parietal', 'occipital']:
    w = scipy.stats.wilcoxon(lt['central'], lt[other])
    ph.append((other, float(w.statistic), float(w.pvalue)))
pb = multipletests([x[2] for x in ph], method='bonferroni')
print('  post-hoc central vs other (Wilcoxon, Bonferroni x3):')
for (o, st, p), pa, rj in zip(ph, pb[1], pb[0]):
    dirn = 'central higher' if np.mean(lt['central']) > np.mean(lt[o]) else 'other higher'
    print(f'    central vs {o:<10} W={st:5.1f}  p={p:.5f}  p_bonf={pa:.5f}  '
          f'{"SIG" if rj else "ns":>3}  ({dirn})')
rec['lobe'] = dict(F=float(row['F']), p_unc=punc, p_GG=pgg,
                   means={k: float(np.mean(v)) for k, v in lt.items()},
                   posthoc=[dict(vs=o, p=p, p_bonf=float(pa)) for (o, _, p), pa
                            in zip(ph, pb[1])])

# ---- 5. per-channel terminal slopes (Fig 3 alternative) ---------------
print('\n--- 5. Per-channel terminal slopes with BH-FDR (Fig 3 candidate) ---')
ps, sl = [], []
for c in range(N_CH):
    s = slopes(pmn, centres, TW, [c])
    o, p, d, _ = perm1(s)
    sl.append(o); ps.append(p)
fdr = multipletests(ps, method='fdr_bh')
rec['per_channel'] = dict(slope=sl, p=ps, p_fdr=fdr[1].tolist(),
                          sig=fdr[0].tolist(), names=CH)
order = np.argsort(sl)
print(f'  {int(np.sum(fdr[0]))}/32 channels significant after BH-FDR')
print('  10 steepest declines:')
for i in order[:10]:
    print(f'    {CH[i]:<6}{sl[i]:+.4e}  p={ps[i]:.5f}  p_fdr={fdr[1][i]:.5f}'
          f'  {"*" if fdr[0][i] else ""}')

# ---- 6. event timings --------------------------------------------------
print('\n--- 6. Event annotation timings, verified from onset CSV ---')
od = pd.read_csv(ONSET)
od = od[~od['excluded'].astype(str).str.lower().isin(['true', '1'])]
rt = pd.to_numeric(od['emg_onset_ms_post_led'], errors='coerce').dropna()
eo = pd.to_numeric(od['emg_onset_s'], errors='coerce')
dig = (pd.to_numeric(od['tFirstDigitTouch_s'], errors='coerce') - eo) * 1000
lif = (pd.to_numeric(od['tLiftOff_s'], errors='coerce') - eo) * 1000
print(f'  mean LED cue rel. onset : {-rt.mean():+.1f} ms  (SD {rt.std(ddof=1):.1f}) '
      f'[= -(mean reaction time)]')
print(f'  mean digit contact      : {dig.mean():+.1f} ms  (SD {dig.std(ddof=1):.1f})')
print(f'  mean liftoff            : {lif.mean():+.1f} ms  (SD {lif.std(ddof=1):.1f})')
print(f'  n trials = {len(od)}')
rec['events'] = dict(led_mean=float(-rt.mean()), led_sd=float(rt.std(ddof=1)),
                     digit=float(dig.mean()), liftoff=float(lif.mean()),
                     n_trials=int(len(od)))

# ---- 7. trajectories for plotting -------------------------------------
rec['centres'] = centres.tolist()
rec['term_idx'] = TW.tolist()
for nm, idx in [('LH3', LH3), ('RH3', RH3), ('ROI7', ROI7)]:
    X = np.nanmean(pmn[:, :, idx], axis=2)
    rec[f'traj_{nm}'] = dict(mean=np.nanmean(X, axis=0).tolist(),
                             sem=scipy.stats.sem(X, axis=0, nan_policy='omit').tolist(),
                             per_participant=X.tolist())
if pmn_pw is not None:
    Xpw = np.nanmean(pmn_pw[:, :, LH3], axis=2)
    rec['traj_LH3_perwindow'] = dict(mean=np.nanmean(Xpw, axis=0).tolist())
with open(os.path.join(OUT, 'verified_stats.json'), 'w') as f:
    json.dump(rec, f, indent=1)
print(f'\nWrote {OUT}/verified_stats.json')
