# EEG sample entropy and EMG-defined movement onset (WAY-EEG-GAL)

Does EEG sample entropy (SampEn) change reliably *before* EMG-defined movement
onset during a cued grasp-and-lift task?

This repository is **code only**: the analysis pipeline, the statistics, and
the figure scripts behind the SfN poster. No raw data, no intermediate data and
no finished figures are redistributed here. Everything is regenerated from the
publicly available WAY-EEG-GAL dataset by the scripts in `src/` and `figures/`.

---

## Before you use any of this

```bash
python verify_corrections.py
```

Two corrections were applied after the first round of results. Neither fails
loudly if it is lost — the pipeline would simply produce the earlier,
uncorrected numbers. This script checks that the corrected code is what you
actually have. **If it reports a failure, the outputs will not match the
published results.**

---

## Reproducing everything, end to end

Four steps. There is no shortcut that skips step 3: the intermediate entropy
tables are not included in this repository, and the figure scripts read them.

### 1. Get the dataset

WAY-EEG-GAL is public and hosted on figshare
(https://doi.org/10.6084/m9.figshare.988376). Download the `.mat` files and
place them **flat** in `data/mat/`:

```
data/mat/P1_AllLifts.mat
data/mat/WS_P1_S1.mat
...
```

See `data/README.md` for the citation, the licence and the expected layout.

**You also need the electrode-coordinate file** `chanlabels_32channel.xyz`,
which is **required** to complete the pipeline and is **not redistributed
here** — it is part of the dataset, not of this code.

It ships with WAY-EEG-GAL in **Supplementary File 2, `Utilities.zip`**, on the
dataset's figshare record (https://doi.org/10.6084/m9.figshare.988376). Luciw,
Jarocka & Edin describe it as the file "used to localize the electrode
positions in EEGLab". Extract `Utilities.zip` and copy the file to:

```
data/chanlabels_32channel.xyz
```

That path is `CHAN_FILE` in `src/config.py`; keep the dataset's own filename.
`src/visualization.py` reads it in `_load_montage()` to build the MNE montage
for the pipeline's internal topographic map (Figure 2 of Phase 5), so a
complete `src/main.py` run requires it. Without it, Phase 5 raises
`FileNotFoundError: data/chanlabels_32channel.xyz` after writing the QC ERP
figure and Figure 1, and the run aborts there. Phases 0–4 finish first, so
everything in `results/` is already written, and the three figures in step 4
below are unaffected — they use MNE's `standard_1020` template instead
(limitation 5).

`src/verify_setup.py` reports `FOUND` / `MISSING` for this file and warns if it
is absent. **Check that before starting a long run**, so a 3–9 hour
computation does not end on a missing-file error.

For reference, the format is one row per channel, space-separated, **no
header**, coordinates in **centimetres** (`_load_montage()` multiplies by 0.01,
because MNE works in metres):

```
index  X  Y  Z  label
```

32 rows, whose labels match `EEG_CHANNEL_NAMES` in `src/config.py`.

### 2. Install dependencies

```bash
pip install -r requirements-lock.txt   # exact tested versions (needs Python >= 3.12)
```

See *Environment* below for what was and was not tested.

### 3. Run the pipeline

Run **from the repository root**, not from inside `src/`:

```bash
python src/verify_setup.py   # environment and data checks
python src/main.py           # Phases 0-7
```

`verify_setup.py` imports `src.sampen`, so it needs the repository root on the
path; run from inside `src/` it reports a spurious
`sampen.py error: No module named 'src'`. `main.py` is unaffected either way —
it calls `os.chdir()` to the repository root at startup — but running both the
same way avoids the confusion.

`main.py` creates `results/` and `figures/` output automatically. It **stops at
an interactive checkpoint** after EMG onset detection so you can inspect the
onset QC figures, and resumes when you type `continue`. In a non-interactive
session it exits cleanly at that point instead, so run it in a terminal.

By its own estimate the entropy phase takes **3–9 hours** depending on
`N_JOBS`. Per-participant results are written as they are computed, so the run
is safe to interrupt and resume. Phase timings are recorded in
`results/pipeline_log.txt`.

This step produces everything downstream needs:

| output | contents |
|---|---|
| `results/sliding_window_entropy_P1.csv` … `_P12.csv` | per-trial, per-window, per-channel SampEn: raw (`sampen_ch0..31`) and baseline-normalised (`sampen_norm_ch0..31`), with `window_index` and `window_centre_ms`. 29 windows per trial. **Read by every figure script.** |
| `results/trial_level_entropy_P1.csv` … `_P12.csv` | one row per trial; the per-participant cache that controls re-run skipping |
| `results/onset_detection_results.csv` | one row per trial: EMG onset time, detecting channel, reaction time relative to the LED cue, exclusion flag and reason. **Read by the figure scripts.** |
| `results/channel_stats.csv`, `cluster_permutation_results.csv`, `participant_summary.csv`, `roi_slope_results.csv`, `roi_cluster_permutation_results.csv`, `roi_participant_summary.csv`, `roi_condition_generalizability.csv`, `hemisphere_terminal_results.csv`, `lobe_terminal_slopes_anova.csv`, `lobe_terminal_means_anova.csv`, `excluded_trials.csv` | the pipeline's own statistical results |
| `results/pipeline_log.txt` | library versions, settings and phase timings for that run |
| `figures/qc_*.png` and the pipeline's internal figures | Phase 5 output |

### 4. Reproduce the three poster figures

```bash
cd figures
python sfn_verify.py       # ROI slopes, paired contrasts, events -> verified_stats.json
python chunk1_stats.py     # cluster statistics + cue-locking gate -> graphA_stats.json
python graphA.py           # Figure 1: movement-locked time course
python graphB.py           # Figure 2: cue-relative time course
python stats_panel.py      # terminal-slope statistics panel
python verify_topo.py      # per-channel slopes + BH-FDR          -> topo_verified.json
python topo_final.py       # Figure 3: scalp topography
python qc_check.py         # automated QC of Figure 3
python overlap_check.py    # automated QC of Figures 1 and 2
```

Run them in that order — later scripts read JSON written by earlier ones.
Outputs (PNG, PDF, SVG and the statistics JSON) land in `output/`, which is
created automatically.

These scripts read `results/` by default. `_paths.py` resolves the inputs and
each location can be overridden:

```bash
WEG_RESULTS=/path/to/results python verify_topo.py
```

| variable | default | purpose |
|---|---|---|
| `WEG_RESULTS` | `results/` | directory holding the per-participant entropy tables and the onset table |
| `WEG_ONSET` | inside `WEG_RESULTS` | explicit path to `onset_detection_results.csv` |
| `WEG_OUT` | `output/` | where figures and statistics JSON are written |
| `WEG_PERWINDOW` | `results/per_window` | optional legacy per-window-tolerance dataset; absent by default and not needed for any figure |

`.csv` and `.csv.gz` are both accepted; pandas decompresses by extension.
Resolution is deliberately strict — if two differently-named candidates for the
same participant are present, the scripts stop with an error rather than
choosing one. `results/` also holds `trial_level_entropy_P*.csv`, which has a
different shape, and silently loading the wrong table would corrupt every
downstream number.

### Re-running after you change a setting

`main.py` skips any participant whose `results/trial_level_entropy_P*.csv`
already exists, and it does **not** record which settings produced that file.
So if you change `SAMPEN_TOLERANCE_MODE`, `TERMINAL_WINDOW_RULE` or any other
analysis parameter and simply re-run, the stale files are silently reused and
your change has no effect.

Move the old results aside rather than deleting them, so a run you may still
want is never lost:

```bash
mv results "results_$(date +%Y%m%d_%H%M%S)"
python src/main.py
```

A fresh clone has no `results/` directory, so a first run is always a genuine
computation.

---

## Method in brief

- **Dataset**: WAY-EEG-GAL, 12 right-handed participants, 32-channel EEG at
  500 Hz, 5-channel EMG at 4000 Hz, cued grasp-and-lift task. 2,982 trials
  retained after exclusions.
- **EMG onset**: 20–450 Hz bandpass, full-wave rectify, 50 ms RMS envelope;
  onset = first crossing of mean + 3 SD of the pre-cue baseline sustained for
  25 ms, taken as the earliest across the 5 channels. The kinematic marker
  `tHandStart` is used only as an exclusion sanity check, never as the anchor.
- **EEG**: 1–40 Hz Butterworth (order 4), zero-phase `filtfilt`.
- **SampEn**: m = 2, Chebyshev distance, r = 0.2 × SD computed **once per
  channel per epoch** and held fixed across all windows of that epoch.
- **Sliding windows**: 29 windows of 500 ms (250 samples), stepped 124 ms.
  A window spans `[centre − 250 ms, centre + 248 ms]` — asymmetric, because
  the window length is even and `centre_idx = start + 125`.
- **Baseline**: mean of windows 0 and 1 (centres −1750 and −1626 ms),
  subtracted per trial per channel.
- **Terminal pre-onset slope**: linear slope across the four final windows
  whose last sample still precedes onset — centres
  `[−634, −510, −386, −262] ms`, EEG coverage **−884 to −14 ms**.
- **Inference**: participant-level. Sign-flip permutation (N = 10,000,
  seed 42, two-tailed, (b+1)/(N+1)) for slopes; cluster-based permutation
  (Maris & Oostenveld 2007) for time series; Benjamini–Hochberg FDR across
  the 32 channels for the topography.

## Headline results

| test | value |
|---|---|
| Left sensorimotor ROI (FC1, C3, CP1) terminal slope | −1.99e−05 SampEn/ms, *p* = .0015, *d* = −1.29, 11/12 participants |
| Right sensorimotor ROI (FC2, C4, CP2) terminal slope | −1.40e−05 SampEn/ms, *p* = .0023, *d* = −1.27, 11/12 |
| Paired left − right terminal slope | −5.98e−06, *p* = .040 unadjusted, **.081 Bonferroni (n.s.)** |
| Paired left − right cluster | −138 to +1722 ms (window centres), *p* < .001 |
| Channels with significant terminal slope | 8 of 32 after BH-FDR (*q* < .05): C3, Cz, C4, CP5, CP1, CP2, P3, P4 |

---

## The two corrections

**CF1 — SampEn tolerance.** The original pipeline recomputed `r = 0.2 × SD`
separately for *each* 250-sample window. SampEn is only comparable across
segments when `r` is held constant across them, so a window whose amplitude
rose (an MRCP, or post-onset movement activity) received a larger `r`, more
template matches, and a lower SampEn with no change in underlying complexity.
`r` is now derived once per channel from the full epoch and passed to every
window. `SAMPEN_TOLERANCE_MODE = 'per_window'` reproduces the original
behaviour for comparison and must not be used for reported results.

**W1 — terminal window selection.** The original terminal set
`[−386, −262, −138, −14] ms` looked pre-onset by window *centre*, but because
a window extends to `centre + 248 ms`, two of those windows contained
post-onset EEG (−138 → 22.4% post-onset; −14 → 47.2%). 174 of 1000 samples
behind the "pre-onset" slope were recorded at or after onset. The corrected
set is the four consecutive windows whose last sample still precedes onset.
`stats.py` now asserts this at runtime, so a contaminated set cannot pass
silently.

Correcting CF1 reduced the slope magnitude substantially. The principal
sensorimotor decline remained significant; some secondary regional
comparisons changed materially.

---

## Known limitations

These are stated because they are real, not because they are fatal.

1. **Zero-phase filtering leaks backwards.** `filtfilt` is non-causal, so
   post-onset activity can influence pre-onset samples. Measured leakage into
   the four terminal windows is +0.04%, +0.32%, +1.82% and +3.30% of SD, worst
   in the window nearest onset. The terminal windows are strictly pre-onset in
   *sample index*, not necessarily in *information content*.
2. **Cluster tests search all 29 windows** with no pre-onset restriction. The
   paired left−right cluster begins at window centre −138 ms, whose EEG spans
   −388 to +110 ms. No window in that cluster is entirely pre-movement, so the
   cluster should not be read as establishing a pre-movement onset time. This
   applies to the cluster analysis only. The primary terminal-slope analysis is
   a separate test restricted to the four windows whose samples all precede
   onset (coverage −884 to −14 ms), and is unaffected by this limitation.
3. **The baseline-contrast clusters begin inside the baseline.** Both start at
   window 1, one of the two windows that define the baseline. Because the
   baseline is subtractive, `norm[0] = −norm[1]` exactly, so that left edge is
   algebraically forced rather than evidence.
4. **Scalar pre/post SampEn uses a different tolerance convention** from the
   sliding windows — pre and post epochs each get their own `r`. This affects
   `channel_stats.csv` and the pipeline's internal topomap. It does **not**
   affect the three poster figures, which use only the baseline-normalised
   sliding-window values.
5. **The topography uses MNE's `standard_1020` template**, not the project's
   `data/chanlabels_32channel.xyz`. Channel names and all statistics are
   unaffected; plotted electrode positions could differ slightly.
6. **Multiplicity across cluster families** is not corrected. Three cluster
   families were run, each controlling family-wise error internally.
7. **`sampen.py` deviations** from Richman & Moorman: `ddof=1` for the SD, and
   template counts of `n−m+1` vs `n−m` (matching `antropy`). Both are constant
   for a fixed window length and therefore cancel in baseline subtraction and
   in slopes. They do not cancel between the 250-sample windows and the
   1000-sample pre/post epochs.

---

## Layout

```
src/         analysis pipeline (run via main.py)
figures/     poster figure + statistics scripts
             (_paths.py resolves data locations; qc_check.py and
              overlap_check.py are the automated figure checks)
data/        where the raw dataset and chanlabels_32channel.xyz go;
             neither is redistributed
results/     created by src/main.py; gitignored
output/      created by the figure scripts; gitignored
verify_corrections.py
```

No precomputed intermediate data ships with this repository. `results/` and
`output/` are both absent from a fresh clone and are created by the scripts
that fill them. The two inputs you must supply yourself are the WAY-EEG-GAL
`.mat` files and `data/chanlabels_32channel.xyz` (step 1).

## Provenance note

This repository was assembled from the working project files and then verified
against them. `diff -r src/` against the live working copy found all eleven
source files identical apart from one path comment and one trailing blank line
in `main.py`. That matters most for `src/main.py`, which had been recovered
from an RTF copy; the diff confirms it matches the original. Re-check the two
corrections at any time with `python verify_corrections.py`.

The figure scripts were validated against the published poster outputs before
those reference files were removed from this repository: running the chain in
step 4 against the original validated entropy tables reproduced all five
statistics files byte-for-byte and all four figures identically in PNG, PDF and
SVG (apart from embedded timestamps and matplotlib's per-run element ids).

## Environment

Two dependency files, for two different purposes.

| file | what it is |
|---|---|
| `requirements-lock.txt` | the **exact** versions used to produce every published figure and statistic. This is the only combination that has actually been executed. |
| `requirements.txt` | lower bounds reflecting the APIs the code calls, for an easier install. **These ranges have not been tested** — no claim is made that any version other than the lock file works. |

```bash
pip install -r requirements-lock.txt   # reproduce exactly  (needs Python >=3.12)
pip install -r requirements.txt        # easier install, untested combination
```

Tested interpreter: **Python 3.13.18**, the only one used. The source parses
under the 3.9 through 3.12 grammars (syntax check only, not executed there),
but the locked libraries require 3.12+.

`main.py` writes the library versions it actually ran under to
`results/pipeline_log.txt` at Phase 0, so every run records its own
environment.

## Code development

Python code development and review were assisted by Claude through Cursor.

## Data citation

Luciw, M. D., Jarocka, E., & Edin, B. B. (2014). Multi-channel EEG recordings
during 3,936 grasp and lift trials with varying weight and friction.
*Scientific Data*, 1, 140047. https://doi.org/10.1038/sdata.2014.47

## Licence and attribution

**The code in this repository** — `src/`, `figures/`, `verify_corrections.py` —
is released under the MIT licence. See `LICENSE`.

**The dataset is not part of this repository.** WAY-EEG-GAL is published by
Luciw, Jarocka & Edin under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and must be
downloaded from the original source (see `data/README.md`). Anyone running this
pipeline is using that dataset directly and remains bound by its terms, which
require attribution to the original authors, an indication of any
modifications, and a link to the licence. CC BY 4.0 carries no ShareAlike
term, and CC licences do not permit sublicensing, so the MIT licence above
covers only the code and makes no claim over the dataset or over anything
computed from it.

No processed data, figures or poster materials are distributed here, so no
additional licence is needed for them. If you publish outputs computed with
this code, the attribution above is what CC BY 4.0 asks you to carry:

> Computed from WAY-EEG-GAL — Luciw, M. D., Jarocka, E., & Edin, B. B. (2014),
> *Scientific Data*, 1, 140047, https://doi.org/10.1038/sdata.2014.47 —
> licensed under CC BY 4.0. Modified by bandpass filtering, epoching to
> EMG-defined movement onset, computing sample entropy in 500 ms sliding
> windows with a per-epoch tolerance, baseline-normalising, and aggregating to
> participant level.
