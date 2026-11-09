"""
config.py — Central parameter registry for the WAY-EEG-GAL entropy pipeline.

All analysis parameters, file paths, filter settings, and statistical thresholds
are defined here as named constants. No other module may use magic numbers.
Every parameter used anywhere in the codebase must be imported from this module.
"""

import os

# ---------------------------------------------------------------------------
# Directory paths (relative to project root way-eeg-gal-sample-entropy/)
# ---------------------------------------------------------------------------

DATA_DIR    = 'data/mat'                  # flat directory containing all .mat files
CHAN_FILE   = 'data/chanlabels_32channel.xyz'  # 32-channel electrode coordinates
FIGURES_DIR = 'figures'                   # all figure output files
RESULTS_DIR = 'results'                   # all CSV and log output files

# ---------------------------------------------------------------------------
# Sampling rates
# ---------------------------------------------------------------------------

EEG_SR = 500     # Hz — EEG hardware anti-aliased and digitised at 500 Hz
EMG_SR = 4000    # Hz — EMG digitised at 4000 Hz (8× oversampled vs EEG)

# ---------------------------------------------------------------------------
# EEG bandpass filter parameters
# ---------------------------------------------------------------------------

EEG_BP_LOW  = 1.0    # Hz — lower cutoff; preserves Bereitschaftspotential and
                     # MRCP (0.1–3 Hz band, onset 1–2 s before voluntary movement)
EEG_BP_HIGH = 40.0   # Hz — upper cutoff; removes EMG contamination (>40 Hz)
FILTER_ORDER = 4     # Butterworth filter order, applied zero-phase via filtfilt

# EEG filter padding for scipy.signal.filtfilt (padtype='odd')
# Requirement: padlen >= 3 time constants of the lowest cutoff frequency.
# At 1 Hz: 1 time constant = 1/1 Hz × EEG_SR = 500 samples.
# 3 time constants = 1500 samples = 3.0 s.
# This ensures filter transients are negligible in the pre-onset baseline window.
FILTFILT_PADLEN_EEG = int(3.0 * EEG_SR)    # = 1500 samples

# ---------------------------------------------------------------------------
# EMG bandpass filter parameters
# ---------------------------------------------------------------------------

EMG_BP_LOW  = 20.0   # Hz — lower cutoff; removes movement artefact below 20 Hz
EMG_BP_HIGH = 450.0  # Hz — upper cutoff; below Nyquist at 4000 Hz (max = 2000 Hz)

# EMG filter padding for scipy.signal.filtfilt (padtype='odd')
# At 20 Hz: 1 time constant ≈ 1/20 Hz × EMG_SR = 200 samples.
# 3 time constants = 600 samples. 800 used for an additional safety margin.
FILTFILT_PADLEN_EMG = int(0.2 * EMG_SR)     # = 800 samples

# ---------------------------------------------------------------------------
# Trial timing constants (all in seconds, window-relative)
# ---------------------------------------------------------------------------

LED_ON_TIME    = 2.000   # s — LEDon is always exactly 2.000 s from window start
BASELINE_START = 0.0     # s — window start; pre-cue baseline begins here
BASELINE_END   = 2.000   # s — pre-cue baseline ends at LEDon (equals LED_ON_TIME)

# ---------------------------------------------------------------------------
# EMG onset detection thresholds
# ---------------------------------------------------------------------------

EMG_THRESHOLD_SD  = 3.0    # SD above baseline for onset detection
EMG_SUSTAIN_MS    = 25     # ms — threshold must be sustained continuously
MIN_REACTION_MS   = 100    # ms after LEDon — minimum voluntary motor RT;
                           # onsets within 100 ms of cue are physiologically
                           # implausible and indicate noise crossings
MAX_ONSET_DELAY_MS = 3000  # ms after LEDon — beyond this = missed trial or
                           # EMG electrode failure; trial excluded

# ---------------------------------------------------------------------------
# Epoch extraction parameters
# ---------------------------------------------------------------------------

EPOCH_PRE_MS  = 2000   # ms before EMG onset — full pre-onset epoch
EPOCH_POST_MS = 2000   # ms after  EMG onset — full post-onset epoch

# ---------------------------------------------------------------------------
# Sample entropy (SampEn) parameters
# ---------------------------------------------------------------------------

SAMPEN_ORDER = 2   # embedding dimension m; tolerance r is computed per-epoch
                   # per-channel as 0.2 * std(signal) — see src/sampen.py

# ---------------------------------------------------------------------------
# SampEn tolerance mode  (CORRECTION CF1, 2026-10-25)
# ---------------------------------------------------------------------------
# SampEn is only comparable across segments when the tolerance r is held
# FIXED across those segments. The sliding-window analysis compares 29
# windows drawn from one epoch, so r must be constant across those windows.
#
#   'per_epoch'  (CORRECTED, default)
#       r = 0.2 * std(combined_epoch[:, ch]), computed once per channel and
#       passed to every window of that channel. Scale invariance across
#       trials, channels and participants is preserved; amplitude changes
#       WITHIN the epoch no longer leak into the entropy time course.
#       This is the behaviour the SAMPEN_ORDER comment above and the
#       sample_entropy docstring have always described.
#
#   'per_window' (LEGACY — reproduces the pre-correction pipeline)
#       r = 0.2 * std(window), recomputed for each 250-sample window.
#       A window whose amplitude rises (e.g. an MRCP, or post-onset
#       movement activity) gets a larger r, more template matches, and a
#       lower SampEn with no change in underlying complexity.
#
# Retain 'per_window' ONLY to regenerate the original outputs for
# comparison. It must not be used for reported results.
SAMPEN_TOLERANCE_MODE = 'per_epoch'

# ---------------------------------------------------------------------------
# Sliding window parameters
# ---------------------------------------------------------------------------

SLIDE_WIN_MS  = 500   # ms — duration of each sliding window
SLIDE_STEP_MS = 125   # ms — nominal step between window starts

# NOTE ON SLIDING WINDOW OVERLAP:
# step_samples = int(125/1000 * 500) = 62 samples → true step = 124 ms
# win_samples  = int(500/1000 * 500) = 250 samples
# Overlap fraction = (250 - 62) / 250 = 75.2%
# Adjacent windows share 75 % of their samples, making adjacent SampEn values
# strongly correlated. Standard FDR correction (which assumes independence)
# is therefore INVALID for the sliding-window time series.
# Cluster-based permutation testing (Maris & Oostenveld, 2007) handles
# correlated time points correctly and is used for ALL sliding-window tests.
# Standard FDR is reserved for the 32-channel scalar pre/post comparison only,
# where channels are not temporally overlapping.

# Baseline window contamination threshold:
# The pre-onset baseline is defined as the first two sliding windows of the
# combined epoch (indices 0 and 1). For these windows to fall before LEDon,
# the EMG onset must be sufficiently early in the trial.
# Second window centre, relative to EMG onset:
#   = -EPOCH_PRE_MS/1000 + SLIDE_WIN_MS/2000 + step_samples/EEG_SR
#   = -2.000 + 0.250 + 0.124 = -1.626 s (from onset)
# In window-absolute time: onset_time_s - 1.626
# For this to precede LEDon (2.000 s): onset_time_s < 3.626 s
# 3.624 used for floating-point safety (2-ms margin).
BASELINE_CONTAMINATION_THRESHOLD_S = 3.624   # s from window start

# ---------------------------------------------------------------------------
# Statistical analysis parameters
# ---------------------------------------------------------------------------

N_PERMUTATIONS = 10000   # cluster permutation test iterations (seed=42)
CLUSTER_ALPHA  = 0.05    # cluster-forming threshold (MNE derives t from this)
ALPHA          = 0.05    # family-wise error rate for FDR and cluster tests

# ---------------------------------------------------------------------------
# Dataset dimensions
# ---------------------------------------------------------------------------

N_PARTICIPANTS = 12   # participants P1–P12 — NEVER change this value.
                     # N_PARTICIPANTS controls array allocation inside
                     # run_all_statistics (participant_mean_norm shape and the
                     # cluster permutation test input shape assertion). Setting
                     # it to 3 for a batch run would cause the cluster test
                     # shape assertion to pass incorrectly with 3 rows instead
                     # of 12, producing silently wrong statistics.
N_CHANNELS     = 32   # EEG channels

# ---------------------------------------------------------------------------
# Batch execution control
# ---------------------------------------------------------------------------
# PARTICIPANT_START and PARTICIPANT_END define which participants are processed
# in Phases 1-3 of the current run. They do NOT affect N_PARTICIPANTS, which
# must always remain 12 and must never be changed.
# For a full single run: PARTICIPANT_START=1,  PARTICIPANT_END=12.
# For batch run 1:       PARTICIPANT_START=1,  PARTICIPANT_END=3.
# For batch run 2:       PARTICIPANT_START=4,  PARTICIPANT_END=6.
# For batch run 3:       PARTICIPANT_START=7,  PARTICIPANT_END=9.
# For batch run 4:       PARTICIPANT_START=10, PARTICIPANT_END=12.
PARTICIPANT_START = 1    # inclusive
PARTICIPANT_END   = 12   # inclusive

# Whether to run statistics and figures after entropy computation.
# Set to False for all batch runs 1-4.
# Set to True only for the final consolidation run, after all 12
# per-participant entropy CSV files exist on disk.
# When False, Phases 4 and 5 are skipped entirely — no statistics
# are computed, no group-level figures are generated.
RUN_STATISTICS_AND_FIGURES = True

# ---------------------------------------------------------------------------
# Parallelization
# ---------------------------------------------------------------------------

N_JOBS = -1   # joblib: -1 = use all available CPU cores;
              # set to 1 to disable parallelization for debugging

# ---------------------------------------------------------------------------
# Channel and weight metadata (for reference and validation)
# ---------------------------------------------------------------------------

# EEG channel names in order (index 0–31), confirmed from ws.names.eeg
EEG_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',   'F4',  'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7',   'C3',  'Cz',
    'C4',  'T8',  'TP9', 'CP5', 'CP1',  'CP2', 'CP6',
    'TP10','P7',  'P3',  'Pz',  'P4',   'P8',  'PO9',
    'O1',  'Oz',  'O2',  'PO10'
]

# ------------------------------------------------------------------
# Spatial analysis channel group definitions
# Added for sensorimotor ROI, hemisphere, and lobe analyses.
# All channel names must exactly match entries in EEG_CHANNEL_NAMES.
# ------------------------------------------------------------------

SENSORIMOTOR_ROI_CHANNELS = [
    'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2'
]

LEFT_HEMISPHERE_CHANNELS = [
    'F7', 'F3', 'FC5', 'FC1', 'T7', 'C3', 'CP5', 'CP1',
    'P7', 'P3', 'O1'
]

RIGHT_HEMISPHERE_CHANNELS = [
    'F8', 'F4', 'FC6', 'FC2', 'T8', 'C4', 'CP6', 'CP2',
    'P8', 'P4', 'O2'
]

LOBE_FRONTAL_CHANNELS   = ['F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC6']
LOBE_CENTRAL_CHANNELS   = ['FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2']
LOBE_PARIETAL_CHANNELS  = ['CP5', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8']
LOBE_OCCIPITAL_CHANNELS = ['PO9', 'O1', 'Oz', 'O2', 'PO10']

LOBE_GROUPS = {
    'frontal':   LOBE_FRONTAL_CHANNELS,
    'central':   LOBE_CENTRAL_CHANNELS,
    'parietal':  LOBE_PARIETAL_CHANNELS,
    'occipital': LOBE_OCCIPITAL_CHANNELS,
}

# ---------------------------------------------------------------------------
# Terminal pre-onset window set  (CORRECTION W1, 2026-10-25)
# ---------------------------------------------------------------------------
# Each sliding window spans [centre - 250 ms, centre + 248 ms]. The window is
# NOT symmetric about its reported centre: build_sliding_windows uses
# centre_idx = start + win_samples // 2 = start + 125, which is the 126th
# sample of 250, one sample past the true midpoint.
#   last sample offset = (250 - 1 - 125) x (1000/500) = +248 ms
#
# Under the original rule (centre in [-500, 0)) the set was
# [-386, -262, -138, -14] ms, whose spans are:
#       -386 -> [-636, -138]    0 / 250 samples at t >= 0
#       -262 -> [-512,  -14]    0 / 250
#       -138 -> [-388, +110]   56 / 250   <-- 22.4% post-onset
#        -14 -> [-264, +234]  118 / 250   <-- 47.2% post-onset
# 174 of the 1000 samples (17.4%) behind the "pre-onset" terminal slope were
# recorded at or after EMG onset. Window 14 additionally extends past
# kinematic tHandStart on 73.4% of retained trials.
#
# The corrected set is the four consecutive windows whose LAST SAMPLE still
# precedes onset — same count, same 124 ms spacing, same statistical
# machinery, zero post-onset samples:
#       -634 -> [-884, -386]
#       -510 -> [-760, -262]
#       -386 -> [-636, -138]
#       -262 -> [-512,  -14]
#
# This change follows from the window arithmetic alone. It was NOT selected
# by inspecting results.
TERMINAL_WINDOW_RULE = 'strict_pre_onset'

_TERMINAL_LEGACY_CENTRE   = [-386.0, -262.0, -138.0, -14.0]
_TERMINAL_STRICT_PRE      = [-634.0, -510.0, -386.0, -262.0]

TERMINAL_WINDOW_CENTRES_MS = (
    _TERMINAL_STRICT_PRE if TERMINAL_WINDOW_RULE == 'strict_pre_onset'
    else _TERMINAL_LEGACY_CENTRE
)

# Last-sample offset of a sliding window relative to its reported centre.
# Used by stats.py to assert at runtime that every selected terminal window
# is genuinely pre-onset.
SLIDE_WIN_LAST_SAMPLE_OFFSET_MS = (
    (int(SLIDE_WIN_MS / 1000 * EEG_SR) - 1 - int(SLIDE_WIN_MS / 1000 * EEG_SR) // 2)
    * (1000.0 / EEG_SR)
)   # = +248.0 ms

# EMG channel names in order (index 0–4), confirmed from ws.names.emg
EMG_CHANNEL_NAMES = [
    'Anterior Deltoid',
    'Brachoradial',
    'Flexor Digitorum',
    'Common Extensor Digitorum',
    'First Dorsal Interosseus'
]

# Weight coding: CurW values in AllLifts (confirmed from WEEG_FindEvents.m)
# 660 g is coded as 4, NOT 3. This is a non-obvious feature of the dataset.
WEIGHT_MAP = {1: 165, 2: 330, 4: 660}   # CurW code → grams

# AllLifts column indices (0-indexed) — critical columns used by pipeline
COL_RUN           = 1    # series number (matches WS file series index)
COL_LIFT          = 2    # lift number within series (1-indexed, sorted)
COL_CURW          = 3    # current weight condition (1, 2, or 4)
COL_CURS          = 4    # current surface condition (1=sand, 2=suede, 3=silk)
COL_PREVW         = 5    # previous weight (NaN for first trial in series)
COL_PREVS         = 6    # previous surface (NaN for first trial in series)
COL_START_TIME    = 7    # absolute start time of window (only absolute-time col)
COL_LEDON         = 8    # LEDon time (always 2.000 s, window-relative)
COL_LEDOFF        = 9    # LEDoff time (window-relative, varies by trial)
COL_BLOCK_TYPE    = 10   # 1=weight, 2=friction, 3=mixed
COL_T_FIRST_DIG   = 13   # tFirstDigitTouch (window-relative)
COL_T_LIFTOFF     = 18   # tLiftOff — time object leaves the surface
                          # (window-relative seconds, AllLifts col 18)
COL_T_HANDSTART   = 33   # tHandStart — kinematic onset, SANITY CHECK ONLY,
                         # never used as epoch anchor or onset substitute

# EEG unit conversion factor (applied only in visualization, not in analysis)
EEG_UV_SCALE = 0.1   # raw units are 10 µV; multiply by 0.1 to get µV
