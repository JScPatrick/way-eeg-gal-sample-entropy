"""
main.py — Pipeline orchestration for the WAY-EEG-GAL entropy analysis.

Research question:
    "Does EEG sample entropy change reliably precede EMG-defined onset of
    motor movement?"

Execution phases:
    Phase 0 — Environment setup and directory creation
    Phase 1 — EMG onset detection (all participants, all series)
              → MANUAL CHECKPOINT (inspect QC figures before continuing)
    Phase 2 — EEG preprocessing and epoch extraction
    Phase 3 — Sample entropy computation (parallelized, per-participant caching)
    Phase 4 — Statistical analysis
    Phase 5 — Figure generation
    Phase 6 — Console summary table
    Phase 7 — Pipeline log finalisation

Run from project root:
    python src/main.py

Requirements:
    numpy==2.4.4, scipy==1.17.1, mne (latest), pandas (latest)
    matplotlib (latest), pingouin (latest), scikit-learn==1.8.0
    statsmodels (auto-installed below), joblib (auto-installed below)
NOT USED: antropy — use src/sampen.py for all entropy computation.
"""

# ===========================================================================
# Auto-install optional dependencies before any other import
# ===========================================================================

import sys
import os


def _ensure_package(package_name, import_name=None):
    """Install package if not importable. Called before downstream imports."""
    import importlib
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        import subprocess
        print(f'  Auto-installing {package_name} ...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               package_name])


_ensure_package('statsmodels', 'statsmodels.stats.multitest')
_ensure_package('joblib')

# ===========================================================================
# Standard imports (after auto-install)
# ===========================================================================

import time
import datetime
import traceback

import numpy as np
import scipy.io
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; must be set before pyplot
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup: ensure src/ is on sys.path regardless of invocation directory.
# All config-relative paths (data/mat, figures/, results/) are relative to
# the project root. We compute the project root as the parent of this file's
# directory and change the working directory to it at startup.
# ---------------------------------------------------------------------------
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Change to project root so that all relative paths in config.py resolve
# correctly regardless of where the user invoked python from.
os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Pipeline module imports
# ---------------------------------------------------------------------------
from config import (
    DATA_DIR, FIGURES_DIR, RESULTS_DIR, CHAN_FILE,
    EEG_SR, EMG_SR,
    N_PARTICIPANTS, N_CHANNELS,
    PARTICIPANT_START, PARTICIPANT_END, RUN_STATISTICS_AND_FIGURES,
    BASELINE_START, BASELINE_END, LED_ON_TIME,
    EMG_THRESHOLD_SD,
    EPOCH_PRE_MS, EPOCH_POST_MS,
    EEG_CHANNEL_NAMES, EMG_CHANNEL_NAMES, WEIGHT_MAP,
    COL_RUN, COL_LIFT, COL_CURW, COL_CURS, COL_PREVW, COL_PREVS,
    COL_T_HANDSTART, COL_T_FIRST_DIG, COL_T_LIFTOFF, COL_BLOCK_TYPE, COL_LEDON,
    EEG_UV_SCALE, N_JOBS,
)
from load_data import (load_ws_series, load_allifts,
                        get_series_allifts_rows, get_trial_allifts)
from emg_onset import preprocess_emg, detect_onset, check_exclusion
from eeg_preprocess import preprocess_eeg
from epoching import extract_epochs, build_sliding_windows
from entropy import _compute_one_trial, compute_participant_entropy
from stats import run_all_statistics
from visualization import (
    generate_all_figures,
    plot_qc_emg_onset_participant,
    plot_qc_erp_sanity_check,
)

from joblib import Parallel, delayed


# ===========================================================================
# Logging helper
# ===========================================================================

_log_fh = None   # file handle opened in Phase 0


def log(msg):
    """Write timestamped message to pipeline_log.txt and print to console."""
    ts  = datetime.datetime.now().strftime('%H:%M:%S')
    out = f'[{ts}] {msg}'
    print(out)
    if _log_fh is not None:
        _log_fh.write(out + '\n')
        _log_fh.flush()


# ===========================================================================
# Phase 0 — Environment setup
# ===========================================================================

def phase0_setup():
    """Create output directories, open log file, print package versions."""
    global _log_fh

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_path = os.path.join(RESULTS_DIR, 'pipeline_log.txt')
    _log_fh  = open(log_path, 'a', encoding='utf-8')

    run_ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log('=' * 70)
    log(f'WAY-EEG-GAL Entropy Pipeline — Run started: {run_ts}')
    log(f'Project root: {_PROJECT_ROOT}')
    log('=' * 70)

    # Print and log package versions for reproducibility.
    import numpy, scipy, pandas, matplotlib, mne, pingouin
    versions = {
        'numpy':      numpy.__version__,
        'scipy':      scipy.__version__,
        'pandas':     pandas.__version__,
        'matplotlib': matplotlib.__version__,
        'mne':        mne.__version__,
        'pingouin':   pingouin.__version__,
    }
    try:
        import statsmodels
        versions['statsmodels'] = statsmodels.__version__
    except ImportError:
        versions['statsmodels'] = 'not found'
    try:
        import joblib
        versions['joblib'] = joblib.__version__
    except ImportError:
        versions['joblib'] = 'not found'
    try:
        import sklearn
        versions['scikit-learn'] = sklearn.__version__
    except ImportError:
        versions['scikit-learn'] = 'not found'

    log('Package versions:')
    for pkg, ver in versions.items():
        log(f'  {pkg}: {ver}')

    log(f'N_JOBS={N_JOBS} | N_PARTICIPANTS={N_PARTICIPANTS} '
        f'| N_CHANNELS={N_CHANNELS}')


# ===========================================================================
# Phase 1 — EMG onset detection
# ===========================================================================

def phase1_onset_detection():
    """
    Iterate all participants × series × trials, detect EMG onset, apply
    exclusion criteria, and save results/onset_detection_results.csv.

    Also accumulates:
      - per-participant QC data for plot_qc_emg_onset_participant
      - per-trial digit-contact times for Figure 3 annotation

    Returns
    -------
    onset_records : list of dict
        One dict per trial (valid and excluded). Keys include:
        participant, series, trial_in_series, allifts_row_idx,
        block_type, weight, surface, emg_onset_s, emg_onset_ms_post_led,
        emg_onset_channel, eeg_onset_idx, tHandStart_s, onset_lag_ms,
        excluded, exclusion_reason, threshold, tFirstDigitTouch_s.
    """
    log('--- PHASE 1: EMG onset detection ---')
    onset_records = []

    # Per-participant QC data for Figure B.
    # qc_data[pid] = {'valid_records': [], 'emg_env': [], 'emg_t': [],
    #                 'channel_names': [...]}
    qc_data = {pid: {'valid_records': [], 'emg_env': [], 'emg_t': [],
                     'channel_names': EMG_CHANNEL_NAMES}
               for pid in range(PARTICIPANT_START, PARTICIPANT_END + 1)}

    t0_phase = time.time()

    for participant in range(PARTICIPANT_START, PARTICIPANT_END + 1):
        t0_part = time.time()
        log(f'  P{participant}: loading AllLifts ...')

        allifts, col_names, found = load_allifts(participant, DATA_DIR)
        if not found:
            log(f'  P{participant}: AllLifts file not found — skipping.')
            continue

        part_total = 0
        part_excl  = 0
        excl_reasons = {}

        for series in range(1, 10):
            ws, trials, found = load_ws_series(participant, series, DATA_DIR)
            if not found:
                # Missing series is expected (not all participants have 9 series).
                continue

            series_rows, valid, msg = get_series_allifts_rows(allifts, series)
            if not valid:
                log(f'  P{participant} S{series}: AllLifts validation failed: {msg}')
                continue

            if len(trials) != len(series_rows):
                log(f'  P{participant} S{series}: trial count mismatch — '
                    f'ws.win={len(trials)}, AllLifts rows={len(series_rows)}. '
                    f'Skipping series.')
                continue

            for trial_idx, trial in enumerate(trials):
                part_total += 1
                allifts_row = get_trial_allifts(series_rows, trial_idx)

                # Extract metadata from AllLifts row.
                block_type = int(allifts_row[COL_BLOCK_TYPE]) \
                    if not np.isnan(allifts_row[COL_BLOCK_TYPE]) else -1
                weight     = int(allifts_row[COL_CURW]) \
                    if not np.isnan(allifts_row[COL_CURW]) else -1
                surface    = int(allifts_row[COL_CURS]) \
                    if not np.isnan(allifts_row[COL_CURS]) else -1
                tHandStart = float(allifts_row[COL_T_HANDSTART])
                tFirstDig  = float(allifts_row[COL_T_FIRST_DIG]) \
                    if not np.isnan(allifts_row[COL_T_FIRST_DIG]) else np.nan

                # Check for NaN in critical AllLifts fields (excluding known NaNs).
                non_nan_cols = [COL_CURW, COL_CURS, COL_BLOCK_TYPE]
                has_nan = any(np.isnan(allifts_row[c]) for c in non_nan_cols)
                if has_nan:
                    reason = 'nan_in_allifts'
                    log(f'  P{participant} S{series} T{trial_idx}: '
                        f'NaN in critical AllLifts columns — excluding.')
                    onset_records.append(_make_onset_record(
                        participant, series, trial_idx,
                        allifts_row, block_type, weight, surface,
                        None, None, None, tHandStart, tFirstDig,
                        True, reason, np.nan
                    ))
                    part_excl += 1
                    excl_reasons[reason] = excl_reasons.get(reason, 0) + 1
                    continue

                # EMG preprocessing and onset detection.
                onset_s = onset_ch = eeg_idx = None
                threshold = np.nan
                reason    = None

                try:
                    smoothed = preprocess_emg(trial.emg, trial.emg_t)

                    # Compute threshold (same formula as detect_onset) so it
                    # can be stored in onset_records and shown in QC Figure B.
                    bl_mask   = ((trial.emg_t >= BASELINE_START) &
                                 (trial.emg_t  < BASELINE_END))
                    if np.sum(bl_mask) >= 100:
                        bl_data   = smoothed[bl_mask, :]
                        threshold = (float(np.mean(bl_data)) +
                                     EMG_THRESHOLD_SD * float(np.std(bl_data)))

                    # EMG channel names from ws.names.emg (list of 5 strings).
                    ch_names = list(ws.names.emg)

                    onset_s, onset_ch, eeg_idx = detect_onset(
                        smoothed, trial.emg_t, trial.eeg_t, ch_names
                    )
                    reason = check_exclusion(onset_s, eeg_idx,
                                             allifts_row, onset_s is not None)

                    # Accumulate QC data for valid trials.
                    if reason is None:
                        latency_ms = (onset_s - LED_ON_TIME) * 1000.0
                        qc_data[participant]['valid_records'].append({
                            'trial_in_series':      trial_idx,
                            'weight':               weight,
                            'emg_onset_s':          float(onset_s),
                            'emg_onset_ms_post_led': float(latency_ms),
                            'tHandStart_s':         float(tHandStart),
                            'threshold':            float(threshold),
                        })
                        qc_data[participant]['emg_env'].append(smoothed)
                        qc_data[participant]['emg_t'].append(
                            np.array(trial.emg_t))

                except ValueError as e:
                    reason = 'emg_preprocessing_error'
                    log(f'  P{participant} S{series} T{trial_idx}: '
                        f'EMG preprocessing error: {e}')
                    smoothed = None

                onset_ms_post = ((onset_s - LED_ON_TIME) * 1000.0
                                 if onset_s is not None else np.nan)
                onset_lag_ms  = ((onset_s - tHandStart) * 1000.0
                                 if (onset_s is not None and
                                     not np.isnan(tHandStart))
                                 else np.nan)

                excluded = reason is not None
                if excluded:
                    part_excl += 1
                    excl_reasons[reason] = excl_reasons.get(reason, 0) + 1

                onset_records.append(_make_onset_record(
                    participant, series, trial_idx,
                    allifts_row, block_type, weight, surface,
                    onset_s, onset_ch, eeg_idx,
                    tHandStart, tFirstDig,
                    excluded, reason, threshold,
                    onset_ms_post, onset_lag_ms
                ))

        elapsed_part = time.time() - t0_part
        log(f'  P{participant}: {part_total} trials, '
            f'{part_excl} excluded ({100*part_excl/max(part_total,1):.1f}%), '
            f'reasons: {excl_reasons} | {elapsed_part:.1f}s')

    # ------------------------------------------------------------------
    # Save onset_detection_results.csv immediately.
    # ------------------------------------------------------------------
    onset_df = pd.DataFrame(onset_records)
    onset_path = os.path.join(RESULTS_DIR, 'onset_detection_results.csv')
    onset_df.to_csv(onset_path, index=False)
    log(f'Saved: {onset_path}  ({len(onset_df)} rows)')

    # Save excluded_trials.csv.
    excl_df   = onset_df[onset_df['excluded']].copy()
    excl_path = os.path.join(RESULTS_DIR, 'excluded_trials.csv')
    excl_df[['participant', 'series', 'trial_in_series',
             'exclusion_reason']].to_csv(excl_path, index=False)
    log(f'Saved: {excl_path}  ({len(excl_df)} excluded trials)')

    # ------------------------------------------------------------------
    # Generate QC Figure B per participant.
    # ------------------------------------------------------------------
    log('Generating QC Figure B (EMG onset) ...')
    for pid in range(PARTICIPANT_START, PARTICIPANT_END + 1):
        qd = qc_data[pid]
        if not qd['valid_records']:
            log(f'  P{pid}: no valid trials for QC EMG plot.')
            continue
        try:
            plot_qc_emg_onset_participant(
                participant_id=pid,
                trial_records=qd['valid_records'],
                emg_data_list=qd['emg_env'],
                emg_t_list=qd['emg_t'],
                smoothed_list=qd['emg_env'],   # alias
                channel_names=qd['channel_names'],
            )
        except Exception as e:
            log(f'  P{pid}: QC Figure B failed: {e}')
            traceback.print_exc()

    elapsed_phase = time.time() - t0_phase
    log(f'--- PHASE 1 complete: {elapsed_phase/60:.1f} min ---')
    return onset_records


def _make_onset_record(participant, series, trial_idx, allifts_row,
                        block_type, weight, surface,
                        onset_s, onset_ch, eeg_idx,
                        tHandStart, tFirstDig,
                        excluded, reason, threshold,
                        onset_ms_post=np.nan, onset_lag_ms=np.nan):
    """Build one onset record dict from per-trial data."""
    allifts_row_idx = int(allifts_row[COL_LIFT]) if allifts_row is not None \
        else -1
    tLiftOff_s = float(allifts_row[COL_T_LIFTOFF]) \
        if not np.isnan(allifts_row[COL_T_LIFTOFF]) else np.nan
    return {
        'participant':          participant,
        'series':               series,
        'trial_in_series':      trial_idx,
        'allifts_row_idx':      allifts_row_idx,
        'block_type':           block_type,
        'weight':               weight,
        'surface':              surface,
        'emg_onset_s':          float(onset_s) if onset_s is not None
                                else np.nan,
        'emg_onset_ms_post_led': float(onset_ms_post),
        'emg_onset_channel':    onset_ch if onset_ch is not None else '',
        'eeg_onset_idx':        int(eeg_idx) if eeg_idx is not None else -1,
        'tHandStart_s':         float(tHandStart),
        'tFirstDigitTouch_s':   float(tFirstDig),
        'tLiftOff_s':           tLiftOff_s,
        'onset_lag_ms':         float(onset_lag_ms),
        'threshold':            float(threshold),
        'excluded':             bool(excluded),
        'exclusion_reason':     reason if reason is not None else '',
    }


# ===========================================================================
# Manual checkpoint
# ===========================================================================

def phase1_checkpoint():
    """
    Print inspection instructions and block until user types 'continue'.
    Exits cleanly in non-interactive sessions (e.g. remote/scheduled jobs).
    """
    print('\n' + '=' * 70)
    print('PHASE 1 COMPLETE — MANUAL INSPECTION REQUIRED')
    print('=' * 70)
    print('Before continuing, inspect:')
    print('  1. figures/qc_emg_onset_P*.png — verify red onset lines fall on')
    print('     real muscle bursts, not noise spikes.')
    print('     If onset markers look wrong: stop, adjust EMG_THRESHOLD_SD')
    print('     in config.py, and re-run from Phase 1.')
    print('  2. results/onset_detection_results.csv — check:')
    print('     - Mean onset latency per participant (expect 200–800 ms)')
    print('     - Exclusion rate (expect <20 % for clean data)')
    print('     - onset_lag_ms column (expect mostly negative: EMG before kin)')
    print(f'\nEstimated entropy computation time: 3–9 hours (N_JOBS={N_JOBS})')
    print('Per-participant results saved as computed — safe to interrupt.')
    print('=' * 70)

    try:
        user_input = input('\nType \'continue\' to begin entropy computation, '
                           'or anything else to exit: ')
        if user_input.strip().lower() != 'continue':
            log('User chose to exit before entropy computation.')
            print('Exiting. Re-run main.py to resume from entropy computation.')
            print('Completed per-participant entropy files will be skipped.')
            if _log_fh:
                _log_fh.close()
            sys.exit(0)
    except EOFError:
        # Non-interactive session: stdin is not a terminal.
        log('Non-interactive session detected — exiting before entropy phase.')
        print('Non-interactive session detected. Exiting before entropy phase.')
        print('Re-run interactively to confirm QC and proceed.')
        if _log_fh:
            _log_fh.close()
        sys.exit(0)

    log('User confirmed QC — proceeding to entropy computation.')
    print('\nStarting entropy computation ...\n')


# ===========================================================================
# Phase 2 — EEG preprocessing and epoch extraction
# ===========================================================================

def phase2_preprocess_epochs(onset_records):
    """
    For each valid trial, load EEG, bandpass filter, CAR, extract epochs,
    and build sliding windows.

    Also accumulates per-participant ERP data for QC Figure A.

    Parameters
    ----------
    onset_records : list of dict
        Full onset detection output from phase1_onset_detection.

    Returns
    -------
    trial_data_by_participant : dict
        Keys: participant id (int 1–12).
        Values: list of dicts, one per valid trial, with keys:
            'pre_epoch', 'post_epoch', 'windows',
            'combined_t_ms', 'record' (the onset record dict).
    centres_ms : list of float
        Sliding-window centre times in ms (identical for all trials).
    erp_accum : dict
        Per-participant ERP accumulation for QC Figure A.
        Keys: participant id.
        Values: {'sum': ndarray (2000,32), 'count': int}
    digit_pairs : list of tuple (tFirstDig_s, onset_s)
        For computing mean digit contact time in Figure 3.
    """
    log('--- PHASE 2: EEG preprocessing and epoch extraction ---')
    t0 = time.time()

    # Build lookup from (participant, series) → onset_records for valid trials.
    valid_records = [r for r in onset_records if not r['excluded']]
    log(f'  Valid trials for preprocessing: {len(valid_records)}')

    # Cache WS files to avoid reloading the same file for every trial.
    # Key: (participant, series) → list of trial structs.
    ws_cache   = {}
    ws_names_cache = {}   # (participant, series) → ws.names

    trial_data_by_participant = {pid: [] for pid in range(PARTICIPANT_START, PARTICIPANT_END + 1)}
    erp_accum  = {pid: {'sum': np.zeros((2000, N_CHANNELS)),
                         'count': 0}
                  for pid in range(PARTICIPANT_START, PARTICIPANT_END + 1)}
    digit_pairs   = []
    liftoff_pairs = []
    centres_ms    = None   # set from first valid trial

    pre_samples  = int(EPOCH_PRE_MS  / 1000 * EEG_SR)   # = 1000
    post_samples = int(EPOCH_POST_MS / 1000 * EEG_SR)   # = 1000

    for i, rec in enumerate(valid_records):
        pid    = rec['participant']
        series = rec['series']
        t_idx  = rec['trial_in_series']
        key    = (pid, series)

        # Load (and cache) the WS file for this (participant, series).
        if key not in ws_cache:
            ws, trials, found = load_ws_series(pid, series, DATA_DIR)
            if not found:
                log(f'  P{pid} S{series}: WS file missing in Phase 2 '
                    f'(was present in Phase 1) — skipping {len([r for r in valid_records if r["participant"]==pid and r["series"]==series])} trials.')
                ws_cache[key]       = None
                ws_names_cache[key] = None
            else:
                ws_cache[key]       = trials
                ws_names_cache[key] = ws.names

        trials = ws_cache[key]
        if trials is None:
            continue   # WS file disappeared — skip gracefully

        trial = trials[t_idx]

        try:
            eeg_filt = preprocess_eeg(trial.eeg, trial.eeg_t)
        except ValueError as e:
            log(f'  P{pid} S{series} T{t_idx}: EEG preprocess error: {e}')
            continue

        eeg_onset_idx = rec['eeg_onset_idx']

        # Guard: eeg_onset_idx must be valid (check_exclusion already verified
        # eeg_onset_idx >= pre_samples, but re-check for safety).
        if (eeg_onset_idx < pre_samples or
                eeg_onset_idx + post_samples > len(trial.eeg_t)):
            log(f'  P{pid} S{series} T{t_idx}: onset index {eeg_onset_idx} '
                f'out of bounds for epoch extraction — skipping.')
            continue

        try:
            (pre_epoch, post_epoch, combined_epoch,
             pre_t_ms, post_t_ms, combined_t_ms) = extract_epochs(
                 eeg_filt, eeg_onset_idx
             )
        except AssertionError as e:
            log(f'  P{pid} S{series} T{t_idx}: epoch assertion failed: {e}')
            continue

        try:
            windows, w_centres_ms, epoch_labels = build_sliding_windows(
                combined_epoch, combined_t_ms
            )
        except AssertionError as e:
            log(f'  P{pid} S{series} T{t_idx}: window assertion failed: {e}')
            continue

        # Record centres_ms from the first valid trial (same for all trials).
        if centres_ms is None:
            centres_ms = w_centres_ms
            # Persist exact float64 window centre times to disk so that the
            # consolidation run can load them without CSV round-trip error.
            # Only save if the file does not already exist so that later batch
            # runs do not overwrite the values written by batch run 1.
            centres_ms_path = os.path.join(RESULTS_DIR, 'centres_ms.npy')
            if not os.path.exists(centres_ms_path):
                np.save(centres_ms_path, np.array(centres_ms))
                log(f'Saved centres_ms to {centres_ms_path} '
                    f'({len(centres_ms)} windows)')

        trial_data_by_participant[pid].append({
            'pre_epoch':     pre_epoch,
            'post_epoch':    post_epoch,
            'windows':       windows,
            'combined_t_ms': combined_t_ms,
            'record':        rec,
        })

        # Accumulate ERP (raw units; EEG is still in 10-µV units).
        erp_accum[pid]['sum']   += combined_epoch
        erp_accum[pid]['count'] += 1

        # Accumulate digit contact pairs for Figure 3 annotation.
        t_first_dig = rec.get('tFirstDigitTouch_s', np.nan)
        onset_s     = rec.get('emg_onset_s', np.nan)
        if not np.isnan(t_first_dig) and not np.isnan(onset_s):
            digit_pairs.append((t_first_dig, onset_s))

        # Accumulate liftoff pairs — three-element tuples (participant_id,
        # tLiftOff_s, emg_onset_s). participant_id is stored in each tuple
        # so per-participant means can be computed by grouping on it directly,
        # without any zip alignment with other lists.
        t_liftoff = rec.get('tLiftOff_s', np.nan)
        p_id      = rec.get('participant', np.nan)
        if not np.isnan(t_liftoff) and not np.isnan(onset_s) \
                and not np.isnan(p_id):
            liftoff_pairs.append((int(p_id), t_liftoff, onset_s))

        if (i + 1) % 100 == 0:
            log(f'  Phase 2 progress: {i+1}/{len(valid_records)} trials.')

    elapsed = time.time() - t0
    log(f'--- PHASE 2 complete: {elapsed/60:.1f} min ---')

    for pid in range(PARTICIPANT_START, PARTICIPANT_END + 1):
        n = len(trial_data_by_participant[pid])
        log(f'  P{pid}: {n} trials ready for entropy computation.')

    return trial_data_by_participant, centres_ms, erp_accum, digit_pairs, liftoff_pairs


# ===========================================================================
# Phase 3 — Entropy computation
# ===========================================================================

def phase3_entropy(trial_data_by_participant, centres_ms):
    """
    Compute SampEn per participant, saving per-participant CSVs immediately.
    Skips participants whose CSV already exists (restart-safe).

    Parameters
    ----------
    trial_data_by_participant : dict, from phase2_preprocess_epochs
    centres_ms : list of float, window centre times in ms

    Returns
    -------
    trial_df : pandas.DataFrame  (trial_level_entropy.csv)
    sw_df    : pandas.DataFrame  (sliding_window_entropy.csv)
    """
    log('--- PHASE 3: entropy computation ---')
    t0 = time.time()

    N_windows  = len(centres_ms)
    pre_cols   = [f'pre_sampen_ch{ch}'  for ch in range(N_CHANNELS)]
    post_cols  = [f'post_sampen_ch{ch}' for ch in range(N_CHANNELS)]
    diff_cols  = [f'diff_sampen_ch{ch}' for ch in range(N_CHANNELS)]
    norm_cols  = [f'sampen_norm_ch{ch}' for ch in range(N_CHANNELS)]
    sw_cols    = [f'sampen_ch{ch}'      for ch in range(N_CHANNELS)]

    all_trial_rows = []
    all_sw_rows    = []

    for pid in range(PARTICIPANT_START, PARTICIPANT_END + 1):
        per_file = os.path.join(RESULTS_DIR, f'trial_level_entropy_P{pid}.csv')

        if os.path.exists(per_file):
            log(f'  P{pid}: entropy file found — loading (skip recompute).')
            p_trial_df = pd.read_csv(per_file)
            all_trial_rows.append(p_trial_df)

            # Rebuild sliding-window rows from sw file if it exists, else warn.
            sw_per_file = os.path.join(
                RESULTS_DIR, f'sliding_window_entropy_P{pid}.csv')
            if os.path.exists(sw_per_file):
                all_sw_rows.append(pd.read_csv(sw_per_file))
            else:
                log(f'  P{pid}: per-participant SW file missing — '
                    f'SW entropy will be incomplete for this participant.')
            continue

        td_list = trial_data_by_participant[pid]
        if not td_list:
            log(f'  P{pid}: no valid trials — skipping entropy.')
            continue

        n_trials = len(td_list)
        log(f'  P{pid}: computing entropy for {n_trials} trials ...')
        t0_p = time.time()

        # Build argument list for joblib: (pre_epoch, post_epoch, windows).
        args = [(td['pre_epoch'], td['post_epoch'], td['windows'])
                for td in td_list]

        results = Parallel(n_jobs=N_JOBS)(
            delayed(_compute_one_trial)(a) for a in args
        )

        # Print progress summary (post-hoc after parallel completion).
        for i in range(0, n_trials, 10):
            upper = min(i + 10, n_trials)
            log(f'  P{pid}: trials {i+1}–{upper}/{n_trials} complete.')

        # ------------------------------------------------------------------
        # Assemble per-participant trial-level and sliding-window CSVs.
        # ------------------------------------------------------------------
        trial_rows = []
        sw_rows    = []

        for trial_i, (td, result) in enumerate(zip(td_list, results)):
            pre_se, post_se, sw_se, sw_norm = result
            rec = td['record']

            nan_pre  = int(np.sum(np.isnan(pre_se)))
            nan_post = int(np.sum(np.isnan(post_se)))

            pre_mean  = float(np.nanmean(pre_se))
            post_mean = float(np.nanmean(post_se))
            diff_mean = pre_mean - post_mean

            row = {
                'participant':         rec['participant'],
                'series':              rec['series'],
                'trial_in_series':     rec['trial_in_series'],
                'block_type':          rec['block_type'],
                'weight':              rec['weight'],
                'surface':             rec['surface'],
                'emg_onset_ms_post_led': rec['emg_onset_ms_post_led'],
                'pre_sampen_mean':     pre_mean,
                'post_sampen_mean':    post_mean,
                'diff_sampen_mean':    diff_mean,
                'direction_pre_vs_post': ('higher_pre' if diff_mean > 0
                                          else 'lower_pre'),
                'nan_count_pre':       nan_pre,
                'nan_count_post':      nan_post,
            }
            # Per-channel pre/post/diff values.
            for ch in range(N_CHANNELS):
                row[pre_cols[ch]]  = float(pre_se[ch])
                row[post_cols[ch]] = float(post_se[ch])
                row[diff_cols[ch]] = float(pre_se[ch] - post_se[ch]) \
                    if not (np.isnan(pre_se[ch]) or np.isnan(post_se[ch])) \
                    else np.nan
            trial_rows.append(row)

            # Sliding-window rows.
            for w_idx in range(sw_se.shape[0]):
                sw_row = {
                    'participant':      rec['participant'],
                    'series':           rec['series'],
                    'trial_in_series':  rec['trial_in_series'],
                    'weight':           rec['weight'],
                    'surface':          rec['surface'],
                    'window_index':     w_idx,
                    'window_centre_ms': centres_ms[w_idx],
                    'epoch':            'pre' if centres_ms[w_idx] < 0
                                        else 'post',
                    'sampen_mean':      float(np.nanmean(sw_se[w_idx, :])),
                    'sampen_norm_mean': float(np.nanmean(sw_norm[w_idx, :])),
                }
                for ch in range(N_CHANNELS):
                    sw_row[sw_cols[ch]]  = float(sw_se[w_idx, ch])
                    sw_row[norm_cols[ch]] = float(sw_norm[w_idx, ch])
                sw_rows.append(sw_row)

        p_trial_df = pd.DataFrame(trial_rows)
        p_trial_df.to_csv(per_file, index=False)
        log(f'  P{pid}: saved {per_file}  ({len(p_trial_df)} trial rows)')

        p_sw_df = pd.DataFrame(sw_rows)
        sw_per = os.path.join(RESULTS_DIR, f'sliding_window_entropy_P{pid}.csv')
        p_sw_df.to_csv(sw_per, index=False)
        log(f'  P{pid}: saved {sw_per}  ({len(p_sw_df)} SW rows)')

        elapsed_p = time.time() - t0_p
        log(f'  P{pid}: entropy done in {elapsed_p/60:.1f} min.')

        all_trial_rows.append(p_trial_df)
        all_sw_rows.append(p_sw_df)

    # ------------------------------------------------------------------
    # Concatenate all participants.
    # ------------------------------------------------------------------
    if not all_trial_rows:
        log('ERROR: no entropy data collected — cannot proceed.')
        raise RuntimeError('No entropy data collected in Phase 3.')

    trial_df = pd.concat(all_trial_rows, ignore_index=True)
    trial_path = os.path.join(RESULTS_DIR, 'trial_level_entropy.csv')
    trial_df.to_csv(trial_path, index=False)
    log(f'Saved: {trial_path}  ({len(trial_df)} rows)')

    sw_df = pd.concat(all_sw_rows, ignore_index=True) if all_sw_rows \
        else pd.DataFrame()
    sw_path = os.path.join(RESULTS_DIR, 'sliding_window_entropy.csv')
    sw_df.to_csv(sw_path, index=False)
    log(f'Saved: {sw_path}  ({len(sw_df)} rows)')

    elapsed = time.time() - t0
    log(f'--- PHASE 3 complete: {elapsed/60:.1f} min ---')
    return trial_df, sw_df


# ===========================================================================
# CSV loading helper (used by Phase 4 in consolidation runs)
# ===========================================================================

def _load_or_assemble_csv(combined_path, per_participant_template, label):
    """
    Return a DataFrame from disk, assembling it from per-participant files
    if the combined file does not yet exist.

    This makes Phase 4 robust to consolidation runs where Phase 3 saved all
    12 per-participant files successfully but was interrupted before writing
    the combined file (e.g. because of a disk-full error or a SIGKILL after
    the last participant completed).

    Parameters
    ----------
    combined_path : str
        Expected path of the combined CSV (e.g. results/trial_level_entropy.csv).
    per_participant_template : str
        Path template with a ``{pid}`` placeholder for each participant file
        (e.g. 'results/trial_level_entropy_P{pid}.csv').
    label : str
        Human-readable name used in log messages and error text.

    Returns
    -------
    df : pandas.DataFrame
        Contents of the combined file.

    Raises
    ------
    RuntimeError
        If the combined file is absent AND any per-participant file is also
        absent.  A partial set of per-participant files would produce a
        silently incomplete combined DataFrame, so we fail loudly instead.
    """
    if os.path.exists(combined_path):
        log(f'  Loading {label} from {combined_path}')
        return pd.read_csv(combined_path)

    log(f'  {combined_path} not found — assembling from per-participant files.')

    # Check all files exist before reading any, so we fail atomically with a
    # clear list of every missing file rather than stopping at the first one.
    missing = []
    for pid in sorted(range(1, N_PARTICIPANTS + 1)):
        per_path = per_participant_template.format(pid=pid)
        if not os.path.exists(per_path):
            missing.append(per_path)

    if missing:
        missing_str = '\n  '.join(missing)
        raise RuntimeError(
            f'Cannot assemble {label}: the following per-participant files '
            f'are missing:\n  {missing_str}\n'
            f'Re-run Phase 3 to regenerate them.'
        )

    # sorted() guarantees ascending participant order, which determines row
    # order in the combined DataFrame and therefore participant ordering in
    # participant_mean_norm inside run_all_statistics.  Wrong ordering would
    # silently scramble participant-level results without producing any error.
    frames = []
    for pid in sorted(range(1, N_PARTICIPANTS + 1)):
        per_path = per_participant_template.format(pid=pid)
        frames.append(pd.read_csv(per_path))
        log(f'  Loaded: {per_path}')

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    log(f'  Assembled and saved: {combined_path}  ({len(combined)} rows)')
    return combined


# ===========================================================================
# Phase 4 — Statistics
# ===========================================================================

def phase4_statistics(centres_ms, trial_df=None, sw_df=None):
    """
    Run all statistical analyses and save result CSVs.

    Parameters
    ----------
    centres_ms : list of float or None
        Sliding-window centre times in ms.  When None (consolidation / stats-
        only run), they are derived from the ``window_centre_ms`` column of
        the loaded sw_df.
    trial_df : pandas.DataFrame or None
        Trial-level entropy data.  If None, loaded (or assembled) from disk.
    sw_df : pandas.DataFrame or None
        Sliding-window entropy data.  If None, loaded (or assembled) from disk.

    Returns
    -------
    stats_output : dict
        Output of stats.run_all_statistics.

    Notes
    -----
    Loading strategy for both DataFrames:
        1. If the combined CSV exists  → read it directly.
        2. If it does not exist        → assemble it by concatenating all 12
           per-participant CSVs, save the combined file, then proceed.
        3. If any per-participant file is also missing → RuntimeError with the
           full list of missing files so the user knows exactly what to re-run.

    This makes Phase 4 safe to call both from a fresh full-pipeline run
    (trial_df / sw_df passed in from Phase 3) and from a consolidation run
    where Phases 1-3 are skipped but all 12 per-participant files exist.
    """
    log('--- PHASE 4: statistical analysis ---')
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load trial-level entropy data if not supplied by Phase 3.
    # ------------------------------------------------------------------
    if trial_df is None:
        trial_df = _load_or_assemble_csv(
            combined_path=os.path.join(RESULTS_DIR,
                                       'trial_level_entropy.csv'),
            per_participant_template=os.path.join(RESULTS_DIR,
                                                  'trial_level_entropy_P{pid}.csv'),
            label='trial_level_entropy',
        )

    # ------------------------------------------------------------------
    # Load sliding-window entropy data if not supplied by Phase 3.
    # ------------------------------------------------------------------
    if sw_df is None:
        sw_df = _load_or_assemble_csv(
            combined_path=os.path.join(RESULTS_DIR,
                                       'sliding_window_entropy.csv'),
            per_participant_template=os.path.join(RESULTS_DIR,
                                                  'sliding_window_entropy_P{pid}.csv'),
            label='sliding_window_entropy',
        )

    # ------------------------------------------------------------------
    # Load centres_ms from disk if not supplied.
    # The .npy file is written by Phase 2 of the first batch run that
    # produces valid sliding windows.  It preserves exact float64 values;
    # deriving centres_ms from sw_df['window_centre_ms'].unique() would
    # introduce CSV round-trip rounding that can alter window centre times
    # by small but non-zero amounts, corrupting participant_mean_norm
    # array construction inside run_all_statistics.
    # ------------------------------------------------------------------
    if centres_ms is None:
        centres_ms_path = os.path.join(RESULTS_DIR, 'centres_ms.npy')
        if not os.path.exists(centres_ms_path):
            raise RuntimeError(
                'centres_ms.npy not found in results/. '
                'Run at least one batch (Phases 1-3) before the '
                'consolidation run so that centres_ms.npy is saved to disk.'
            )
        centres_ms = list(np.load(centres_ms_path))
        log(f'  Loaded centres_ms from {centres_ms_path} '
            f'({len(centres_ms)} windows)')

    log(f'  trial_df: {len(trial_df)} rows | '
        f'sw_df: {len(sw_df)} rows | '
        f'N_windows: {len(centres_ms)}')

    stats_output = run_all_statistics(trial_df, sw_df, centres_ms, log)

    # ------------------------------------------------------------------
    # Save participant_summary.csv
    # ------------------------------------------------------------------
    part_summary = stats_output['participant_summary']
    # Merge per-participant channel-level significant count.
    ch_results_list = stats_output['participant_ch_results']
    for i, ps in enumerate(part_summary):
        pid = ps['participant']
        if ps.get('skipped'):
            ps['n_sig_channels_fdr'] = 0
        else:
            crs = ch_results_list[i] if i < len(ch_results_list) else []
            ps['n_sig_channels_fdr'] = sum(
                1 for r in crs if isinstance(r, dict) and r.get('significant')
            )

    summary_df   = pd.DataFrame(part_summary)
    summary_path = os.path.join(RESULTS_DIR, 'participant_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log(f'Saved: {summary_path}')

    # ------------------------------------------------------------------
    # Save channel_stats.csv (group-level per channel).
    # ------------------------------------------------------------------
    # Augment with within-participant info: most common test_used per channel.
    ch_stats = stats_output['channel_stats']
    for ch_i, ch_row in enumerate(ch_stats):
        # Collect per-participant test_used for this channel.
        tests_used = [
            ch_results_list[p][ch_i].get('test_used')
            for p in range(len(ch_results_list))
            if (ch_i < len(ch_results_list[p]) and
                isinstance(ch_results_list[p][ch_i], dict))
        ]
        tests_used = [t for t in tests_used if t is not None]
        ch_row['within_participant_test'] = (
            max(set(tests_used), key=tests_used.count)
            if tests_used else 'n/a'
        )

    ch_stats_df   = pd.DataFrame(ch_stats)
    ch_stats_path = os.path.join(RESULTS_DIR, 'channel_stats.csv')
    ch_stats_df.to_csv(ch_stats_path, index=False)
    log(f'Saved: {ch_stats_path}')

    # ------------------------------------------------------------------
    # Save cluster_permutation_results.csv
    # ------------------------------------------------------------------
    clust_df   = pd.DataFrame(stats_output['cluster_results']) \
        if stats_output['cluster_results'] else pd.DataFrame()
    clust_path = os.path.join(RESULTS_DIR, 'cluster_permutation_results.csv')
    clust_df.to_csv(clust_path, index=False)
    log(f'Saved: {clust_path}  '
        f'({len(stats_output["cluster_results"])} significant clusters)')

    elapsed = time.time() - t0
    log(f'--- PHASE 4 complete: {elapsed:.1f}s ---')
    return stats_output


# ===========================================================================
# Phase 5 — Figures
# ===========================================================================

def phase5_figures(stats_output, trial_df, sw_df, centres_ms,
                    erp_accum, digit_pairs, liftoff_pairs,
                    spatial_results=None):
    """
    Generate QC Figure A and all publication figures.

    Parameters
    ----------
    erp_accum : dict
        Per-participant ERP accumulation: {pid: {'sum': (2000,32), 'count': int}}
    digit_pairs : list of (tFirstDig_s, onset_s) tuples
    """
    log('--- PHASE 5: figure generation ---')
    t0 = time.time()

    # ------------------------------------------------------------------
    # Build ERP data dict for QC Figure A.
    # Compute per-participant mean epoch, then grand mean and SEM.
    # ------------------------------------------------------------------
    part_means = []
    for pid in range(1, N_PARTICIPANTS + 1):
        cnt = erp_accum[pid]['count']
        if cnt == 0:
            continue
        part_means.append(erp_accum[pid]['sum'] / cnt)   # (2000, 32)

    if part_means:
        part_means_arr = np.array(part_means)   # (N_valid_parts, 2000, 32)
        grand_avg      = np.mean(part_means_arr, axis=0)   # (2000, 32)
        sem_arr        = (np.std(part_means_arr, axis=0, ddof=1) /
                          np.sqrt(part_means_arr.shape[0]))   # (2000, 32)
    else:
        # No valid participants — create empty arrays.
        grand_avg = np.zeros((2000, N_CHANNELS))
        sem_arr   = np.zeros((2000, N_CHANNELS))

    # Use combined_t_ms from first participant's first trial.
    # All trials produce identical time vectors (same epoch parameters).
    # Reconstruct directly from sample positions.
    ms_per_sample  = 1000.0 / EEG_SR   # = 2.0 ms
    pre_samples    = int(EPOCH_PRE_MS / 1000 * EEG_SR)   # = 1000
    post_samples   = int(EPOCH_POST_MS / 1000 * EEG_SR)  # = 1000
    combined_t_ms  = np.concatenate([
        np.arange(-pre_samples, 0)  * ms_per_sample,
        np.arange(0, post_samples)  * ms_per_sample,
    ])

    erp_data = {
        'grand_avg':      grand_avg,      # (2000, 32), raw 10-µV units
        'sem':            sem_arr,         # (2000, 32)
        'combined_t_ms':  combined_t_ms,  # (2000,) ms
    }

    # ------------------------------------------------------------------
    # Compute mean digit contact time relative to EMG onset.
    # ------------------------------------------------------------------
    if digit_pairs:
        diffs         = [(td - os_) * 1000.0 for td, os_ in digit_pairs]
        digit_contact_ms = float(np.mean(diffs))
        log(f'  Mean digit contact (re. EMG onset): {digit_contact_ms:.1f} ms')
    else:
        digit_contact_ms = np.nan
        log('  No digit contact data available for Figure 3 annotation.')

    # Grand mean tLiftOff relative to EMG onset (ms).
    # liftoff_pairs contains (participant_id, tLiftOff_s, emg_onset_s).
    if liftoff_pairs:
        liftoff_arr = np.array(
            [(lo - eo) * 1000 for _, lo, eo in liftoff_pairs],
            dtype=np.float64)
        liftoff_contact_ms = float(np.nanmean(liftoff_arr))
    else:
        liftoff_contact_ms = np.nan

    # Per-participant mean tLiftOff relative to EMG onset (ms).
    # Computed from participant_id stored in each liftoff_pairs tuple —
    # no ordering assumptions, no zip alignment with other lists.
    per_participant_liftoff_ms = {}
    if liftoff_pairs:
        from collections import defaultdict
        _pid_liftoff = defaultdict(list)
        for (pid_inner, lo, eo) in liftoff_pairs:
            _pid_liftoff[pid_inner].append((lo - eo) * 1000)
        for pid in range(1, N_PARTICIPANTS + 1):
            vals = _pid_liftoff.get(pid, [])
            per_participant_liftoff_ms[pid] = float(
                np.nanmean(vals)) if vals else np.nan
    else:
        for pid in range(1, N_PARTICIPANTS + 1):
            per_participant_liftoff_ms[pid] = np.nan

    # LED cue timing relative to EMG onset (ms).
    # LED cue is always at 2.000 s window-relative.
    # LED cue relative to EMG onset = -emg_onset_ms_post_led (negative).
    # Computed from onset_detection_results.csv already on disk.
    try:
        _onset_df = pd.read_csv(
            os.path.join(RESULTS_DIR, 'onset_detection_results.csv'))
        _valid = _onset_df[_onset_df['excluded'] == False]
        _led_rel = -_valid['emg_onset_ms_post_led'].values.astype(
            np.float64)
        led_mean_ms = float(np.nanmean(_led_rel))
        led_sd_ms   = float(np.nanstd(_led_rel, ddof=1))
        # Note: led_sd_ms pools within- and between-participant
        # variability. This is documented in the figure caption.

        # Per-participant mean LED cue time relative to EMG onset.
        per_participant_led_ms = {}
        for pid in range(1, N_PARTICIPANTS + 1):
            p_rows = _valid[_valid['participant'] == pid]
            if len(p_rows) > 0:
                per_participant_led_ms[pid] = float(
                    np.nanmean(
                        -p_rows['emg_onset_ms_post_led'].values))
            else:
                per_participant_led_ms[pid] = np.nan
    except Exception as e:
        log(f'[WARNING] Could not compute LED cue figure markers: {e}')
        led_mean_ms            = np.nan
        led_sd_ms              = np.nan
        per_participant_led_ms = {
            p: np.nan for p in range(1, N_PARTICIPANTS + 1)}

    # ------------------------------------------------------------------
    # Generate all figures.
    # ------------------------------------------------------------------
    generate_all_figures(
        stats_output=stats_output,
        trial_df=trial_df,
        sw_df=sw_df,
        centres_ms=centres_ms,
        erp_data=erp_data,
        digit_contact_ms=digit_contact_ms,
        led_mean_ms=led_mean_ms,
        led_sd_ms=led_sd_ms,
        per_participant_led_ms=per_participant_led_ms,
        liftoff_contact_ms=liftoff_contact_ms,
        per_participant_liftoff_ms=per_participant_liftoff_ms,
        spatial_results=spatial_results,
    )

    print('Inspect figures/qc_erp_sanity_check.png before interpreting results.')

    elapsed = time.time() - t0
    log(f'--- PHASE 5 complete: {elapsed:.1f}s ---')


# ===========================================================================
# Phase 6 — Console summary table
# ===========================================================================

def phase6_summary(stats_output, trial_df):
    """Print per-participant summary table to console and log."""
    log('--- PHASE 6: console summary ---')

    header = (
        f"{'P':>2} | {'Valid':>5} | {'Excl':>4} | "
        f"{'Onset(ms)':>9} | {'Pre SE':>6} | {'Post SE':>7} | "
        f"{'Dir1':>9} | {'Test':>9} | {'Stat':>6} | "
        f"{'p(FDR)':>7} | {'ES':>5} | "
        f"{'Slope':>8} | {'1stSig(ms)':>10} | {'Rec(ms)':>8} | {'NaN%':>5}"
    )
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    log(header)

    part_summaries  = stats_output['participant_summary']
    ch_results_list = stats_output['participant_ch_results']

    for ps in part_summaries:
        pid = ps['participant']
        if ps.get('skipped'):
            row = (f"{pid:>2} | {'0':>5} | {'all':>4} | "
                   f"{'—':>9} | {'—':>6} | {'—':>7} | "
                   f"{'—':>9} | {'—':>9} | {'—':>6} | "
                   f"{'—':>7} | {'—':>5} | "
                   f"{'—':>8} | {'—':>10} | {'—':>8} | {'—':>5}")
            print(row)
            log(row)
            continue

        p_df      = trial_df[trial_df['participant'] == pid]
        n_valid   = len(p_df)
        n_total   = n_valid   # excluded are not in trial_df
        mean_pre  = ps.get('mean_pre_sampen',  np.nan)
        mean_post = ps.get('mean_post_sampen', np.nan)
        dir1      = ps.get('direction_pre_vs_post', '—')[:9]
        slope_dir = ps.get('slope_direction', '—')[:8]
        first_sig = ps.get('first_sig_pre_window_ms', np.nan)
        rec_ms    = ps.get('recovery_ms', np.nan)

        # Use Cz (idx 13) as the representative channel for stat reporting.
        crs    = ch_results_list[ps['participant'] - 1] \
            if ps['participant'] - 1 < len(ch_results_list) else []
        cz_res = crs[13] if len(crs) > 13 and isinstance(crs[13], dict) else {}

        test_s  = cz_res.get('test_used', '—')[:9]
        stat_s  = f"{cz_res.get('statistic', np.nan):.2f}" \
            if not np.isnan(cz_res.get('statistic', np.nan)) else '—'
        p_fdr_s = f"{cz_res.get('p_fdr', np.nan):.3f}" \
            if not np.isnan(cz_res.get('p_fdr', np.nan)) else '—'
        es_s    = f"{cz_res.get('effect_size', np.nan):.2f}" \
            if not np.isnan(cz_res.get('effect_size', np.nan)) else '—'

        onset_ms = np.nanmean(p_df['emg_onset_ms_post_led'].values) \
            if 'emg_onset_ms_post_led' in p_df.columns else np.nan

        # Compute NaN rate from pre_sampen_ch* columns.
        pre_cols = [f'pre_sampen_ch{ch}' for ch in range(N_CHANNELS)]
        n_nan    = p_df[pre_cols].isna().sum().sum()
        n_total_cells = len(p_df) * N_CHANNELS
        nan_pct  = 100.0 * n_nan / max(n_total_cells, 1)

        first_sig_s = f'{first_sig:.0f}' if not np.isnan(first_sig) else '—'
        rec_ms_s    = f'{rec_ms:.0f}'    if not np.isnan(rec_ms)    else '—'
        onset_s_str = f'{onset_ms:.0f}'  if not np.isnan(onset_ms)  else '—'

        row = (
            f"{pid:>2} | {n_valid:>5} | {'—':>4} | "
            f"{onset_s_str:>9} | {mean_pre:>6.3f} | {mean_post:>7.3f} | "
            f"{dir1:>9} | {test_s:>9} | {stat_s:>6} | "
            f"{p_fdr_s:>7} | {es_s:>5} | "
            f"{slope_dir:>8} | {first_sig_s:>10} | {rec_ms_s:>8} | "
            f"{nan_pct:>4.1f}%"
        )
        print(row)
        log(row)

    print(sep + '\n')
    log(sep)


# ===========================================================================
# Phase 7 — Finalise log
# ===========================================================================

def phase7_finalise(t_start):
    """Write completion timestamp and full file list to pipeline_log.txt."""
    log('--- PHASE 7: finalising ---')

    # List all output files.
    for dirname in [RESULTS_DIR, FIGURES_DIR]:
        log(f'Files in {dirname}/:')
        try:
            for fname in sorted(os.listdir(dirname)):
                fpath = os.path.join(dirname, fname)
                size  = os.path.getsize(fpath)
                log(f'  {fname}  ({size/1024:.1f} kB)')
        except OSError as e:
            log(f'  Could not list {dirname}: {e}')

    total_elapsed = time.time() - t_start
    end_ts        = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log(f'Total pipeline runtime: {total_elapsed/60:.1f} min')
    log(f'Run ended: {end_ts}')
    log('=' * 70)

    if _log_fh is not None:
        _log_fh.close()


# ===========================================================================
# Phase 4b — Spatial analyses (ROI, hemisphere, lobe)
# ===========================================================================

def phase4b_spatial_analyses(stats_output, trial_df,
                              centres_ms, log_fn,
                              sw_df=None):
    """
    Runs all new spatial analyses in parallel to the existing
    32-channel whole-scalp analyses.

    Does NOT modify stats_output or participant_mean_norm.
    All results returned as a new separate dictionary.
    sw_df is intentionally NOT passed — the ROI condition
    generalizability test is skipped pending a future update
    (Stage 7) that will correctly compute per-trial ROI means
    from sw_df. This is documented in run_roi_analyses.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics. Contains
        participant_mean_norm which is read but never modified.
    trial_df : pd.DataFrame
        Trial-level entropy data. Passed to run_roi_analyses
        for weight condition label extraction only.
    centres_ms : list of float
        Sliding window centre times in milliseconds.
    log_fn : callable
        Logging function.

    Returns
    -------
    dict with keys:
        'roi'        : output of run_roi_analyses
        'hemisphere' : output of run_hemisphere_analyses
        'lobe'       : output of run_lobe_analyses
    """
    # Use 'from stats import' NOT 'from src.stats import'
    # src/ is already on sys.path — src.stats will fail.
    from stats import (run_roi_analyses,
                       run_hemisphere_analyses,
                       run_lobe_analyses)
    log_fn('=== Phase 4b: Spatial analyses ===')
    roi_results  = run_roi_analyses(
        stats_output, trial_df, centres_ms, log_fn,
        sw_df=sw_df)
    hemi_results = run_hemisphere_analyses(
        stats_output, centres_ms, log_fn)
    lobe_results = run_lobe_analyses(
        stats_output, centres_ms, log_fn)
    return {
        'roi':        roi_results,
        'hemisphere': hemi_results,
        'lobe':       lobe_results,
    }


# ===========================================================================
# Entry point
# ===========================================================================
# RUN_STATISTICS_AND_FIGURES, PARTICIPANT_START, and PARTICIPANT_END are
# configured in config.py — not here.  Edit config.py before each run.

if __name__ == '__main__':
    t_start = time.time()

    # Phase 0 — setup (always runs)
    phase0_setup()
    log(f'Batch range: P{PARTICIPANT_START}–P{PARTICIPANT_END} | '
        f'RUN_STATISTICS_AND_FIGURES={RUN_STATISTICS_AND_FIGURES}')

    # ------------------------------------------------------------------
    # Phases 1–3 always run.
    # In batch runs, only PARTICIPANT_START..PARTICIPANT_END are processed.
    # In the consolidation run (all 12), Phase 3 automatically skips every
    # participant whose per-participant entropy CSV already exists on disk
    # (restart-safe logic already present inside phase3_entropy).
    # ------------------------------------------------------------------

    # Phase 1 — EMG onset detection
    onset_records = phase1_onset_detection()

    # Manual checkpoint (blocks until user types 'continue')
    phase1_checkpoint()

    # Phase 2 — EEG preprocessing and epoch extraction
    trial_data_by_participant, centres_ms, erp_accum, digit_pairs, liftoff_pairs = \
        phase2_preprocess_epochs(onset_records)

    if centres_ms is None:
        log('ERROR: No valid trials produced sliding windows — '
            'cannot proceed to entropy.')
        raise RuntimeError('centres_ms is None — no valid trials found.')

    # Phase 3 — entropy computation (parallelized, per-participant caching)
    trial_df, sw_df = phase3_entropy(trial_data_by_participant, centres_ms)

    # ------------------------------------------------------------------
    # Phases 4 and 5 run only when RUN_STATISTICS_AND_FIGURES=True.
    # Set this flag in config.py once all 4 batch runs are complete and
    # all 12 per-participant entropy CSV files exist in results/.
    # ------------------------------------------------------------------
    if RUN_STATISTICS_AND_FIGURES:
        # Phase 4 — statistics
        stats_output = phase4_statistics(centres_ms,
                                         trial_df=trial_df,
                                         sw_df=sw_df)

        # ---------------------------------------------------------
        # Phase 4b: Spatial analyses (ROI, hemisphere, lobe).
        # Runs in parallel to existing 32-channel analyses.
        # stats_output is NOT modified by this call.
        # sw_df is intentionally not passed — see docstring.
        # ---------------------------------------------------------
        spatial_results = phase4b_spatial_analyses(
            stats_output, trial_df, centres_ms, log,
            sw_df=sw_df)

        # ---------------------------------------------------------
        # Save spatial analysis results to CSV files.
        # numpy arrays (roi_matrix, left_matrix, right_matrix,
        # lr_diff, lobe_matrices, lobe_term_means,
        # lobe_term_slopes) are NOT saved to CSV.
        # ---------------------------------------------------------

        # Block A — ROI cluster permutation results
        _roi_cl_cols = ['cluster_index', 'cluster_p_value',
                        'time_start_ms', 'time_end_ms',
                        'epoch', 'direction']
        _roi_clusters = spatial_results['roi']['cluster_results']
        if _roi_clusters:
            _roi_cl_df = pd.DataFrame(
                _roi_clusters)[_roi_cl_cols]
        else:
            _roi_cl_df = pd.DataFrame(columns=_roi_cl_cols)
        _roi_cl_path = os.path.join(
            RESULTS_DIR, 'roi_cluster_permutation_results.csv')
        _roi_cl_df.to_csv(_roi_cl_path, index=False)
        log(f'Saved: {_roi_cl_path}')

        # Block B — ROI participant summary
        _roi_r = spatial_results['roi']
        _roi_ps_df = pd.DataFrame([{
            'full_slope_mean':          _roi_r['full_slope_mean'],
            'full_slope_direction':     _roi_r['full_slope_direction'],
            'full_slope_p':             _roi_r['full_slope_p'],
            'terminal_slope_mean':      _roi_r['terminal_slope_mean'],
            'terminal_slope_direction': _roi_r['terminal_slope_direction'],
            'terminal_slope_p':         _roi_r['terminal_slope_p'],
            'prepost_direction':        _roi_r['prepost_direction'],
            'prepost_p':                _roi_r['prepost_p'],
            'condition_F':              _roi_r['condition_F'],
            'condition_p':              _roi_r['condition_p'],
            'condition_significant':    _roi_r['condition_significant'],
        }])
        _roi_ps_path = os.path.join(
            RESULTS_DIR, 'roi_participant_summary.csv')
        _roi_ps_df.to_csv(_roi_ps_path, index=False)
        log(f'Saved: {_roi_ps_path}')

        # Block C — ROI slope results
        _roi_sl_df = pd.DataFrame([{
            'full_slope_mean':          _roi_r['full_slope_mean'],
            'full_slope_direction':     _roi_r['full_slope_direction'],
            'full_slope_p':             _roi_r['full_slope_p'],
            'terminal_slope_mean':      _roi_r['terminal_slope_mean'],
            'terminal_slope_direction': _roi_r['terminal_slope_direction'],
            'terminal_slope_p':         _roi_r['terminal_slope_p'],
        }])
        _roi_sl_path = os.path.join(
            RESULTS_DIR, 'roi_slope_results.csv')
        _roi_sl_df.to_csv(_roi_sl_path, index=False)
        log(f'Saved: {_roi_sl_path}')

        # Block D — ROI condition generalizability
        _roi_cg_df = pd.DataFrame([{
            'condition_F':           _roi_r['condition_F'],
            'condition_p':           _roi_r['condition_p'],
            'condition_significant': _roi_r['condition_significant'],
        }])
        _roi_cg_path = os.path.join(
            RESULTS_DIR, 'roi_condition_generalizability.csv')
        _roi_cg_df.to_csv(_roi_cg_path, index=False)
        log(f'Saved: {_roi_cg_path}')

        # Block E — Hemisphere cluster permutation results
        _hemi_cl_cols = ['cluster_index', 'cluster_p_value',
                         'time_start_ms', 'time_end_ms',
                         'epoch', 'direction']
        _hemi_clusters = spatial_results['hemisphere'][
            'cluster_results']
        if _hemi_clusters:
            _hemi_cl_df = pd.DataFrame(
                _hemi_clusters)[_hemi_cl_cols]
        else:
            _hemi_cl_df = pd.DataFrame(columns=_hemi_cl_cols)
        _hemi_cl_path = os.path.join(
            RESULTS_DIR,
            'hemisphere_cluster_permutation_results.csv')
        _hemi_cl_df.to_csv(_hemi_cl_path, index=False)
        log(f'Saved: {_hemi_cl_path}')

        # Block F — Hemisphere terminal results
        _hemi_r = spatial_results['hemisphere']
        _hemi_tr_df = pd.DataFrame([{
            'terminal_mean_lr':         _hemi_r['terminal_mean_lr'],
            'terminal_mean_direction':  _hemi_r['terminal_mean_direction'],
            'terminal_mean_p':          _hemi_r['terminal_mean_p'],
            'terminal_slope_mean':      _hemi_r['terminal_slope_mean'],
            'terminal_slope_direction': _hemi_r['terminal_slope_direction'],
            'terminal_slope_p':         _hemi_r['terminal_slope_p'],
        }])
        _hemi_tr_path = os.path.join(
            RESULTS_DIR, 'hemisphere_terminal_results.csv')
        _hemi_tr_df.to_csv(_hemi_tr_path, index=False)
        log(f'Saved: {_hemi_tr_path}')

        # Block G — Lobe terminal means ANOVA
        # Unified column structure across three sections:
        #   Section 1 — per-lobe means (result_type='lobe_mean')
        #   Section 2 — omnibus ANOVA result (result_type='omnibus')
        #   Section 3 — post hoc rows (result_type='posthoc'),
        #               only if optionA_posthoc is not empty.
        # Fields not applicable to a row are left as NaN.
        _lobe_r = spatial_results['lobe']
        _lta_cols = [
            'result_type', 'lobe',
            'mean_terminal_sampen', 'sem_terminal_sampen',
            'optionA_F', 'optionA_p', 'optionA_significant',
            'contrast', 'stat', 'p_uncorrected',
            'p_bonferroni', 'posthoc_significant',
            'posthoc_direction',
        ]
        _lta_rows = []
        for _lobe_name in ['frontal', 'central',
                            'parietal', 'occipital']:
            _lv = _lobe_r['lobe_term_means'][_lobe_name]
            _lta_rows.append({
                'result_type':          'lobe_mean',
                'lobe':                 _lobe_name,
                'mean_terminal_sampen': float(np.nanmean(_lv)),
                'sem_terminal_sampen':  float(
                    np.nanstd(_lv, ddof=1) / np.sqrt(len(_lv))),
                'optionA_F':            np.nan,
                'optionA_p':            np.nan,
                'optionA_significant':  np.nan,
                'contrast':             np.nan,
                'stat':                 np.nan,
                'p_uncorrected':        np.nan,
                'p_bonferroni':         np.nan,
                'posthoc_significant':  np.nan,
                'posthoc_direction':    np.nan,
            })
        _lta_rows.append({
            'result_type':          'omnibus',
            'lobe':                 np.nan,
            'mean_terminal_sampen': np.nan,
            'sem_terminal_sampen':  np.nan,
            'optionA_F':            _lobe_r['optionA_F'],
            'optionA_p':            _lobe_r['optionA_p'],
            'optionA_significant':  _lobe_r['optionA_significant'],
            'contrast':             np.nan,
            'stat':                 np.nan,
            'p_uncorrected':        np.nan,
            'p_bonferroni':         np.nan,
            'posthoc_significant':  np.nan,
            'posthoc_direction':    np.nan,
        })
        for _ph_key, _ph_val in _lobe_r['optionA_posthoc'].items():
            _lta_rows.append({
                'result_type':          'posthoc',
                'lobe':                 np.nan,
                'mean_terminal_sampen': np.nan,
                'sem_terminal_sampen':  np.nan,
                'optionA_F':            np.nan,
                'optionA_p':            np.nan,
                'optionA_significant':  np.nan,
                'contrast':             _ph_key,
                'stat':                 _ph_val['stat'],
                'p_uncorrected':        _ph_val['p_uncorrected'],
                'p_bonferroni':         _ph_val['p_bonferroni'],
                'posthoc_significant':  _ph_val['significant'],
                'posthoc_direction':    _ph_val['direction'],
            })
        _lta_df = pd.DataFrame(_lta_rows, columns=_lta_cols)
        _lta_path = os.path.join(
            RESULTS_DIR, 'lobe_terminal_means_anova.csv')
        _lta_df.to_csv(_lta_path, index=False)
        log(f'Saved: {_lta_path}')

        # Block H — Lobe terminal slopes ANOVA
        _ltsa_df = pd.DataFrame([{
            'optionB_F':           _lobe_r['optionB_F'],
            'optionB_p':           _lobe_r['optionB_p'],
            'optionB_significant': _lobe_r['optionB_significant'],
        }])
        _ltsa_path = os.path.join(
            RESULTS_DIR, 'lobe_terminal_slopes_anova.csv')
        _ltsa_df.to_csv(_ltsa_path, index=False)
        log(f'Saved: {_ltsa_path}')

        # Block I — Lobe cluster permutation results
        # If a contrast has no significant clusters, one row is
        # written for that contrast with all cluster fields as NaN.
        _lobe_cl_cols = [
            'contrast', 'cluster_index', 'cluster_p_value',
            'time_start_ms', 'time_end_ms', 'epoch', 'direction',
        ]
        _lobe_cl_rows = []
        for _contrast_key, _cl_list in \
                _lobe_r['optionC_clusters'].items():
            if _cl_list:
                for _cl in _cl_list:
                    _lobe_cl_rows.append({
                        'contrast':        _contrast_key,
                        'cluster_index':   _cl['cluster_index'],
                        'cluster_p_value': _cl['cluster_p_value'],
                        'time_start_ms':   _cl['time_start_ms'],
                        'time_end_ms':     _cl['time_end_ms'],
                        'epoch':           _cl['epoch'],
                        'direction':       _cl['direction'],
                    })
            else:
                _lobe_cl_rows.append({
                    'contrast':        _contrast_key,
                    'cluster_index':   np.nan,
                    'cluster_p_value': np.nan,
                    'time_start_ms':   np.nan,
                    'time_end_ms':     np.nan,
                    'epoch':           np.nan,
                    'direction':       np.nan,
                })
        _lobe_cl_df = pd.DataFrame(
            _lobe_cl_rows, columns=_lobe_cl_cols)
        _lobe_cl_path = os.path.join(
            RESULTS_DIR,
            'lobe_cluster_permutation_results.csv')
        _lobe_cl_df.to_csv(_lobe_cl_path, index=False)
        log(f'Saved: {_lobe_cl_path}')

        # Phase 5 — figures
        phase5_figures(stats_output, trial_df, sw_df, centres_ms,
                       erp_accum, digit_pairs, liftoff_pairs,
                       spatial_results=spatial_results)

        # Phase 6 — console summary
        phase6_summary(stats_output, trial_df)

    else:
        log('RUN_STATISTICS_AND_FIGURES=False — '
            'skipping statistics and figures. '
            'Set RUN_STATISTICS_AND_FIGURES=True and re-run after all '
            '4 batch runs are complete (all 12 per-participant entropy '
            'CSV files must exist in results/ before running statistics).')

    # Phase 7 — finalise log (always runs)
    phase7_finalise(t_start)

