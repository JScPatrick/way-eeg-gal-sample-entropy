"""
epoching.py — Epoch extraction and sliding-window construction.

Two public functions:
    extract_epochs         Cut pre- and post-onset epochs from filtered EEG
    build_sliding_windows  Slide a 500 ms window in 125 ms steps across epoch

Boundary convention
-------------------
pre_epoch  contains samples [onset_idx - pre_samples, onset_idx - 1]  inclusive.
post_epoch contains samples [onset_idx, onset_idx + post_samples - 1] inclusive.
The onset sample belongs to post_epoch ONLY. Including it in pre_epoch would
produce a combined epoch of 2001 samples, shifting every window index by one
and corrupting the time alignment of every downstream SampEn value.

All parameters imported from config. No magic numbers in this module.
"""

import numpy as np

from config import (
    EEG_SR,
    EPOCH_PRE_MS, EPOCH_POST_MS,
    SLIDE_WIN_MS, SLIDE_STEP_MS,
    N_CHANNELS,
)


# ---------------------------------------------------------------------------
# Epoch extraction
# ---------------------------------------------------------------------------

def extract_epochs(eeg_filtered, eeg_onset_idx):
    """
    Extract pre-onset, post-onset, and combined EEG epochs centred on EMG onset.

    Parameters
    ----------
    eeg_filtered : numpy.ndarray, shape (4714, 32)
        Preprocessed EEG in 10-µV units (output of preprocess_eeg).
    eeg_onset_idx : int
        EEG sample index of the detected EMG onset (0-indexed into eeg_filtered).
        Guaranteed >= pre_samples by check_exclusion criterion 5.

    Returns
    -------
    pre_epoch : numpy.ndarray, shape (1000, 32)
        EEG samples in the 2000 ms window ending one sample before onset.
        Time range: [-2000 ms, -2 ms] relative to onset (does NOT include onset).
    post_epoch : numpy.ndarray, shape (1000, 32)
        EEG samples in the 2000 ms window starting at the onset sample.
        Time range: [0 ms, 1998 ms] relative to onset (DOES include onset).
    combined_epoch : numpy.ndarray, shape (2000, 32)
        Concatenation of pre_epoch followed by post_epoch.
        Combined time range: [-2000 ms, 1998 ms] relative to onset.
    pre_t_ms : numpy.ndarray, shape (1000,)
        Time in milliseconds for each sample of pre_epoch, relative to onset.
        Values: [-2000, -1998, ..., -2] ms.
    post_t_ms : numpy.ndarray, shape (1000,)
        Time in milliseconds for each sample of post_epoch, relative to onset.
        Values: [0, 2, ..., 1998] ms.
    combined_t_ms : numpy.ndarray, shape (2000,)
        Time in milliseconds for each sample of combined_epoch.
        Values: [-2000, -1998, ..., 0, 2, ..., 1998] ms.
        Note: the last value is 1998 ms, NOT 2000 ms — there are 1000 post-onset
        samples indexed 0..999, and sample 999 is at 999 × (1000/500) = 1998 ms.

    Notes
    -----
    Sample counts are derived from integer arithmetic to guarantee exact epoch
    sizes regardless of floating-point rounding:
        pre_samples  = int(2000/1000 × 500) = 1000 exactly
        post_samples = int(2000/1000 × 500) = 1000 exactly

    Time vectors are computed from sample positions rather than assumed from
    the nominal sampling rate. This avoids silent misalignment if, for any
    reason, the time axis has a non-standard start.

    An assertion guards the combined epoch shape. A violation here would
    corrupt all downstream window indices and SampEn values and must be
    caught immediately rather than propagating silently.
    """

    pre_samples  = int(EPOCH_PRE_MS  / 1000 * EEG_SR)   # = 1000 exactly
    post_samples = int(EPOCH_POST_MS / 1000 * EEG_SR)   # = 1000 exactly

    # ------------------------------------------------------------------
    # Slice pre- and post-onset epochs
    # ------------------------------------------------------------------
    # pre_epoch: the 1000 samples immediately before onset.
    # Python slice [a:b] gives indices a, a+1, ..., b-1.
    # [onset_idx - pre_samples : onset_idx] → indices onset_idx-1000 .. onset_idx-1
    # This does NOT include the onset sample at onset_idx.
    pre_epoch  = eeg_filtered[eeg_onset_idx - pre_samples : eeg_onset_idx, :]

    # post_epoch: onset sample through the next 999 samples.
    # [onset_idx : onset_idx + post_samples] → indices onset_idx .. onset_idx+999
    # The onset sample (index onset_idx) IS the first sample of post_epoch.
    post_epoch = eeg_filtered[eeg_onset_idx : eeg_onset_idx + post_samples, :]

    # ------------------------------------------------------------------
    # Combine and validate
    # ------------------------------------------------------------------
    combined_epoch = np.concatenate([pre_epoch, post_epoch], axis=0)

    # Hard assertion: any shape other than (2000, 32) indicates a boundary
    # definition error and must stop execution immediately. Silently proceeding
    # with a wrong shape would corrupt every window index and SampEn value.
    assert combined_epoch.shape == (pre_samples + post_samples, N_CHANNELS), (
        f'Combined epoch shape error: got {combined_epoch.shape}, '
        f'expected ({pre_samples + post_samples}, {N_CHANNELS}). '
        f'eeg_onset_idx={eeg_onset_idx}, eeg_filtered.shape={eeg_filtered.shape}. '
        f'Check epoch boundary definitions in epoching.py.'
    )

    # ------------------------------------------------------------------
    # Build time vectors from sample positions
    # ------------------------------------------------------------------
    # ms_per_sample = 1000 / EEG_SR = 2.0 ms at 500 Hz.
    # Pre-onset samples run from index -pre_samples to -1 (relative to onset).
    # np.arange(-pre_samples, 0) → [-1000, -999, ..., -1]
    # × (1000/EEG_SR) → [-2000, -1998, ..., -2] ms
    ms_per_sample = 1000.0 / EEG_SR   # = 2.0 ms

    pre_t_ms  = np.arange(-pre_samples, 0)   * ms_per_sample
    post_t_ms = np.arange(0, post_samples)   * ms_per_sample

    combined_t_ms = np.concatenate([pre_t_ms, post_t_ms])

    # Sanity: verify time vector length matches epoch length.
    assert len(combined_t_ms) == combined_epoch.shape[0], (
        f'Time vector length {len(combined_t_ms)} does not match '
        f'epoch length {combined_epoch.shape[0]}.'
    )

    return (
        pre_epoch, post_epoch, combined_epoch,
        pre_t_ms, post_t_ms, combined_t_ms
    )


# ---------------------------------------------------------------------------
# Sliding window construction
# ---------------------------------------------------------------------------

def build_sliding_windows(combined_epoch, combined_t_ms):
    """
    Build overlapping sliding-window segments across the full ±2000 ms epoch.

    A 500 ms window (250 samples at 500 Hz) is advanced in 125 ms nominal
    steps (62 samples) across the combined epoch, producing windows with
    75.2 % overlap.

    Parameters
    ----------
    combined_epoch : numpy.ndarray, shape (2000, 32)
        Combined pre+post EEG epoch (output of extract_epochs).
    combined_t_ms : numpy.ndarray, shape (2000,)
        Time in milliseconds for each sample (from extract_epochs).
        Used to read actual window centre times rather than computing them
        from nominal step size (which is 124 ms, not 125 ms due to integer
        truncation of 62 samples × 2 ms/sample).

    Returns
    -------
    windows : list of numpy.ndarray, each shape (250, 32)
        Sliding window data segments. Each window is an independent view into
        combined_epoch (not a copy) — do not modify in-place downstream.
    centres_ms : list of float
        Actual centre time in milliseconds for each window, read from
        combined_t_ms at the centre sample index. These are the x-axis
        values used for all time-series plots and cluster permutation tests.
    epoch_labels : list of str
        'pre'  if the window centre is before onset (centre_ms < 0.0)
        'post' if the window centre is at or after onset (centre_ms >= 0.0)

    Notes
    -----
    Integer truncation of step size:
        SLIDE_STEP_MS = 125 ms → step_samples = int(125/1000 × 500) = 62 samples
        True step in ms = 62 × (1000/500) = 124 ms (not 125 ms).
        Centre times are therefore read from combined_t_ms rather than
        computed as start × ms_per_sample + SLIDE_WIN_MS/2, which would give
        the wrong value if the nominal and actual step sizes differ.

    Overlap and statistical consequences:
        Overlap fraction = (win_samples - step_samples) / win_samples
                        = (250 - 62) / 250 = 75.2 %
        Adjacent windows share 188 of their 250 samples. This makes adjacent
        SampEn values strongly correlated. Standard FDR correction (which
        assumes independence or weak positive dependence) is therefore invalid
        for the sliding-window time series — it would produce anti-conservative
        results (inflated false positives), directly undermining the temporal
        precedence claim. Cluster-based permutation testing is used instead
        in stats.py (Maris & Oostenveld, 2007, J Neurosci Methods).

    Window count:
        With 2000 samples, win=250, step=62:
        Number of complete windows = floor((2000 - 250) / 62) + 1
        The while loop enforces 'start + win_samples <= 2000' to avoid
        partial windows at the trailing edge.
    """

    win_samples  = int(SLIDE_WIN_MS  / 1000 * EEG_SR)   # = 250 samples
    step_samples = int(SLIDE_STEP_MS / 1000 * EEG_SR)   # = 62 samples

    windows      = []
    centres_ms   = []
    start        = 0

    while start + win_samples <= len(combined_epoch):
        # Slice window data — this is a view, not a copy.
        window_data = combined_epoch[start : start + win_samples, :]

        # Centre index of this window within combined_epoch.
        # Integer division gives the index of the middle sample.
        centre_idx = start + win_samples // 2   # = start + 125

        # Read the actual centre time from the time vector rather than
        # computing it from nominal step arithmetic, avoiding the 1 ms
        # rounding error introduced by step_samples=62 vs. nominal 62.5.
        centre_ms = float(combined_t_ms[centre_idx])

        windows.append(window_data)
        centres_ms.append(centre_ms)

        start += step_samples   # advance by 62 samples = 124 ms

    # Label each window by whether its centre falls before or after onset.
    # onset is at t=0 by convention (combined_t_ms crosses 0 at the onset sample).
    epoch_labels = ['pre' if c < 0.0 else 'post' for c in centres_ms]

    # Sanity: all windows must have the exact expected shape.
    for i, w in enumerate(windows):
        assert w.shape == (win_samples, N_CHANNELS), (
            f'Window {i} has unexpected shape {w.shape}; '
            f'expected ({win_samples}, {N_CHANNELS}).'
        )

    assert len(windows) == len(centres_ms) == len(epoch_labels), (
        'Mismatch between windows, centres_ms, and epoch_labels list lengths.'
    )

    return windows, centres_ms, epoch_labels
