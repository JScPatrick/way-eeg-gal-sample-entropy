"""
entropy.py — Sample entropy computation for the WAY-EEG-GAL pipeline.

Two public symbols:
    compute_trial_entropy   Compute SampEn for one trial's pre/post epochs
                            and all sliding windows, with baseline normalisation
    _compute_one_trial      Thin wrapper accepted by joblib.Parallel

SampEn is computed via src/sampen.py. antropy is never imported.

SampEn parameters (from config)
--------------------------------
    m = SAMPEN_ORDER = 2        embedding dimension
    r = 0.2 × std(signal)       tolerance, computed per epoch per channel
                                 (r=None in sampen.py triggers this behaviour)

The per-epoch, per-channel r is mandatory. A fixed r across epochs would
conflate genuine complexity changes with amplitude changes, because SampEn
is sensitive to the ratio of template matches relative to the tolerance
width. When amplitude changes (as it does around movement), fixing r
produces spurious entropy changes that are really amplitude effects.

All parameters imported from config. antropy is never imported.

References
----------
Richman & Moorman (2000), Am J Physiol Heart Circ Physiol — SampEn definition.
Lake et al. (2002), Am J Physiol — per-epoch r scaling.
"""

import numpy as np
from joblib import Parallel, delayed

# sampen.py lives in the same src/ directory.
# It accepts (signal, order, metric='chebyshev') and computes
# r = 0.2 * std(signal) internally when r is not passed.
from sampen import sample_entropy

from config import (
    SAMPEN_ORDER,
    SAMPEN_TOLERANCE_MODE,
    N_CHANNELS,
    N_JOBS,
)


# ---------------------------------------------------------------------------
# Per-trial entropy computation
# ---------------------------------------------------------------------------

def compute_trial_entropy(pre_epoch, post_epoch, windows):
    """
    Compute SampEn for the pre-epoch, post-epoch, and all sliding windows
    for one trial, then baseline-normalise the sliding-window series.

    Parameters
    ----------
    pre_epoch : numpy.ndarray, shape (1000, 32)
        Pre-onset EEG epoch (2000 ms before EMG onset).
    post_epoch : numpy.ndarray, shape (1000, 32)
        Post-onset EEG epoch (2000 ms after EMG onset, onset sample included).
    windows : list of numpy.ndarray, each shape (250, 32)
        Sliding-window segments from build_sliding_windows (epoching.py).

    Returns
    -------
    pre_sampen : numpy.ndarray, shape (32,)
        SampEn for the full 2000 ms pre-onset epoch, one value per channel.
        NaN where sample_entropy returns NaN (e.g. all templates matched or
        no templates matched — both indicate the epoch is unsuitable for SampEn).
    post_sampen : numpy.ndarray, shape (32,)
        SampEn for the full 2000 ms post-onset epoch, one value per channel.
        NaN where undefined.
    sw_sampen : numpy.ndarray, shape (N_windows, 32)
        SampEn for each sliding window × channel. NaN where undefined.
    sw_sampen_norm : numpy.ndarray, shape (N_windows, 32)
        Baseline-normalised SampEn. Baseline = mean of first two windows
        (indices 0 and 1) per channel. sw_sampen_norm[w, ch] =
        sw_sampen[w, ch] - baseline_mean[ch].
        NaN propagates: if baseline is NaN for a channel, all windows for
        that channel are NaN in sw_sampen_norm.

    Notes
    -----
    SampEn tolerance r:
        sample_entropy(signal, order=SAMPEN_ORDER) computes r = 0.2 × std(signal)
        internally. This per-call r scaling ensures that the tolerance adapts
        to each epoch's amplitude, preventing amplitude changes around movement
        onset from being misidentified as complexity changes.

    Baseline windows:
        Windows 0 and 1 are guaranteed to fall before LEDon by the
        'baseline_window_post_led' exclusion criterion in emg_onset.py.
        Their mean therefore reflects pre-cue resting complexity, providing
        a meaningful zero reference for the normalised series.

    NaN policy:
        NaN values are preserved and propagated deliberately. Downstream
        statistics use np.nanmean / np.nanstd to exclude NaN cells.
        Total NaN counts per participant are logged in pipeline_log.txt.
        Do NOT replace NaN with zero — zero is a valid SampEn value.

    No random state:
        SampEn is deterministic given the same input. No random seed is needed.
    """

    N_windows = len(windows)

    # ------------------------------------------------------------------
    # (a) Scalar pre / post SampEn (full 2000 ms epochs)
    # ------------------------------------------------------------------
    # Initialise with NaN so that channels where sample_entropy raises or
    # returns NaN are clearly marked rather than silently zero.
    pre_sampen  = np.full(N_CHANNELS, np.nan, dtype=np.float64)
    post_sampen = np.full(N_CHANNELS, np.nan, dtype=np.float64)

    for ch in range(N_CHANNELS):
        # r is computed inside sample_entropy as 0.2 * std(signal).
        # Passing order=SAMPEN_ORDER (=2) selects embedding dimension m=2.
        try:
            pre_sampen[ch]  = sample_entropy(pre_epoch[:, ch],
                                             order=SAMPEN_ORDER)
        except Exception:
            # Any failure (e.g. constant signal with std=0) leaves NaN.
            pass

        try:
            post_sampen[ch] = sample_entropy(post_epoch[:, ch],
                                             order=SAMPEN_ORDER)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # (b) Sliding-window SampEn
    # ------------------------------------------------------------------
    sw_sampen = np.full((N_windows, N_CHANNELS), np.nan, dtype=np.float64)

    # ---- CF1 CORRECTION: tolerance r held fixed across the window series ----
    # SampEn is comparable across segments only when r is constant across
    # them. These 29 windows are compared to one another, so r must not be
    # recomputed per window. With SAMPEN_TOLERANCE_MODE == 'per_epoch', r is
    # derived once per channel from the full combined epoch (the same
    # pre+post concatenation epoching.py builds) and passed to every window.
    #
    # r_per_ch is None in 'per_window' (legacy) mode, which restores the
    # original behaviour of letting sample_entropy derive r from each window.
    if SAMPEN_TOLERANCE_MODE == 'per_epoch':
        combined_epoch = np.concatenate([pre_epoch, post_epoch], axis=0)
        r_per_ch = 0.2 * np.std(combined_epoch, axis=0, ddof=1)   # shape (32,)
    elif SAMPEN_TOLERANCE_MODE == 'per_window':
        r_per_ch = None
    else:
        raise ValueError(
            f"Unknown SAMPEN_TOLERANCE_MODE {SAMPEN_TOLERANCE_MODE!r}; "
            f"expected 'per_epoch' or 'per_window'.")

    for w, window in enumerate(windows):
        for ch in range(N_CHANNELS):
            # r=None reproduces the legacy per-window tolerance.
            r_ch = None if r_per_ch is None else float(r_per_ch[ch])
            try:
                sw_sampen[w, ch] = sample_entropy(window[:, ch],
                                                  order=SAMPEN_ORDER,
                                                  r=r_ch)
            except Exception:
                # Leave NaN for this window × channel combination.
                pass

    # ------------------------------------------------------------------
    # (c) Baseline normalisation
    # ------------------------------------------------------------------
    # Baseline = mean of first two sliding windows (indices 0 and 1).
    # These windows are guaranteed to fall before LEDon by the exclusion
    # check in emg_onset.py, so they reflect pre-cue resting complexity.
    #
    # np.nanmean across axis=0 averages windows 0 and 1 per channel.
    # If both window 0 and window 1 are NaN for a channel, nanmean returns
    # NaN (with a RuntimeWarning) — this is correct: the baseline is
    # undefined for that channel and NaN must propagate to all windows.
    #
    # Shape note: sw_sampen[:2, :] has shape (2, 32).
    # np.nanmean(..., axis=0) produces shape (32,) — one baseline per channel.
    baseline_mean = np.nanmean(sw_sampen[:2, :], axis=0)   # shape (32,)

    # Subtract per-channel baseline from every window.
    # np.newaxis makes baseline_mean shape (1, 32) so it broadcasts against
    # sw_sampen shape (N_windows, 32) along the window axis.
    sw_sampen_norm = sw_sampen - baseline_mean[np.newaxis, :]

    return pre_sampen, post_sampen, sw_sampen, sw_sampen_norm


# ---------------------------------------------------------------------------
# joblib-compatible single-trial wrapper
# ---------------------------------------------------------------------------

def _compute_one_trial(trial_data):
    """
    Unpack a trial data tuple and call compute_trial_entropy.

    This thin wrapper exists so that joblib.Parallel can call a single
    callable rather than needing to unpack arguments itself, which would
    require lambda functions (not picklable on all platforms).

    Parameters
    ----------
    trial_data : tuple of (pre_epoch, post_epoch, windows)
        pre_epoch  : numpy.ndarray, shape (1000, 32)
        post_epoch : numpy.ndarray, shape (1000, 32)
        windows    : list of numpy.ndarray, each shape (250, 32)

    Returns
    -------
    Tuple of (pre_sampen, post_sampen, sw_sampen, sw_sampen_norm) as returned
    by compute_trial_entropy.
    """
    pre_epoch, post_epoch, windows = trial_data
    return compute_trial_entropy(pre_epoch, post_epoch, windows)


# ---------------------------------------------------------------------------
# Parallel entropy computation across all trials for one participant
# ---------------------------------------------------------------------------

def compute_participant_entropy(trial_data_list, participant_id):
    """
    Compute SampEn in parallel across all valid trials for one participant.

    Each trial is processed independently — there is no shared state between
    trials. joblib.Parallel distributes trials across available CPU cores.

    Parameters
    ----------
    trial_data_list : list of tuples
        Each element is (pre_epoch, post_epoch, windows) for one valid trial.
        Order must match the order in which trials will be saved to CSV.
    participant_id : int
        Participant number (1–12), used only for progress printing.

    Returns
    -------
    results : list of tuples
        Each element is (pre_sampen, post_sampen, sw_sampen, sw_sampen_norm)
        as returned by compute_trial_entropy, in the same order as input.

    Notes
    -----
    N_JOBS = -1 uses all available CPU cores. Set N_JOBS = 1 in config.py
    to disable parallelisation for debugging (serial execution, full tracebacks).

    joblib uses the 'loky' backend by default, which spawns independent worker
    processes. numpy arrays passed in trial_data_list are serialised via pickle
    between processes — ensure no non-picklable objects (e.g. file handles,
    MATLAB struct objects) are included in the tuples.

    Progress is printed every 10 trials after results are collected, since
    joblib does not natively support per-item callbacks with n_jobs != 1.
    """
    n_trials = len(trial_data_list)
    print(f'  P{participant_id}: launching parallel entropy computation '
          f'for {n_trials} trials (N_JOBS={N_JOBS}) ...')

    results = Parallel(n_jobs=N_JOBS)(
        delayed(_compute_one_trial)(td)
        for td in trial_data_list
    )

    # Print progress summary every 10 trials (post-hoc — parallel jobs
    # complete in arbitrary order; this gives a completion count, not live).
    for i in range(0, n_trials, 10):
        upper = min(i + 10, n_trials)
        print(f'  P{participant_id}: trials {i + 1}–{upper} of {n_trials} complete.')

    print(f'  P{participant_id}: entropy computation done.')
    return results
