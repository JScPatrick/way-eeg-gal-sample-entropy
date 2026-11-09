"""
emg_onset.py — EMG preprocessing and onset detection for the WAY-EEG-GAL pipeline.

Three public functions:
    preprocess_emg   Bandpass filter → full-wave rectify → RMS smooth
    detect_onset     Threshold + sustain algorithm → earliest onset across channels
    check_exclusion  Apply all trial exclusion criteria in priority order

All parameters imported from config. No magic numbers in this module.

Algorithm references
--------------------
Threshold + sustain: Hodges & Bui (1996), J Electromyogr Kinesiol.
RMS envelope:        Merletti & Parker (2004), Electromyography, IEEE Press.
Zero-phase filter:   Oppenheim & Schafer (2010), Discrete-Time Signal Processing.
"""

import numpy as np
import scipy.signal

from config import (
    EMG_SR, EEG_SR,
    EMG_BP_LOW, EMG_BP_HIGH,
    FILTER_ORDER, FILTFILT_PADLEN_EMG,
    BASELINE_START, BASELINE_END, LED_ON_TIME,
    EMG_THRESHOLD_SD, EMG_SUSTAIN_MS,
    MIN_REACTION_MS, MAX_ONSET_DELAY_MS,
    EPOCH_PRE_MS, BASELINE_CONTAMINATION_THRESHOLD_S,
    COL_T_HANDSTART,
)


# ---------------------------------------------------------------------------
# Step 1 — EMG preprocessing
# ---------------------------------------------------------------------------

def preprocess_emg(emg, emg_t):
    """
    Bandpass filter, rectify, and RMS-smooth the raw EMG signal.

    Processing chain:
        raw EMG → zero-phase bandpass → full-wave rectify → RMS envelope

    Parameters
    ----------
    emg : numpy.ndarray, shape (N_emg_samples, 5)
        Raw EMG in volts. Mean has already been removed by the original authors.
        Do not scale or convert units.
    emg_t : numpy.ndarray, shape (N_emg_samples,)
        EMG time vector in seconds, window-relative (starts at ~0.00025 s).

    Returns
    -------
    smoothed : numpy.ndarray, shape (N_emg_samples, 5)
        Smoothed RMS envelope in volts, same shape as input.

    Raises
    ------
    ValueError
        If emg is empty or contains NaN values. Both conditions make onset
        detection unreliable and the trial must be excluded by the caller.
    """

    # ------------------------------------------------------------------
    # 1a. Input validation
    # ------------------------------------------------------------------
    if emg.size == 0:
        raise ValueError('Empty EMG array — cannot preprocess.')

    if np.any(np.isnan(emg)):
        raise ValueError('NaN in raw EMG — trial must be excluded.')

    # ------------------------------------------------------------------
    # 1b. Zero-phase Butterworth bandpass filter (20–450 Hz)
    # ------------------------------------------------------------------
    # Zero-phase (filtfilt) is mandatory here. A causal filter introduces
    # group delay that shifts the onset estimate forward in time, biasing
    # the pre-onset epoch by underestimating onset latency.
    # padlen=FILTFILT_PADLEN_EMG (800 samples = 200 ms at 4000 Hz) ensures
    # the filter has settled before the pre-cue baseline window begins.
    # padtype='odd' reflects the signal at the edges, which is preferred over
    # zero-padding for continuous physiological recordings because it avoids
    # introducing a discontinuity at the boundary.
    b, a = scipy.signal.butter(
        FILTER_ORDER,
        [EMG_BP_LOW, EMG_BP_HIGH],
        btype='bandpass',
        fs=EMG_SR
    )

    n_samples, n_channels = emg.shape
    filtered = np.zeros_like(emg, dtype=np.float64)

    for ch in range(n_channels):
        filtered[:, ch] = scipy.signal.filtfilt(
            b, a,
            emg[:, ch].astype(np.float64),
            padtype='odd',
            padlen=FILTFILT_PADLEN_EMG   # = 800 samples
        )

    # ------------------------------------------------------------------
    # 1c. Full-wave rectification
    # ------------------------------------------------------------------
    # Rectification maps the bipolar EMG signal to non-negative amplitudes,
    # which is required before RMS smoothing. Full-wave (absolute value)
    # retains all energy and is standard for surface EMG amplitude estimation.
    envelope = np.abs(filtered)

    # ------------------------------------------------------------------
    # 1d. RMS smoothing
    # ------------------------------------------------------------------
    # RMS window: 50 ms = 200 samples at 4000 Hz.
    # RMS (root-mean-square) better represents signal amplitude than a
    # simple moving average because it weights larger deflections more
    # heavily, matching the energy content of the EMG burst.
    # convolution with a box kernel of length win implements a moving mean;
    # sqrt(moving_mean(x²)) gives the RMS.
    # mode='same' preserves the original signal length so the time axis
    # is unchanged.
    win    = int(0.050 * EMG_SR)          # = 200 samples = 50 ms window
    kernel = np.ones(win, dtype=np.float64) / win

    smoothed = np.zeros_like(envelope, dtype=np.float64)
    for ch in range(n_channels):
        sq       = envelope[:, ch] ** 2
        rms_conv = np.sqrt(np.convolve(sq, kernel, mode='same'))
        smoothed[:, ch] = rms_conv

    return smoothed


# ---------------------------------------------------------------------------
# Step 2 — Onset detection
# ---------------------------------------------------------------------------

def detect_onset(smoothed_envelope, emg_t, eeg_t, channel_names):
    """
    Detect EMG onset using a vectorized convolution-based threshold + sustain
    algorithm applied independently to each of the 5 EMG channels.

    The onset is defined as the earliest time point at which the smoothed RMS
    envelope exceeds a threshold (computed from the pre-cue baseline) and
    remains above that threshold continuously for at least EMG_SUSTAIN_MS ms.
    The earliest onset across all 5 channels is chosen as the trial onset.

    Parameters
    ----------
    smoothed_envelope : numpy.ndarray, shape (N_emg_samples, 5)
        Smoothed RMS EMG envelope in volts (output of preprocess_emg).
    emg_t : numpy.ndarray, shape (N_emg_samples,)
        EMG time vector in seconds (window-relative).
    eeg_t : numpy.ndarray, shape (N_eeg_samples,)
        EEG time vector in seconds (same reference frame as emg_t, pre-synchronised).
    channel_names : list of str, length 5
        EMG channel names from ws.names.emg (used to label onset_channel).

    Returns
    -------
    onset_time_s : float or None
        Time of the earliest EMG onset in seconds (window-relative).
        None if no onset detected on any channel.
    onset_channel : str or None
        Name of the channel on which the earliest onset was detected.
        None if no onset detected.
    eeg_onset_idx : int or None
        Index into eeg_t (and eeg array) corresponding to onset_time_s.
        Computed by nearest-neighbour matching against eeg_t to avoid
        floating-point accumulation errors from sample-rate arithmetic.
        None if no onset detected.

    Notes
    -----
    Threshold computation:
        All 5 channels are pooled for baseline statistics. Pooling gives a
        single robust threshold per trial. Per-channel thresholds risk
        inconsistent onset assignments when one channel has a noisier baseline.

    Sustain check via convolution:
        Summing a binary above-threshold mask over a sliding window of length
        sustain_samples using np.convolve with ones(sustain_samples) gives the
        number of consecutive above-threshold samples ending at each position.
        A value == sustain_samples at position k means all samples in
        [k - sustain_samples + 1, k] are above threshold.
        The onset index is therefore candidates[0] - sustain_samples + 1,
        i.e. the first sample of the first fully sustained run.
    """

    # ------------------------------------------------------------------
    # 2a. Compute threshold from pre-cue baseline (0.0–2.0 s, window-relative)
    # ------------------------------------------------------------------
    baseline_mask = (emg_t >= BASELINE_START) & (emg_t < BASELINE_END)

    # Guard against degenerate trials with almost no baseline data.
    # 100 samples at 4000 Hz = 25 ms — if the baseline window is this short
    # the threshold estimate is unreliable.
    if np.sum(baseline_mask) < 100:
        return None, None, None

    # Pool all 5 channels for a single threshold per trial.
    # This uses the full (N_baseline_samples, 5) block, flattened implicitly
    # by np.mean/np.std, giving N_baseline_samples × 5 data points.
    baseline_data = smoothed_envelope[baseline_mask, :]   # (N_bl, 5)
    mean_bl       = np.mean(baseline_data)                # scalar
    std_bl        = np.std(baseline_data)                 # scalar
    threshold     = mean_bl + EMG_THRESHOLD_SD * std_bl   # 3 SD above baseline

    # ------------------------------------------------------------------
    # 2b. Find per-channel onset using vectorized convolution sustain check
    # ------------------------------------------------------------------
    # sustain_samples: number of consecutive samples that must stay above
    # threshold for the crossing to count as a genuine onset.
    # EMG_SUSTAIN_MS = 25 ms → 25/1000 × 4000 = 100 samples at 4000 Hz.
    sustain_samples = int(EMG_SUSTAIN_MS / 1000 * EMG_SR)   # = 100 samples

    # Only search for onset after LEDon — anything before the cue is noise.
    post_led_mask = emg_t > LED_ON_TIME

    channel_onsets = {}   # channel_name → onset time in seconds (or np.inf)

    for ch_idx, ch_name in enumerate(channel_names):
        signal = smoothed_envelope[:, ch_idx]

        # Binary mask: 1 where signal exceeds threshold AND is post-LEDon.
        above = ((signal > threshold) & post_led_mask).astype(np.float32)

        # Sliding sum of length sustain_samples via convolution.
        # sum[k] = number of 1s in above[k-sustain_samples+1 : k+1].
        # mode='full' produces length len(above)+sustain_samples-1;
        # truncating to [:len(signal)] keeps only the causal part.
        sustained = np.convolve(
            above,
            np.ones(sustain_samples, dtype=np.float32),
            mode='full'
        )[:len(signal)]

        # A position where sustained == sustain_samples is the LAST sample
        # of the first fully sustained run above threshold.
        candidates = np.where(sustained >= sustain_samples)[0]

        if len(candidates) == 0:
            # No sustained crossing found on this channel.
            channel_onsets[ch_name] = np.inf
        else:
            # Walk back sustain_samples-1 positions to find the FIRST sample
            # of the sustained run — that is the actual onset moment.
            onset_idx_emg = int(candidates[0]) - sustain_samples + 1
            onset_idx_emg = max(0, onset_idx_emg)   # clamp to array bounds
            channel_onsets[ch_name] = float(emg_t[onset_idx_emg])

    # ------------------------------------------------------------------
    # 2c. Select the earliest onset across all channels
    # ------------------------------------------------------------------
    onset_time_s = min(channel_onsets.values())

    if onset_time_s == np.inf:
        # No channel produced a sustained threshold crossing — no onset found.
        return None, None, None

    # Identify which channel fired first (first match if tied).
    onset_channel = next(
        name for name, t in channel_onsets.items() if t == onset_time_s
    )

    # ------------------------------------------------------------------
    # 2d. Convert EMG onset time to EEG sample index via nearest-neighbour
    # ------------------------------------------------------------------
    # argmin(|eeg_t - onset_time_s|) avoids the floating-point accumulation
    # error that arises from computing round(onset_time_s * EEG_SR) directly.
    # EEG and EMG share the same time reference (pre-synchronised ≤2 ms).
    eeg_onset_idx = int(np.argmin(np.abs(eeg_t - onset_time_s)))

    return onset_time_s, onset_channel, eeg_onset_idx


# ---------------------------------------------------------------------------
# Step 3 — Trial exclusion criteria
# ---------------------------------------------------------------------------

def check_exclusion(onset_time_s, eeg_onset_idx, allifts_row, onset_found):
    """
    Apply all trial exclusion criteria in priority order.

    Criteria are evaluated sequentially; the first triggered criterion is
    returned and evaluation stops. This ensures each trial has exactly one
    exclusion reason logged, and the most fundamental reasons take precedence.

    Parameters
    ----------
    onset_time_s : float or None
        Detected EMG onset time in seconds (window-relative).
        None if detect_onset returned no onset.
    eeg_onset_idx : int or None
        EEG sample index of onset. None if no onset detected.
    allifts_row : numpy.ndarray, shape (43,)
        The AllLifts row for this trial (from get_trial_allifts).
        Used only for the tHandStart sanity check (criterion 6).
    onset_found : bool
        True if detect_onset returned a non-None onset_time_s.

    Returns
    -------
    exclusion_reason : str or None
        String label for the first exclusion criterion triggered.
        None means the trial passes all checks and should be included.

    Exclusion criteria (in evaluation order)
    -----------------------------------------
    1. no_onset_detected
       detect_onset found no sustained threshold crossing on any channel.

    2. onset_before_ledon
       Onset precedes LED cue — physiologically impossible, indicates a
       threshold crossing during the pre-cue baseline (false positive).

    3. onset_before_minimum_reaction_time
       Onset is within MIN_REACTION_MS (100 ms) of LEDon. The minimum
       voluntary motor reaction time is ~100 ms; earlier crossings are noise.

    4. no_onset_detected (re-used label for timeout)
       Onset occurred more than MAX_ONSET_DELAY_MS (3000 ms) after LEDon.
       Indicates a missed trial or EMG electrode failure.

    5. insufficient_pre_onset_eeg
       Fewer than EPOCH_PRE_MS/1000 × EEG_SR (1000) EEG samples precede
       the onset index — the full 2000 ms pre-onset epoch cannot be extracted.

    6. onset_after_kinematic_onset
       EMG onset is more than 100 ms later than tHandStart (kinematic onset).
       tHandStart is used ONLY as a sanity check — it is never the anchor.

    7. baseline_window_post_led
       The baseline normalization windows (first two sliding windows of the
       pre-onset epoch) would straddle or follow LEDon, contaminating the
       baseline with post-cue anticipatory activity.
       Triggered when onset_time_s >= BASELINE_CONTAMINATION_THRESHOLD_S
       (3.624 s from window start).
    """

    pre_samples = int(EPOCH_PRE_MS / 1000 * EEG_SR)   # = 1000 samples

    # ------------------------------------------------------------------
    # Criterion 1: No onset detected on any EMG channel
    # ------------------------------------------------------------------
    if not onset_found or onset_time_s is None:
        return 'no_onset_detected'

    # ------------------------------------------------------------------
    # Criterion 2: Onset falls before LED cue (physiologically impossible)
    # ------------------------------------------------------------------
    if onset_time_s < LED_ON_TIME:
        return 'onset_before_ledon'

    # ------------------------------------------------------------------
    # Criterion 3: Onset within 100 ms of LEDon (below minimum voluntary RT)
    # ------------------------------------------------------------------
    if onset_time_s < LED_ON_TIME + MIN_REACTION_MS / 1000:
        return 'onset_before_minimum_reaction_time'

    # ------------------------------------------------------------------
    # Criterion 4: Onset too late (>3 s post-LEDon → missed trial / failure)
    # ------------------------------------------------------------------
    if (onset_time_s - LED_ON_TIME) > MAX_ONSET_DELAY_MS / 1000:
        # Re-use 'no_onset_detected' label so downstream summaries group
        # timeout trials with true non-detections (both mean "no usable onset").
        return 'no_onset_detected'

    # ------------------------------------------------------------------
    # Criterion 5: Not enough EEG data before onset for full pre-epoch
    # ------------------------------------------------------------------
    if eeg_onset_idx is None or eeg_onset_idx < pre_samples:
        return 'insufficient_pre_onset_eeg'

    # ------------------------------------------------------------------
    # Criterion 6: tHandStart sanity check
    # EMG onset must not arrive more than 100 ms after kinematic hand start.
    # tHandStart is used ONLY here — it is never the epoch anchor.
    # ------------------------------------------------------------------
    tHandStart_s = allifts_row[COL_T_HANDSTART]
    if not np.isnan(tHandStart_s):
        if onset_time_s > tHandStart_s + 0.100:
            # EMG onset is more than 100 ms after kinematic onset — this
            # indicates onset detection failure (EMG not detected when hand
            # was already moving), not a valid late EMG response.
            return 'onset_after_kinematic_onset'

    # ------------------------------------------------------------------
    # Criterion 7: Baseline window contamination
    # The first two sliding windows of the pre-onset epoch serve as the
    # baseline for SampEn normalisation. Both must fall before LEDon.
    # If onset_time_s >= 3.624 s, the second baseline window centre
    # exceeds 2.000 s (LEDon), contaminating the baseline with post-cue
    # activity. See BASELINE_CONTAMINATION_THRESHOLD_S in config.py.
    # ------------------------------------------------------------------
    if onset_time_s >= BASELINE_CONTAMINATION_THRESHOLD_S:
        return 'baseline_window_post_led'

    # All criteria passed — trial is valid.
    return None
