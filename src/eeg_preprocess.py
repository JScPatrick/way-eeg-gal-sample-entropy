"""
eeg_preprocess.py — EEG preprocessing for the WAY-EEG-GAL entropy pipeline.

One public function:
    preprocess_eeg   Bandpass filter + common average reference

Raw EEG unit note
-----------------
Raw values are in 10-µV units (200 raw ≈ 20 µV). The ×0.1 conversion to µV
is applied ONLY in visualization.py when labelling figure axes. It is NOT
applied here because SampEn uses a per-epoch tolerance r = 0.2 × std(signal),
which scales with the signal's own units — the entropy values are therefore
identical regardless of whether the raw or µV scale is used.

All parameters imported from config. No magic numbers in this module.

References
----------
Bandpass design:  Proakis & Manolakis (2007), Digital Signal Processing.
CAR reference:    Nunez & Srinivasan (2006), Electric Fields of the Brain.
MRCP / BP band:   Shibasaki & Hallett (2006), Brain.
"""

import numpy as np
import scipy.signal

from config import (
    EEG_SR,
    EEG_BP_LOW, EEG_BP_HIGH,
    FILTER_ORDER, FILTFILT_PADLEN_EEG,
    N_CHANNELS,
)


# ---------------------------------------------------------------------------
# EEG preprocessing
# ---------------------------------------------------------------------------

def preprocess_eeg(eeg, eeg_t):
    """
    Apply zero-phase bandpass filter and common average reference to raw EEG.

    Processing chain:
        raw EEG (10-µV units) → zero-phase Butterworth bandpass → CAR

    No ICA, no artifact rejection, no amplitude thresholding. The original
    WAY-EEG-GAL authors applied none of these, so omitting them preserves
    comparability with the source dataset documentation.

    Parameters
    ----------
    eeg : numpy.ndarray, shape (4714, 32)
        Raw EEG in 10-µV units (hardware bandpass 0.016–1000 Hz, anti-aliased
        to 500 Hz by the original authors). Do NOT multiply by 0.1 here.
    eeg_t : numpy.ndarray, shape (4714,)
        EEG time vector in seconds, window-relative (starts at 0.002 s).

    Returns
    -------
    eeg_filt : numpy.ndarray, shape (4714, 32), dtype float64
        Preprocessed EEG in the same 10-µV units as the input.
        Unit conversion to µV (×0.1) is performed only in visualization.py.

    Raises
    ------
    ValueError
        If eeg contains NaN values or if eeg.shape does not match
        (len(eeg_t), N_CHANNELS). Both conditions indicate a corrupt trial
        that must be excluded before further processing.

    Notes
    -----
    Bandpass 1.0–40.0 Hz:
        Lower bound 1.0 Hz — this is a deliberate trade-off. It attenuates
        some of the Bereitschaftspotential (BP) low-frequency energy (the BP
        onset is ~0.5 Hz), but avoids the extreme filter ringing and baseline
        drift that arise below 1 Hz on short epochs. The MRCP readiness
        potential (0.5–3 Hz, peak ~500 ms before onset) is largely preserved.

    padlen = FILTFILT_PADLEN_EEG = 1500 samples = 3.0 s:
        At the 1 Hz lower cutoff, one filter time constant ≈ 500 samples.
        Three time constants = 1500 samples. This padding guarantees that
        filter transients decay completely before the pre-onset baseline
        window, preventing artificially low or high baseline SampEn values.

    padtype='odd':
        Reflects the signal anti-symmetrically at each edge. Preferred over
        'even' (symmetric) or zero-padding for continuous physiological
        recordings because it avoids introducing an artificial step at the
        boundary that would be amplified by the low-frequency pass band.

    Common Average Reference (CAR):
        Subtracts the mean across all 32 channels at every time point.
        Removes spatially uniform artefacts (movement, power line noise
        picked up coherently across the cap). Standard reference strategy
        for 32-channel BrainProducts ActiCap systems.
        Applied after bandpass so that the reference subtraction operates on
        the already-filtered signals (avoids referencing broadband noise into
        the filtered passband).
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if np.any(np.isnan(eeg)):
        raise ValueError(
            f'NaN detected in raw EEG (shape {eeg.shape}) — trial must be excluded.'
        )

    expected_shape = (len(eeg_t), N_CHANNELS)
    if eeg.shape != expected_shape:
        raise ValueError(
            f'EEG shape mismatch: got {eeg.shape}, expected {expected_shape}. '
            f'Check that the correct WS file was loaded and ws.eeg was not '
            f'transposed during mat loading.'
        )

    # ------------------------------------------------------------------
    # Zero-phase Butterworth bandpass filter (1.0–40.0 Hz)
    # ------------------------------------------------------------------
    # Design the filter coefficients once and reuse across all 32 channels.
    # scipy.signal.butter with btype='bandpass' returns the IIR filter
    # in transfer-function (b, a) form. Order=4 with zero-phase application
    # (filtfilt) gives an effective 8th-order rolloff (filter is applied twice).
    b, a = scipy.signal.butter(
        FILTER_ORDER,
        [EEG_BP_LOW, EEG_BP_HIGH],
        btype='bandpass',
        fs=EEG_SR
    )

    # Allocate output array in float64 to avoid numerical precision loss
    # when adding the CAR (which subtracts a mean of up to ~32 channels).
    eeg_filt = np.zeros_like(eeg, dtype=np.float64)

    for ch in range(N_CHANNELS):
        # Cast channel data to float64 before filtering.
        # Raw EEG may be stored as float32 or int16 in the .mat file;
        # scipy filtfilt requires float64 for numerical stability at these
        # filter coefficients.
        eeg_filt[:, ch] = scipy.signal.filtfilt(
            b, a,
            eeg[:, ch].astype(np.float64),
            padtype='odd',
            padlen=FILTFILT_PADLEN_EEG   # = 1500 samples = 3.0 s at 500 Hz
        )

    # ------------------------------------------------------------------
    # Common Average Reference (CAR)
    # ------------------------------------------------------------------
    # Subtract the instantaneous mean across all N_CHANNELS channels.
    # keepdims=True makes the mean shape (N_samples, 1) so it broadcasts
    # correctly against eeg_filt shape (N_samples, N_CHANNELS) without
    # requiring an explicit transpose.
    eeg_filt -= np.mean(eeg_filt, axis=1, keepdims=True)

    return eeg_filt
