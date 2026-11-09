"""
load_data.py — Single point of contact for reading all .mat files.

All other modules import from this module. No other module calls
scipy.io.loadmat directly. This centralises file-access logic and keeps the
field-access patterns (squeeze_me=True, struct_as_record=False) in one place.

Functions
---------
load_ws_series          Load a WS_PX_SY.mat file → ws struct + trial list
load_allifts            Load a PX_AllLifts.mat file → matrix + column names
get_series_allifts_rows Filter and validate AllLifts rows for one series
get_trial_allifts       Return the AllLifts row for a single trial
"""

import os
import numpy as np
import scipy.io

# Import only the column-index constants needed for validation here.
# All other constants (file paths, participant count, etc.) are passed in
# from the caller so this module does not depend on config at runtime
# (making it independently testable with synthetic data).
from config import COL_RUN, COL_LIFT


# ---------------------------------------------------------------------------
# WS file loader
# ---------------------------------------------------------------------------

def load_ws_series(participant, series, data_dir):
    """
    Load a single WS_PX_SY.mat file and return the ws struct and trial list.

    The WS files contain time-locked EEG and EMG windows for every lift trial
    in one recording series. Each element of ws.win is one trial struct.

    Parameters
    ----------
    participant : int
        Participant number, 1–12.
    series : int
        Series number, 1–9.
    data_dir : str
        Path to the flat directory containing all .mat files (config.DATA_DIR).

    Returns
    -------
    ws : scipy.io.matlab.mio5_params.MatlabObject or None
        Top-level ws struct with fields: name, names, participant, series, win.
        None if file not found.
    trials : list of MatlabObject or None
        ws.win converted to a Python list; trials[n] is the nth trial struct
        (0-indexed). None if file not found.
    found : bool
        False when the file does not exist — caller must skip this series.

    Notes
    -----
    Do NOT call this function more than once per (participant, series) pair.
    The caller (main.py) caches the returned trials list and iterates over it.
    Reloading on every trial would multiply I/O cost by ~32 (trials per series).

    Confirmed trial struct fields:
        LEDoff, LEDon, eeg, eeg_t, emg, emg_t, kin, surf, surf_id,
        surf_prev, surf_prev_id, trial_end_time, trial_start_time,
        weight, weight_id, weight_prev, weight_prev_id

    Confirmed signal shapes:
        trial.eeg   (4714, 32) float — EEG at 500 Hz
        trial.emg   (37712, 5) float — EMG at 4000 Hz
        trial.eeg_t (4714,)    seconds, starts at 0.002
        trial.emg_t (37712,)   seconds, starts at 0.00025
    """
    filepath = os.path.join(data_dir, f'WS_P{participant}_S{series}.mat')

    if not os.path.exists(filepath):
        # Caller logs this as a warning — many (participant, series) pairs
        # are legitimately absent (not every participant has all 9 series).
        return None, None, False

    raw    = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    ws     = raw['ws']

    # ws.win is a numpy object array of MATLAB structs. Converting to a plain
    # Python list makes integer indexing (trials[n]) safe and avoids the
    # numpy object-array behaviour where trials[n] may return a 0-d array.
    trials = list(ws.win)

    return ws, trials, True


# ---------------------------------------------------------------------------
# AllLifts loader
# ---------------------------------------------------------------------------

def load_allifts(participant, data_dir):
    """
    Load PX_AllLifts.mat for one participant.

    AllLifts contains one row per lift trial across ALL series for that
    participant. It is loaded once per participant (not once per series) and
    then filtered by the Run column to obtain series-specific rows.

    Parameters
    ----------
    participant : int
        Participant number, 1–12.
    data_dir : str
        Path to the flat mat directory (config.DATA_DIR).

    Returns
    -------
    allifts_matrix : numpy.ndarray of shape (N_trials, 43) or None
        Full AllLifts matrix for this participant. Each row is one trial;
        each column is one variable (see config.py for column index map).
        None if file not found.
    col_names : list of str or None
        List of 43 column name strings, matching P.ColNames.
        None if file not found.
    found : bool
        False if the file does not exist.

    Notes
    -----
    IMPORTANT column notes:
        col 1  (Run)       — series number; used to filter rows by series
        col 2  (Lift)      — lift index within series (1-indexed)
        col 5  (PrevW)     — NaN for first trial of every series
        col 6  (PrevS)     — NaN for first trial of every series
        col 7  (StartTime) — only ABSOLUTE-time column; all others window-relative
        col 33 (tHandStart)— kinematic onset; SANITY CHECK ONLY, never anchor

    Always guard PrevW and PrevS with np.isnan() before use.
    """
    filepath = os.path.join(data_dir, f'P{participant}_AllLifts.mat')

    if not os.path.exists(filepath):
        return None, None, False

    al        = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    P         = al['P']

    allifts   = P.AllLifts          # shape (N_total_trials, 43) float64 array
    col_names = list(P.ColNames)    # list of 43 strings confirming column order

    return allifts, col_names, True


# ---------------------------------------------------------------------------
# Series-level AllLifts row extraction and validation
# ---------------------------------------------------------------------------

def get_series_allifts_rows(allifts_matrix, series):
    """
    Extract and validate the AllLifts rows for a specific recording series.

    The AllLifts matrix contains trials from all series interleaved. This
    function filters by the Run column (col 1), sorts by the Lift column
    (col 2) to guarantee ordering, and validates that the lift sequence is
    strictly sequential with no duplicates or gaps.

    Parameters
    ----------
    allifts_matrix : numpy.ndarray, shape (N_total, 43)
        Full AllLifts matrix for one participant (from load_allifts).
    series : int
        Series number (1-indexed), matching the Run column value.

    Returns
    -------
    rows : numpy.ndarray of shape (N_series_trials, 43) or None
        Rows for this series, sorted ascending by Lift column.
        None if validation fails.
    valid : bool
        False if no rows found or lift column validation fails.
        Caller must skip this series entirely when valid=False.
    error_msg : str or None
        Human-readable explanation of validation failure. None if valid=True.

    Notes
    -----
    The Lift column (index 2) must be strictly sequential (e.g. 1,2,3,...).
    np.diff(lift_col) must equal 1 at every step — no duplicates, no gaps.
    If this fails the ws.win↔AllLifts alignment cannot be trusted and the
    entire series must be dropped to avoid silently misaligning trial data.

    Alignment rule (enforced in main.py):
        ws.win[0]  ↔  filtered_rows[0]
        ws.win[1]  ↔  filtered_rows[1]   ...etc.
    This is only valid after confirming len(ws.win) == len(filtered_rows).
    """
    # Filter rows where the Run column matches the requested series number.
    # COL_RUN = 1 (0-indexed), confirmed from P.ColNames.
    mask = allifts_matrix[:, COL_RUN] == series
    rows = allifts_matrix[mask]

    if len(rows) == 0:
        return None, False, f'No AllLifts rows found for series {series}'

    # Sort by Lift column (index 2) ascending.
    # The data should already be sorted, but we enforce it explicitly to
    # guarantee the ws.win index alignment holds regardless of file order.
    sort_idx = np.argsort(rows[:, COL_LIFT])
    rows     = rows[sort_idx]

    # Validate that the Lift column is strictly sequential (no duplicates, no gaps).
    # np.diff returns differences between consecutive elements.
    # All diffs must equal 1 for the sequence to be 1,2,3,...
    lift_col = rows[:, COL_LIFT]
    diffs    = np.diff(lift_col)

    if not np.all(diffs == 1):
        # Identify specific problem indices to aid debugging.
        bad_idx = np.where(diffs != 1)[0]
        bad_vals = lift_col[bad_idx]
        return (
            None, False,
            f'Lift column not strictly sequential for series {series}. '
            f'Non-unit diffs at positions {bad_idx.tolist()}, '
            f'lift values {bad_vals.tolist()}. '
            f'Skipping series to prevent trial misalignment.'
        )

    return rows, True, None


# ---------------------------------------------------------------------------
# Single-trial AllLifts row accessor
# ---------------------------------------------------------------------------

def get_trial_allifts(series_rows, trial_idx_0based):
    """
    Return the AllLifts row for a specific trial within a series.

    Parameters
    ----------
    series_rows : numpy.ndarray, shape (N_series_trials, 43)
        Validated and sorted AllLifts rows for the series (from
        get_series_allifts_rows).
    trial_idx_0based : int
        0-based trial index. trial_idx_0based=0 corresponds to ws.win[0]
        and series_rows[0], confirming the alignment contract.

    Returns
    -------
    row : numpy.ndarray, shape (43,)
        The AllLifts row for this trial. All event times are window-relative
        except column 7 (StartTime).

    Notes
    -----
    This is a simple indexing helper. It exists to make the alignment contract
    explicit at every call site rather than relying on bare array indexing,
    which would hide the ws.win ↔ AllLifts correspondence assumption.
    """
    return series_rows[trial_idx_0based]
