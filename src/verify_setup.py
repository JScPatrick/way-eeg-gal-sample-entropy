"""
verify_setup.py
Run this before starting analysis. Confirms all files are present,
packages are importable, and the .mat structure loads as expected.
"""

import os
import glob
import sys

# ── 1. Package imports ──────────────────────────────────────────────────────
print("Checking packages...")
try:
    import numpy as np
    import scipy.io
    import scipy.signal
    import mne
    import pandas as pd
    import matplotlib
    import pingouin
    print("  All packages imported successfully.\n")
except ImportError as e:
    print(f"  MISSING PACKAGE: {e}")
    sys.exit(1)

# ── 2. sampen fallback check ────────────────────────────────────────────────
print("Checking sampen fallback...")
try:
    sys.path.insert(0, '.')
    from src.sampen import sample_entropy
    test_result = sample_entropy(np.random.randn(500), order=2)
    assert isinstance(test_result, float) and not np.isnan(test_result)
    print(f"  sample_entropy working. Test value: {test_result:.4f}\n")
except Exception as e:
    print(f"  sampen.py error: {e}\n")

# ── 3. File counts ──────────────────────────────────────────────────────────
mat_dir   = "data/mat"
ws_files  = sorted(glob.glob(os.path.join(mat_dir, "WS_P*.mat")))
hs_files  = sorted(glob.glob(os.path.join(mat_dir, "HS_P*.mat")))
all_files = sorted(glob.glob(os.path.join(mat_dir, "P*_AllLifts.mat")))
st_files  = sorted(glob.glob(os.path.join(mat_dir, "HS_P*_ST.mat")))
chan_file  = "data/chanlabels_32channel.xyz"

print("File inventory:")
print(f"  WS (windowed) files : {len(ws_files):>4}  (expect ~96)")
print(f"  HS (holistic) files : {len(hs_files):>4}")
print(f"  AllLifts files      : {len(all_files):>4}  (expect 12)")
print(f"  ST files (no EMG)   : {len(st_files):>4}  (expect 0)")
print(f"  Channel coords file : {'FOUND' if os.path.exists(chan_file) else 'MISSING'}\n")

if len(all_files) != 12:
    print("WARNING: Expected 12 AllLifts files. Some participants may be missing.")
if st_files:
    print("WARNING: ST files still present. Delete them before running analysis.")
if not os.path.exists(chan_file):
    print(f"WARNING: {chan_file} is missing. main.py will run Phases 0-4 and")
    print("         then FAIL in Phase 5 (figure generation) with a")
    print("         FileNotFoundError. Get chanlabels_32channel.xyz from")
    print("         WAY-EEG-GAL Supplementary File 2 (Utilities.zip) and place")
    print("         it at the path above before starting a long run.")

# ── 4. Load and inspect one WS file ────────────────────────────────────────
test_file = os.path.join(mat_dir, "WS_P1_S1.mat")
if not os.path.exists(test_file):
    print(f"Cannot find {test_file} — check extraction completed.")
    sys.exit(1)

print(f"Loading {test_file}...")
raw = scipy.io.loadmat(test_file, squeeze_me=True, struct_as_record=False)
ws  = raw["ws"]

print("\n  Top-level ws fields :", [f for f in dir(ws) if not f.startswith("_")])

trial = ws.win[0]
print("  Trial 0 fields      :", [f for f in dir(trial) if not f.startswith("_")])
print(f"  EEG shape           : {trial.eeg.shape}  (expect (samples, 32))")
print(f"  EMG shape           : {trial.emg.shape}  (expect (samples, 5))")
print(f"  EEG time range      : {trial.eeg_t[0]:.3f} s  to  {trial.eeg_t[-1]:.3f} s")
print(f"  EMG time range      : {trial.emg_t[0]:.3f} s  to  {trial.emg_t[-1]:.3f} s")
print(f"  EEG sampling rate   : {1/np.mean(np.diff(trial.eeg_t)):.0f} Hz  (expect 500)")
print(f"  EMG sampling rate   : {1/np.mean(np.diff(trial.emg_t)):.0f} Hz  (expect 4000)")
print(f"  LEDon time          : {trial.LEDon:.3f} s")
print(f"  EEG channel names   : {list(ws.names.eeg)}")
print(f"  EMG channel names   : {list(ws.names.emg)}")

# ── 5. Load and inspect one AllLifts file ──────────────────────────────────
allifts_file = os.path.join(mat_dir, "P1_AllLifts.mat")
if os.path.exists(allifts_file):
    print(f"\nLoading {allifts_file}...")
    al = scipy.io.loadmat(allifts_file, squeeze_me=True, struct_as_record=False)
    P  = al["P"]
    print(f"  AllLifts shape      : {P.AllLifts.shape}  (expect (n_trials, 43))")
    print(f"  Column names (first 10): {list(P.ColNames[:10])}")
else:
    print(f"\nWARNING: {allifts_file} not found.")

print("\n── Verification complete. Copy this entire output into your Cursor prompt. ──")
