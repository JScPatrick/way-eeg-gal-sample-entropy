#!/usr/bin/env python3
"""
Check that this checkout contains the CORRECTED analysis code.

Two corrections were applied to this pipeline after the first round of
results (see README). Both are easy to lose by copying an older file back
over a newer one, and neither fails loudly at runtime -- the pipeline would
simply produce the earlier, uncorrected numbers.

Run this before trusting any output from this repository:

    python verify_corrections.py

Exit status 0 = corrected, 1 = at least one correction is missing.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')

CHECKS = [
    ('config.py', "SAMPEN_TOLERANCE_MODE = 'per_epoch'",
     'CF1: SampEn tolerance r held fixed across the window series'),
    ('config.py', '_TERMINAL_STRICT_PRE      = [-634.0, -510.0, -386.0, -262.0]',
     'W1: terminal windows are the strictly pre-onset set'),
    ('config.py', "TERMINAL_WINDOW_RULE = 'strict_pre_onset'",
     'W1: strict pre-onset rule is the active one'),
    ('entropy.py', 'r=r_ch',
     'CF1: the per-epoch r is actually passed to each sliding window'),
    ('stats.py', 'np.isin(centres_arr, TERMINAL_WINDOW_CENTRES_MS)',
     'B2: terminal windows selected by membership, not by a range filter'),
    ('stats.py', 'SLIDE_WIN_LAST_SAMPLE_OFFSET_MS',
     'W1: runtime assertion that every terminal window is pre-onset'),
]

ANTI = [
    ('config.py', 'TERMINAL_WINDOW_CENTRES_MS = [-386.0, -262.0, -138.0, -14.0]',
     'W1 REVERTED: this is the contaminated terminal set '
     '(windows reach +234 ms post-onset)'),
]


def main():
    bad = 0
    print('Checking for the corrected analysis code\n' + '-' * 62)
    for fname, needle, why in CHECKS:
        path = os.path.join(SRC, fname)
        if not os.path.exists(path):
            print(f'  MISSING FILE  src/{fname}')
            bad += 1
            continue
        text = open(path, encoding='utf-8', errors='ignore').read()
        ok = needle in text
        print(f'  {"ok  " if ok else "FAIL"}  src/{fname:<12s} {why}')
        bad += not ok

    for fname, needle, why in ANTI:
        path = os.path.join(SRC, fname)
        if os.path.exists(path):
            text = open(path, encoding='utf-8', errors='ignore').read()
            if needle in text:
                print(f'  FAIL  src/{fname:<12s} {why}')
                bad += 1

    print('-' * 62)
    if bad:
        print(f'{bad} check(s) FAILED. This checkout does NOT match the '
              f'published results.\nDo not use these outputs.')
        return 1
    print('All checks passed: this is the corrected pipeline.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
