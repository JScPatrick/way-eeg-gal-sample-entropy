"""
sampen.py
Standalone sample entropy implementation.
No dependencies beyond numpy.
Drop-in replacement for antropy.sample_entropy().

Usage:
    # from within src/ (how the pipeline modules import it):
    from sampen import sample_entropy
    # from the repository root:
    from src.sampen import sample_entropy

    value = sample_entropy(signal, order=2, r=None)
"""

import numpy as np


def sample_entropy(x, order=2, r=None):
    """
    Compute sample entropy of a 1D time series.

    Parameters
    ----------
    x     : array-like, 1D signal (e.g. one EEG channel epoch)
    order : int, embedding dimension m (default 2, standard for EEG)
    r     : float, tolerance threshold. If None, defaults to 0.2 * std(x).
            Always pass r=None when calling per-epoch so that tolerance
            scales with each epoch's own amplitude — this is mandatory
            for amplitude-invariant entropy estimation.

    Returns
    -------
    float : sample entropy value, or np.nan if undefined (no template
            matches found, which can occur in very flat or artifacted
            epochs — these should be flagged and excluded downstream).

    Notes
    -----
    Uses Chebyshev distance (max absolute difference across dimensions),
    which is the standard metric for EEG sample entropy in the literature
    and matches the antropy implementation this replaces.
    Excludes self-matches in template counting (correct SampEn definition).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < order + 2:
        # Signal too short to form any templates
        return np.nan

    if r is None:
        r = 0.2 * np.std(x, ddof=1)

    if r == 0:
        # Perfectly flat signal — entropy is undefined
        return np.nan

    # Build template matrices using stride tricks (memory-efficient)
    # Shape: (n - order, order) for length-m templates
    #        (n - order, order + 1) for length-(m+1) templates
    def make_templates(length):
        shape   = (n - length + 1, length)
        strides = (x.strides[0], x.strides[0])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    tm   = make_templates(order)       # (n - order,     order)
    tm1  = make_templates(order + 1)   # (n - order - 1, order + 1)

    # Count template matches using vectorised Chebyshev distance
    # B: number of matching pairs of length-m templates (excluding self)
    # A: number of matching pairs of length-(m+1) templates (excluding self)
    def count_pairs(templates, tol):
        n_t = len(templates)
        count = 0
        # Row-by-row to avoid O(n^2 * m) memory for large n
        for i in range(n_t):
            dists = np.max(np.abs(templates[i] - templates), axis=1)
            # Subtract 1 to exclude self-match
            count += np.sum(dists < tol) - 1
        return count

    B = count_pairs(tm,  r)
    A = count_pairs(tm1, r)

    if B == 0 or A == 0:
        return np.nan

    return float(-np.log(A / B))