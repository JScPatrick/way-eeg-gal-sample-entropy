"""
stats.py — All statistical analyses for the WAY-EEG-GAL entropy pipeline.

One public entry point:
    run_all_statistics(trial_df, sw_df, centres_ms, log_fn)

Internal helpers (also importable for unit testing):
    _scalar_prepost_one_participant   Within-participant pre/post test (32 ch)
    _group_level_scalar               Group-level one-sample test (32 ch)
    _cluster_permutation_test         MNE temporal cluster test
    _slope_permutation_test           Sign-flip permutation on pre-onset slope
    _condition_generalizability       rm_ANOVA or Friedman across 3 weights

Statistical design rationale
-----------------------------
Two distinct directional questions are answered and reported separately:

  Direction 1 — Scalar pre/post:
    Is mean SampEn averaged over [-2000, 0] ms higher or lower than mean
    SampEn averaged over [0, +2000] ms?
    Test: paired t-test or Wilcoxon per participant × channel.
    Correction: BH-FDR across 32 channels per participant.
    BH-FDR is valid here because channels are not temporally overlapping.

  Direction 2 — Pre-onset slope:
    Does SampEn increase or decrease monotonically as movement approaches?
    Test: OLS slope per participant on pre-onset window centres; sign-flip
    permutation test on the mean slope across 12 participants.
    Permutation is used because 75% window overlap makes OLS residuals
    correlated, invalidating OLS standard errors and p-values.

  Temporal precedence (core test):
    Does the entropy change begin before t=0 (EMG onset)?
    Test: MNE cluster-based permutation test on (12, N_windows) matrix.
    Cluster testing handles the correlated time-points correctly
    (Maris & Oostenveld, 2007, J Neurosci Methods).

All parameters imported from config. No magic numbers.
"""

import numpy as np
import scipy.stats
import pandas as pd
from statsmodels.stats.multitest import multipletests
from mne.stats import permutation_cluster_1samp_test
import pingouin

from config import (
    ALPHA, N_PARTICIPANTS, N_CHANNELS,
    N_PERMUTATIONS, N_JOBS,
    EEG_CHANNEL_NAMES, WEIGHT_MAP,
    COL_CURW, COL_PREVW,
)


# ---------------------------------------------------------------------------
# Helper: within-participant scalar pre/post test (Direction 1)
# ---------------------------------------------------------------------------

def _scalar_prepost_one_participant(pre_arr, post_arr):
    """
    Run paired test (t-test or Wilcoxon) and BH-FDR correction for one
    participant across all 32 EEG channels.

    Parameters
    ----------
    pre_arr : numpy.ndarray, shape (N_valid_trials, 32)
        Pre-onset SampEn per trial × channel for this participant.
        May contain NaN (excluded by nan_policy='omit' or masked explicitly).
    post_arr : numpy.ndarray, shape (N_valid_trials, 32)
        Post-onset SampEn per trial × channel. Same shape as pre_arr.

    Returns
    -------
    results : list of dict, length 32
        One dict per channel with keys:
            channel_idx, channel_name,
            test_used ('ttest' | 'wilcoxon'),
            statistic, p_raw, p_fdr,
            effect_size (Cohen's d for t-test; rank-biserial r for Wilcoxon),
            significant (bool, p_fdr < ALPHA),
            direction_pre_vs_post ('higher_pre' | 'lower_pre'),
            n_trials (number of non-NaN pairs used)

    Notes
    -----
    Shapiro-Wilk p > 0.05 → parametric (paired t-test).
    Shapiro-Wilk p ≤ 0.05 → non-parametric (Wilcoxon signed-rank).
    Shapiro-Wilk is tested on the difference vector (pre - post), not on
    pre and post separately, because normality of the difference is what
    the paired t-test requires.

    FDR correction is applied jointly across ALL 32 channels at once so that
    the family-wise error rate is controlled over the full channel set.
    """
    p_raw_32 = np.full(N_CHANNELS, np.nan)
    stats_32  = np.full(N_CHANNELS, np.nan)
    es_32     = np.full(N_CHANNELS, np.nan)
    test_32   = [None] * N_CHANNELS
    n_32      = np.zeros(N_CHANNELS, dtype=int)
    dir_32    = [None] * N_CHANNELS

    for ch in range(N_CHANNELS):
        pre  = pre_arr[:, ch]
        post = post_arr[:, ch]

        # Build mask of pairs where BOTH pre and post are non-NaN.
        valid = ~np.isnan(pre) & ~np.isnan(post)
        n     = np.sum(valid)
        n_32[ch] = int(n)

        if n < 3:
            # Too few observations for any test; leave NaN.
            continue

        pre_v  = pre[valid]
        post_v = post[valid]
        diff   = pre_v - post_v   # positive → higher pre-onset entropy

        dir_32[ch] = 'higher_pre' if np.nanmean(diff) > 0 else 'lower_pre'

        # Normality test on the difference (required assumption for paired t).
        _, p_shapiro = scipy.stats.shapiro(diff)

        if p_shapiro > 0.05:
            # Difference is consistent with normality — use paired t-test.
            t, p = scipy.stats.ttest_rel(pre_v, post_v)
            # Cohen's d for paired data: mean(diff) / std(diff, ddof=1).
            # ddof=1 gives the sample standard deviation (unbiased estimator).
            d = np.mean(diff) / np.std(diff, ddof=1)
            stats_32[ch] = t
            p_raw_32[ch] = p
            es_32[ch]    = d
            test_32[ch]  = 'ttest'
        else:
            # Difference is non-normal — use Wilcoxon signed-rank test.
            # zero_method='wilcox' discards zero differences (default).
            W, p = scipy.stats.wilcoxon(diff, zero_method='wilcox')
            # Rank-biserial correlation as effect size for Wilcoxon:
            # r = 1 - (2W) / (n(n+1)).  Range [-1, +1]; |r| ≥ 0.3 = medium.
            n_w  = len(diff)
            r    = 1.0 - (2.0 * W) / (n_w * (n_w + 1))
            stats_32[ch] = W
            p_raw_32[ch] = p
            es_32[ch]    = r
            test_32[ch]  = 'wilcoxon'

    # ------------------------------------------------------------------
    # BH-FDR correction jointly across all 32 channels.
    # NaN p-values (channels with < 3 trials) are replaced with 1.0 so
    # multipletests does not crash, then the corrected p is set back to NaN.
    # ------------------------------------------------------------------
    p_input = np.where(np.isnan(p_raw_32), 1.0, p_raw_32)
    reject, p_fdr, _, _ = multipletests(p_input, alpha=ALPHA, method='fdr_bh')

    # Restore NaN for channels that had insufficient data.
    p_fdr[np.isnan(p_raw_32)] = np.nan
    reject[np.isnan(p_raw_32)] = False

    results = []
    for ch in range(N_CHANNELS):
        results.append({
            'channel_idx':          ch,
            'channel_name':         EEG_CHANNEL_NAMES[ch],
            'test_used':            test_32[ch],
            'statistic':            float(stats_32[ch]),
            'p_raw':                float(p_raw_32[ch]),
            'p_fdr':                float(p_fdr[ch]),
            'effect_size':          float(es_32[ch]),
            'significant':          bool(reject[ch]),
            'direction_pre_vs_post': dir_32[ch],
            'n_trials':             int(n_32[ch]),
        })

    return results


# ---------------------------------------------------------------------------
# Helper: group-level scalar pre/post test (Direction 1, group inference)
# ---------------------------------------------------------------------------

def _group_level_scalar(participant_diffs):
    """
    Test whether the pre/post SampEn difference generalises across participants.

    The unit of inference at group level is the participant mean, not the
    individual trial. This avoids pseudoreplication from treating trials as
    independent observations across participants.

    Parameters
    ----------
    participant_diffs : numpy.ndarray, shape (N_participants, 32)
        Mean (pre - post) SampEn per participant per channel.
        NaN for participants with no valid trials on a channel.

    Returns
    -------
    group_results : list of dict, length 32
        One dict per channel with keys:
            channel_idx, channel_name,
            test_used, statistic, p_raw, p_fdr,
            group_effect_size, group_significant, group_direction

    Notes
    -----
    One-sample test against zero: does the mean difference differ from zero?
    Shapiro-Wilk on the N_participants values per channel → t-test or Wilcoxon.
    BH-FDR across 32 channels jointly.
    Significant at group level → effect generalises across participants
    (required for the "reliably" criterion in the research question).
    """
    p_raw_32  = np.full(N_CHANNELS, np.nan)
    stats_32  = np.full(N_CHANNELS, np.nan)
    es_32     = np.full(N_CHANNELS, np.nan)
    test_32   = [None] * N_CHANNELS
    dir_32    = [None] * N_CHANNELS

    for ch in range(N_CHANNELS):
        vals  = participant_diffs[:, ch]
        valid = vals[~np.isnan(vals)]

        if len(valid) < 3:
            continue

        dir_32[ch] = 'higher_pre' if np.mean(valid) > 0 else 'lower_pre'

        _, p_shapiro = scipy.stats.shapiro(valid)

        if p_shapiro > 0.05:
            # One-sample t-test: is group mean significantly different from 0?
            t, p = scipy.stats.ttest_1samp(valid, popmean=0.0)
            d = np.mean(valid) / np.std(valid, ddof=1)   # Cohen's d
            stats_32[ch] = t
            p_raw_32[ch] = p
            es_32[ch]    = d
            test_32[ch]  = 'ttest_1samp'
        else:
            # Wilcoxon signed-rank against zero.
            W, p = scipy.stats.wilcoxon(valid, zero_method='wilcox')
            n_w  = len(valid)
            r    = 1.0 - (2.0 * W) / (n_w * (n_w + 1))
            stats_32[ch] = W
            p_raw_32[ch] = p
            es_32[ch]    = r
            test_32[ch]  = 'wilcoxon_1samp'

    p_input = np.where(np.isnan(p_raw_32), 1.0, p_raw_32)
    reject, p_fdr, _, _ = multipletests(p_input, alpha=ALPHA, method='fdr_bh')
    p_fdr[np.isnan(p_raw_32)]  = np.nan
    reject[np.isnan(p_raw_32)] = False

    group_results = []
    for ch in range(N_CHANNELS):
        group_results.append({
            'channel_idx':       ch,
            'channel_name':      EEG_CHANNEL_NAMES[ch],
            'test_used':         test_32[ch],
            'statistic':         float(stats_32[ch]),
            'p_raw':             float(p_raw_32[ch]),
            'p_fdr':             float(p_fdr[ch]),
            'group_effect_size': float(es_32[ch]),
            'group_significant': bool(reject[ch]),
            'group_direction':   dir_32[ch],
        })

    return group_results


# ---------------------------------------------------------------------------
# Helper: MNE cluster permutation test (temporal precedence)
# ---------------------------------------------------------------------------

def _cluster_permutation_test(participant_mean_norm, centres_ms):
    """
    Test whether entropy deviates from baseline at any time point, and whether
    significant deviations precede EMG onset (temporal precedence claim).

    Uses MNE's permutation_cluster_1samp_test on a (12, N_windows) matrix
    of channel-averaged, baseline-normalised SampEn values.

    Parameters
    ----------
    participant_mean_norm : numpy.ndarray, shape (N_participants, N_windows, 32)
        Baseline-normalised SampEn per participant × window × channel.
        NaN cells handled by nanmean in channel averaging step.
    centres_ms : list of float, length N_windows
        Centre time of each sliding window in milliseconds (from epoching.py).

    Returns
    -------
    cluster_results : list of dict
        One dict per SIGNIFICANT cluster (cluster_pv < ALPHA) with keys:
            cluster_id, cluster_p_value,
            time_start_ms, time_end_ms,
            n_windows, mean_t_statistic,
            direction ('above_baseline' | 'below_baseline'),
            epoch ('pre' | 'post' | 'spanning')
    first_sig_pre_window_ms : float or NaN
        Centre time of the earliest window in the earliest significant
        pre-onset cluster (centre_ms < 0). NaN if no pre-onset cluster is
        significant. This is the empirical answer to "precede" in the
        research question.
    recovery_ms : float or NaN
        Centre time of the first post-onset window NOT in any significant
        cluster, after the end of the last significant post-onset cluster.
        NaN if no significant post-onset clusters.
    T_obs : numpy.ndarray, shape (N_windows,)
        Observed t-statistic at each window (for Figure 3).
    all_cluster_pv : list of float
        p-values for ALL clusters (including non-significant), for logging.

    Notes
    -----
    Channel averaging before the cluster test:
        X = nanmean(participant_mean_norm, axis=2)  →  shape (12, N_windows)
        This collapses the spatial dimension to produce a single temporal test.
        The spatial distribution of effects is reported separately via the
        scalar analysis (channel_stats.csv) and Figure 2 (topomap).
        A single temporal test avoids the additional complexity of
        spatiotemporal cluster correction, which would require a different
        adjacency matrix and is beyond the scope of this analysis.

    tail=0 (two-tailed):
        Entropy could increase OR decrease around movement — direction is
        determined post-hoc from the sign of T_obs within significant clusters.
        A one-tailed test would pre-commit to a direction and risk missing an
        effect in the opposite direction.

    threshold=None:
        MNE derives the cluster-forming t-threshold from the t-distribution
        at p=CLUSTER_ALPHA (0.05) two-tailed for N=12 participants (df=11).
        This avoids hardcoding a threshold that would need updating if the
        number of participants changed.

    seed=42: fixed for reproducibility.
    """
    centres_ms_arr = np.array(centres_ms)
    N_windows      = len(centres_ms)

    # ------------------------------------------------------------------
    # Average across 32 channels to get (N_participants, N_windows).
    # nanmean ignores NaN cells from channels with undefined entropy.
    # ------------------------------------------------------------------
    X = np.nanmean(participant_mean_norm, axis=2)   # (12, N_windows)

    assert X.shape == (N_PARTICIPANTS, N_windows), (
        f'Cluster test input shape error: got {X.shape}, '
        f'expected ({N_PARTICIPANTS}, {N_windows}). '
        f'Check participant_mean_norm construction in main.py.'
    )

    # Replace any remaining NaN (participants with no valid trials at all)
    # with zero before passing to MNE, which does not accept NaN.
    # This is conservative: treating absent data as zero (baseline level)
    # rather than excluding the participant entirely, which would change N.
    if np.any(np.isnan(X)):
        n_nan = np.sum(np.isnan(X))
        print(f'  [WARNING] {n_nan} NaN cells in cluster test input — '
              f'replacing with 0 (baseline). Check per-participant NaN rates.')
        X = np.nan_to_num(X, nan=0.0)

    # ------------------------------------------------------------------
    # MNE cluster permutation test (one-sample, two-tailed).
    # ------------------------------------------------------------------
    # threshold=None → MNE computes the t-threshold internally from
    # scipy.stats.t.ppf(1 - CLUSTER_ALPHA/2, df=N_PARTICIPANTS-1).
    # n_jobs=N_JOBS uses all available cores for the permutation loop.
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X,
        n_permutations=N_PERMUTATIONS,
        threshold=None,        # derive from t-distribution at CLUSTER_ALPHA
        tail=0,                # two-tailed: entropy can increase or decrease
        n_jobs=N_JOBS,
        seed=42,               # fixed seed for reproducibility
        verbose=False,
    )

    all_cluster_pv = list(cluster_pv)

    # ------------------------------------------------------------------
    # Parse significant clusters.
    # ------------------------------------------------------------------
    cluster_results          = []
    first_sig_pre_window_ms  = np.nan
    recovery_ms              = np.nan

    sig_post_window_indices  = set()   # collect all post-onset windows in sig clusters

    for c_idx, (cluster_mask, pv) in enumerate(zip(clusters, cluster_pv)):
        # clusters[c] is a tuple of arrays; [0] gives the window-axis indices.
        window_indices = cluster_mask[0]

        t_start = float(centres_ms_arr[window_indices[0]])
        t_end   = float(centres_ms_arr[window_indices[-1]])

        # Direction: above or below baseline (sign of mean T in cluster).
        mean_t  = float(np.mean(T_obs[window_indices]))
        direction = 'above_baseline' if mean_t > 0 else 'below_baseline'

        # Epoch label: where does the cluster fall relative to onset?
        if t_end < 0.0:
            epoch = 'pre'
        elif t_start >= 0.0:
            epoch = 'post'
        else:
            epoch = 'spanning'   # cluster straddles t=0

        if pv >= ALPHA:
            # Non-significant cluster — record p-value for log but skip dict.
            continue

        # Significant cluster — record full details.
        cluster_results.append({
            'cluster_id':       c_idx,
            'cluster_p_value':  float(pv),
            'time_start_ms':    t_start,
            'time_end_ms':      t_end,
            'n_windows':        len(window_indices),
            'mean_t_statistic': mean_t,
            'direction':        direction,
            'epoch':            epoch,
        })

        # Track earliest pre-onset significant window.
        if epoch in ('pre', 'spanning'):
            pre_windows_in_cluster = [
                centres_ms_arr[i] for i in window_indices
                if centres_ms_arr[i] < 0.0
            ]
            if pre_windows_in_cluster:
                candidate = float(min(pre_windows_in_cluster))
                if np.isnan(first_sig_pre_window_ms) or \
                        candidate < first_sig_pre_window_ms:
                    first_sig_pre_window_ms = candidate

        # Collect post-onset windows in significant clusters for recovery calc.
        if epoch in ('post', 'spanning'):
            for i in window_indices:
                if centres_ms_arr[i] >= 0.0:
                    sig_post_window_indices.add(int(i))

    # ------------------------------------------------------------------
    # Compute recovery_ms: first post-onset window NOT in any significant
    # cluster, appearing after the last post-onset significant window.
    # ------------------------------------------------------------------
    if sig_post_window_indices:
        last_sig_post = max(sig_post_window_indices)
        # Search windows after last_sig_post that are post-onset and not in sig clusters.
        for i in range(last_sig_post + 1, N_windows):
            if centres_ms_arr[i] >= 0.0 and i not in sig_post_window_indices:
                recovery_ms = float(centres_ms_arr[i])
                break

    return (
        cluster_results,
        first_sig_pre_window_ms,
        recovery_ms,
        T_obs,
        all_cluster_pv,
    )


# ---------------------------------------------------------------------------
# Helper: slope permutation test on pre-onset SampEn (Direction 2)
# ---------------------------------------------------------------------------

def _slope_permutation_test(participant_mean_norm, centres_ms):
    """
    Test whether SampEn changes monotonically during the pre-onset period
    (-2000 ms to 0 ms) using a sign-flip permutation test on OLS slopes.

    This answers Direction 2: "Does SampEn increase or decrease as movement
    approaches within the pre-onset period?"

    Parameters
    ----------
    participant_mean_norm : numpy.ndarray, shape (N_participants, N_windows, 32)
        Baseline-normalised SampEn per participant × window × channel.
    centres_ms : list of float
        Window centre times in milliseconds.

    Returns
    -------
    observed_mean_slope : float
        Mean OLS slope across 12 participants on the pre-onset windows.
        Units: SampEn units per millisecond.
    slope_direction : str
        'increasing' if observed_mean_slope > 0 (SampEn rises toward onset).
        'decreasing' if observed_mean_slope < 0 (SampEn falls toward onset).
    p_slope : float
        Two-sided permutation p-value: proportion of |null slopes| ≥ |observed|.
    slope_significant : bool
        True if p_slope < ALPHA.
    slopes : numpy.ndarray, shape (N_participants,)
        Per-participant OLS slopes (for reporting and Figure 5).

    Notes
    -----
    Why permutation and not a t-test on the slopes:
        OLS assumes independent residuals. With 75 % window overlap, adjacent
        SampEn values are strongly correlated, making OLS residuals correlated.
        This inflates the apparent degrees of freedom, producing OLS standard
        errors that are too small and t-test p-values that are anti-conservative.
        A sign-flip permutation test avoids distributional assumptions entirely:
        under H0 (no systematic slope), the sign of each participant's slope is
        arbitrary, so randomly flipping signs generates a valid null distribution.

    Why OLS slope is still computed (despite the above):
        OLS slope is the best linear unbiased estimator of the slope magnitude
        under any correlation structure (Gauss-Markov). The issue is only with
        its standard error / p-value — not with the slope estimate itself.
        We use the OLS slope as our effect measure and test it via permutation.

    Seed = 42 for reproducibility (matches cluster permutation test seed).
    """
    centres_ms_arr = np.array(centres_ms)

    # Identify pre-onset windows (centre strictly before t=0).
    pre_mask    = centres_ms_arr < 0.0
    pre_centres = centres_ms_arr[pre_mask]   # shape (N_pre_windows,)

    if len(pre_centres) < 2:
        # Cannot fit a slope with fewer than 2 points.
        print('  [WARNING] Fewer than 2 pre-onset windows — slope test skipped.')
        return np.nan, 'undefined', np.nan, False, np.full(N_PARTICIPANTS, np.nan)

    # Channel-average the normalised SampEn to get (N_participants, N_windows).
    X         = np.nanmean(participant_mean_norm, axis=2)   # (12, N_windows)
    pre_X     = X[:, pre_mask]                              # (12, N_pre_windows)

    # ------------------------------------------------------------------
    # Compute per-participant OLS slope on the pre-onset window series.
    # ------------------------------------------------------------------
    slopes = np.full(N_PARTICIPANTS, np.nan)

    for p in range(N_PARTICIPANTS):
        y     = pre_X[p, :]
        valid = ~np.isnan(y)
        if np.sum(valid) < 2:
            continue   # leave NaN for this participant
        # scipy.stats.linregress returns (slope, intercept, r, p, se).
        # We use only the slope (index 0).
        result      = scipy.stats.linregress(pre_centres[valid], y[valid])
        slopes[p]   = result.slope

    valid_slopes         = slopes[~np.isnan(slopes)]
    observed_mean_slope  = float(np.mean(valid_slopes))
    n_valid              = len(valid_slopes)

    if n_valid < 2:
        print('  [WARNING] Fewer than 2 valid participant slopes — '
              'slope permutation test skipped.')
        return (observed_mean_slope, 'undefined', np.nan, False, slopes)

    # ------------------------------------------------------------------
    # Sign-flip permutation test.
    # Under H0: each participant's slope is equally likely to be positive
    # or negative. Randomly flipping signs generates N_PERMUTATIONS null
    # mean slopes. p = proportion of |null| ≥ |observed| (two-sided).
    # ------------------------------------------------------------------
    rng            = np.random.default_rng(seed=42)
    null_dist      = np.zeros(N_PERMUTATIONS)

    for i in range(N_PERMUTATIONS):
        # Draw n_valid random signs ∈ {-1, +1} with equal probability.
        signs        = rng.choice([-1.0, 1.0], size=n_valid)
        null_dist[i] = float(np.mean(valid_slopes * signs))

    # Two-sided p-value: fraction of permutations with |null| ≥ |observed|.
    p_slope = float(np.mean(np.abs(null_dist) >= np.abs(observed_mean_slope)))

    slope_direction  = 'increasing' if observed_mean_slope > 0 else 'decreasing'
    slope_significant = p_slope < ALPHA

    return (
        observed_mean_slope,
        slope_direction,
        p_slope,
        slope_significant,
        slopes,
    )


# ---------------------------------------------------------------------------
# Helper: terminal pre-onset slope permutation test (secondary/exploratory)
# ---------------------------------------------------------------------------

def _terminal_slope_permutation_test(participant_mean_norm, centres_ms):
    """
    Sign-flip permutation test on the linear slope of channel-averaged
    baseline-normalised SampEn restricted to the four terminal pre-onset
    sliding windows.

    This is a secondary exploratory analysis testing whether entropy shows
    a reliable linear trend over the terminal pre-onset windows.

    The window set is read from TERMINAL_WINDOW_CENTRES_MS in config.py,
    which is the single source of truth shared with the ROI, hemisphere and
    lobe terminal tests. Under TERMINAL_WINDOW_RULE = 'strict_pre_onset' it
    is [-634, -510, -386, -262] ms — the four consecutive windows whose LAST
    SAMPLE precedes EMG onset, spanning [-884, -14] ms.

    This definition is fixed and must not be changed based on inspection of
    results. It was derived from the window boundary arithmetic alone: the
    previous set, [-386, -262, -138, -14] ms, selected windows by centre and
    so admitted windows extending to +110 and +234 ms post-onset, placing
    17.4% of the contributing samples at or after onset.

    Parameters
    ----------
    participant_mean_norm : numpy.ndarray, shape (N_participants, N_windows, 32)
        Baseline-normalised SampEn per participant x window x channel.
        Same array used by _slope_permutation_test and
        _cluster_permutation_test.
    centres_ms : list of float
        Window centre times in milliseconds. Must match the pipeline's
        centres_ms.npy exactly.

    Returns
    -------
    observed_mean_slope : float
        Mean OLS slope across participants on terminal pre-onset windows.
        Units: SampEn per millisecond.
    slope_direction : str
        'decreasing' if observed_mean_slope < 0, 'increasing' if > 0.
    p_terminal : float
        Two-sided permutation p-value using (b+1)/(N+1) correction.
        This correction prevents p=0.000 and maintains correct type-I
        error rate with finite permutations.
    slope_significant : bool
        True if p_terminal < ALPHA.
    slopes : numpy.ndarray, shape (N_participants,)
        Per-participant OLS slopes (for reporting and diagnostics).

    Notes
    -----
    Terminal window definition:
        np.isin(centres_ms, TERMINAL_WINDOW_CENTRES_MS), the same rule the
        ROI, hemisphere and lobe terminal tests use. An assertion verifies
        at runtime that every selected window's last sample
        (centre + SLIDE_WIN_LAST_SAMPLE_OFFSET_MS = centre + 248 ms)
        precedes onset, so a contaminated set cannot pass silently.

    Two-tailed test (tail=0):
        H0: mean slope across participants = 0.
        p = (b + 1) / (N + 1) where b = number of permuted |mean slopes|
        >= |observed mean slope|.
        Direction is reported post-hoc from the sign of observed_mean_slope.

    p-value correction:
        The (b+1)/(N+1) correction is mandatory. It prevents p=0.000,
        maintains correct type-I error rate, and is the standard correction
        for permutation tests with finite permutations.

    Sign-flip permutation:
        Under H0, each participant's slope is equally likely to be positive
        or negative. Randomly flipping signs generates the null distribution.
        This avoids OLS distributional assumptions which are violated by
        the 75% window overlap in the sliding window design.

    Seed = 42 for reproducibility. Matches all other permutation tests
    in the pipeline.
    """
    from config import (TERMINAL_WINDOW_CENTRES_MS, TERMINAL_WINDOW_RULE,
                        SLIDE_WIN_LAST_SAMPLE_OFFSET_MS)

    centres_ms_arr = np.array(centres_ms)

    # W1 CORRECTION (2026-10-25). Previously this site used its own rule,
    # (centres >= -500) & (centres < 0), while the ROI, hemisphere and lobe
    # sites used np.isin against TERMINAL_WINDOW_CENTRES_MS. The two rules
    # happened to select the same four windows, but the duplication meant
    # changing the config constant silently left this test behind.
    # All four sites now read the single config constant.
    terminal_mask    = np.isin(centres_ms_arr, TERMINAL_WINDOW_CENTRES_MS)
    terminal_centres = centres_ms_arr[terminal_mask]

    # Runtime guarantee that every selected window is genuinely pre-onset.
    # A window spans [centre - 250, centre + SLIDE_WIN_LAST_SAMPLE_OFFSET_MS];
    # its last sample must precede EMG onset at t = 0. This assertion is what
    # actually enforces the W1 correction — the config constant alone could be
    # edited back to a contaminated set without anything noticing.
    if TERMINAL_WINDOW_RULE == 'strict_pre_onset':
        last_sample_ms = terminal_centres + SLIDE_WIN_LAST_SAMPLE_OFFSET_MS
        assert np.all(last_sample_ms < 0.0), (
            f'Terminal window set is not strictly pre-onset: centres '
            f'{terminal_centres.tolist()} have last samples '
            f'{last_sample_ms.tolist()} ms, some at or after onset. '
            f'Check TERMINAL_WINDOW_CENTRES_MS in config.py.')

    if len(terminal_centres) < 2:
        print('  [WARNING] Fewer than 2 terminal pre-onset windows found — '
              'terminal slope test skipped.')
        return (np.nan, 'undefined', np.nan, False,
                np.full(N_PARTICIPANTS, np.nan))

    # Channel-average the normalised SampEn:
    # X shape: (N_participants, N_windows)
    X          = np.nanmean(participant_mean_norm, axis=2)
    terminal_X = X[:, terminal_mask]   # (N_participants, N_terminal_windows)

    # Compute per-participant OLS slope on terminal windows only.
    slopes = np.full(N_PARTICIPANTS, np.nan)
    for p in range(N_PARTICIPANTS):
        y     = terminal_X[p, :]
        valid = ~np.isnan(y)
        if np.sum(valid) < 2:
            continue
        result    = scipy.stats.linregress(terminal_centres[valid], y[valid])
        slopes[p] = result.slope

    valid_slopes        = slopes[~np.isnan(slopes)]
    n_valid             = len(valid_slopes)
    observed_mean_slope = float(np.mean(valid_slopes))

    if n_valid < 2:
        print('  [WARNING] Fewer than 2 valid participant slopes for '
              'terminal slope test — test skipped.')
        return (observed_mean_slope, 'undefined', np.nan, False, slopes)

    # Sign-flip permutation test (two-tailed).
    # p = (b + 1) / (N + 1) where:
    #   b = number of permuted |mean slopes| >= |observed mean slope|
    #   N = N_PERMUTATIONS
    # This is the standard correction for finite permutation tests.
    rng       = np.random.default_rng(seed=42)
    null_dist = np.zeros(N_PERMUTATIONS)

    for i in range(N_PERMUTATIONS):
        signs        = rng.choice([-1.0, 1.0], size=n_valid)
        null_dist[i] = float(np.mean(valid_slopes * signs))

    # Count permuted |statistics| >= |observed| (two-tailed).
    b          = int(np.sum(np.abs(null_dist) >= np.abs(observed_mean_slope)))
    p_terminal = float((b + 1) / (N_PERMUTATIONS + 1))

    slope_direction   = 'decreasing' if observed_mean_slope < 0 else 'increasing'
    slope_significant = p_terminal < ALPHA

    return (
        observed_mean_slope,
        slope_direction,
        p_terminal,
        slope_significant,
        slopes,
    )


# ---------------------------------------------------------------------------
# Helper: condition generalizability across weight conditions
# ---------------------------------------------------------------------------

def _condition_generalizability(trial_df, participant_ch_results):
    """
    Test whether the pre/post SampEn effect size is consistent across the
    three weight conditions (165 g, 330 g, 660 g).

    Parameters
    ----------
    trial_df : pandas.DataFrame
        Trial-level entropy data (from results/trial_level_entropy.csv).
        Must have columns: participant, weight, pre_sampen_mean, post_sampen_mean.
    participant_ch_results : list of list of dict
        participant_ch_results[p][ch] = result dict from
        _scalar_prepost_one_participant. Used to get per-participant test_used
        for effect size type labelling.

    Returns
    -------
    cond_result : dict with keys:
        test_used ('rm_anova' | 'friedman'),
        statistic, p_value,
        significant (bool),
        effect_sizes_by_condition : dict {weight_g: mean_cohen_d_across_participants}
        participant_condition_es : numpy.ndarray, shape (N_participants, 3)
            Channel-averaged effect size per participant per condition.
            Column order: [165g, 330g, 660g].

    Notes
    -----
    Effect size per participant per condition:
        For each participant × weight condition, compute Cohen's d of
        (pre_sampen_mean - post_sampen_mean) / std(diff) across valid trials
        in that condition. Then average across the 32 channels is not re-done
        here — instead we use the trial-level pre/post means directly because
        they are already channel-averaged in trial_df.

    PrevW NaN handling:
        First trial of each series has PrevW = NaN. These trials are excluded
        from the condition comparison with np.isnan guard, as instructed.

    Omnibus test selection:
        Shapiro-Wilk per condition column (12 values) → if all normal, use
        pingouin rm_anova; else use scipy.stats.friedmanchisquare.
        No additional FDR correction: this is a single omnibus test on one
        channel-averaged outcome per participant per condition.

    Interpretation:
        p >= ALPHA → no significant difference across conditions → supports
        the "reliably" claim (effect is weight-condition-agnostic).
        p <  ALPHA → effect size differs across conditions → qualified claim.
    """
    weight_codes   = sorted(WEIGHT_MAP.keys())    # [1, 2, 4]
    weight_labels  = [WEIGHT_MAP[w] for w in weight_codes]   # [165, 330, 660]
    n_conditions   = len(weight_codes)

    # participant_condition_es[p, c] = Cohen's d for participant p, condition c
    participant_condition_es = np.full((N_PARTICIPANTS, n_conditions), np.nan)

    for p_idx in range(N_PARTICIPANTS):
        pid = p_idx + 1   # participant IDs are 1-indexed
        p_df = trial_df[trial_df['participant'] == pid].copy()

        for c_idx, w_code in enumerate(weight_codes):
            # Exclude first trial of each series (PrevW is NaN in AllLifts,
            # which maps to NaN in the weight column if carried through).
            # Here we filter on the current weight condition directly.
            cond_df = p_df[p_df['weight'] == w_code].copy()

            if len(cond_df) < 3:
                continue   # too few trials for a meaningful effect size

            diff = cond_df['pre_sampen_mean'].values - \
                   cond_df['post_sampen_mean'].values
            diff = diff[~np.isnan(diff)]

            if len(diff) < 3:
                continue

            # Cohen's d: mean difference / std of differences (ddof=1).
            d = np.mean(diff) / np.std(diff, ddof=1)
            participant_condition_es[p_idx, c_idx] = d

    # ------------------------------------------------------------------
    # Omnibus test: are effect sizes different across the 3 conditions?
    # ------------------------------------------------------------------
    # Extract the three condition columns as 1-D arrays, excluding NaN rows.
    col_arrays = []
    for c_idx in range(n_conditions):
        col = participant_condition_es[:, c_idx]
        col_arrays.append(col)

    # Find participants with valid data in all three conditions.
    all_valid_mask = ~np.any(np.isnan(participant_condition_es), axis=1)
    complete_data  = participant_condition_es[all_valid_mask, :]   # (N_complete, 3)

    if complete_data.shape[0] < 3:
        print('  [WARNING] Fewer than 3 participants with complete condition '
              'data — condition generalizability test skipped.')
        return {
            'test_used':                  'skipped',
            'statistic':                  np.nan,
            'p_value':                    np.nan,
            'significant':                False,
            'effect_sizes_by_condition':  {WEIGHT_MAP[w]: np.nan for w in weight_codes},
            'participant_condition_es':   participant_condition_es,
        }

    # Shapiro-Wilk per condition column on the complete-case data.
    all_normal = True
    for c_idx in range(n_conditions):
        col = complete_data[:, c_idx]
        if len(col) >= 3:
            _, p_sw = scipy.stats.shapiro(col)
            if p_sw <= 0.05:
                all_normal = False
                break
        else:
            all_normal = False
            break

    if all_normal:
        # All three columns are normally distributed — use repeated-measures ANOVA.
        # Build a long-format DataFrame for pingouin.
        rows = []
        p_ids_complete = np.where(all_valid_mask)[0] + 1   # 1-indexed
        for i, pid in enumerate(p_ids_complete):
            for c_idx, w_code in enumerate(weight_codes):
                rows.append({
                    'participant': pid,
                    'condition':   WEIGHT_MAP[w_code],
                    'effect_size': complete_data[i, c_idx],
                })
        long_df = pd.DataFrame(rows)
        anova_result = pingouin.rm_anova(
            data=long_df,
            dv='effect_size',
            within='condition',
            subject='participant',
            correction=True,     # Greenhouse-Geisser correction for sphericity
        )
        stat_val  = float(anova_result['F'].iloc[0])
        # Check which p-value column pingouin returned, in priority order.
        # Different pingouin versions use different column names:
        #   'p-unc'     — older pingouin versions
        #   'p-GG-corr' — when Greenhouse-Geisser correction is applied
        #   'pval'      — some newer pingouin versions
        _p_col = None
        # L1 CORRECTION (2026-10-25): Greenhouse-Geisser columns must be
        # tried BEFORE 'p-unc'. pingouin always emits 'p-unc', so the old
        # ordering always selected the uncorrected p-value even though
        # correction=True was requested above, silently discarding the
        # sphericity correction.
        for _candidate in ['p-GG-corr', 'p_GG_corr', 'p-unc', 'p_unc', 'pval']:
            if _candidate in anova_result.columns:
                _p_col = _candidate
                break

        if _p_col is not None:
            p_val     = float(anova_result[_p_col].iloc[0])
            test_used = 'rm_anova'
        else:
            # No recognizable p-value column found — set all related output
            # fields to safe fallback values consistently and log a warning.
            # Do not crash. Do not attempt to infer a p-value from other columns.
            p_val       = np.nan
            stat_val    = np.nan
            test_used   = 'rm_anova_column_not_found'
            significant = False
    else:
        # At least one condition column is non-normal — use Friedman test.
        stat_val, p_val = scipy.stats.friedmanchisquare(
            complete_data[:, 0],
            complete_data[:, 1],
            complete_data[:, 2],
        )
        stat_val  = float(stat_val)
        p_val     = float(p_val)
        test_used = 'friedman'

    # Mean effect size per condition (across participants with data).
    effect_sizes_by_condition = {}
    for c_idx, w_code in enumerate(weight_codes):
        col = participant_condition_es[:, c_idx]
        effect_sizes_by_condition[WEIGHT_MAP[w_code]] = float(np.nanmean(col))

    return {
        'test_used':                  test_used,
        'statistic':                  stat_val,
        'p_value':                    p_val,
        'significant':                p_val < ALPHA,
        'effect_sizes_by_condition':  effect_sizes_by_condition,
        'participant_condition_es':   participant_condition_es,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_statistics(trial_df, sw_df, centres_ms, log_fn):
    """
    Run all statistical analyses and return structured result dictionaries.

    Parameters
    ----------
    trial_df : pandas.DataFrame
        Trial-level entropy data (results/trial_level_entropy.csv).
        Required columns: participant, series, trial_in_series, weight, surface,
        pre_sampen_ch0 ... pre_sampen_ch31,
        post_sampen_ch0 ... post_sampen_ch31.
    sw_df : pandas.DataFrame
        Sliding-window entropy data (results/sliding_window_entropy.csv).
        Required columns: participant, trial_in_series, window_index,
        window_centre_ms, sampen_norm_ch0 ... sampen_norm_ch31.
    centres_ms : list of float
        Window centre times in milliseconds, from epoching.build_sliding_windows.
    log_fn : callable
        Function accepting a string; writes to pipeline_log.txt.
        Call as log_fn('message') from this module.

    Returns
    -------
    stats_output : dict with keys:
        'participant_ch_results'     list[N_participants] of list[32] of dict
        'participant_summary'        list[N_participants] of dict
        'channel_stats'              list[32] of dict (merged within + group)
        'cluster_results'            list of dict (significant clusters only)
        'first_sig_pre_window_ms'    float or NaN
        'recovery_ms'                float or NaN
        'T_obs'                      numpy array (N_windows,)
        'all_cluster_pv'             list of float
        'slope_result'               dict
        'condition_result'           dict
        'participant_mean_norm'      numpy array (N_participants, N_windows, 32)
    """
    N_windows = len(centres_ms)
    log_fn('='*60)
    log_fn('STATISTICS: starting run_all_statistics')
    log_fn(f'  N_windows={N_windows}, N_participants={N_PARTICIPANTS}, '
           f'N_channels={N_CHANNELS}')

    # ------------------------------------------------------------------
    # Build per-participant arrays from DataFrames
    # ------------------------------------------------------------------
    # Pre/post SampEn per channel columns.
    pre_cols  = [f'pre_sampen_ch{ch}'  for ch in range(N_CHANNELS)]
    post_cols = [f'post_sampen_ch{ch}' for ch in range(N_CHANNELS)]
    norm_cols = [f'sampen_norm_ch{ch}' for ch in range(N_CHANNELS)]

    participant_ch_results = []   # [p][ch] = result dict
    participant_summary    = []   # [p] = summary dict
    participant_diffs      = np.full((N_PARTICIPANTS, N_CHANNELS), np.nan)

    # participant_mean_norm[p, w, ch] = mean baseline-normalised SampEn
    participant_mean_norm = np.full(
        (N_PARTICIPANTS, N_windows, N_CHANNELS), np.nan
    )

    # ------------------------------------------------------------------
    # Direction 1: within-participant scalar pre/post analysis
    # ------------------------------------------------------------------
    log_fn('--- Direction 1: within-participant scalar pre/post ---')

    for p_idx in range(N_PARTICIPANTS):
        pid   = p_idx + 1
        p_df  = trial_df[trial_df['participant'] == pid]
        n_valid = len(p_df)

        if n_valid == 0:
            log_fn(f'  P{pid}: no valid trials — skipping.')
            participant_ch_results.append([{} for _ in range(N_CHANNELS)])
            participant_summary.append({'participant': pid,
                                        'n_valid': 0,
                                        'skipped': True})
            continue

        # Extract (N_trials, 32) arrays for pre and post.
        pre_arr  = p_df[pre_cols].values.astype(np.float64)
        post_arr = p_df[post_cols].values.astype(np.float64)

        # Run within-participant test with BH-FDR across 32 channels.
        ch_results = _scalar_prepost_one_participant(pre_arr, post_arr)
        participant_ch_results.append(ch_results)

        # Compute per-participant, per-channel mean difference for group test.
        for ch in range(N_CHANNELS):
            diff_vals = pre_arr[:, ch] - post_arr[:, ch]
            participant_diffs[p_idx, ch] = np.nanmean(diff_vals)

        # Count significant channels for summary.
        n_sig = sum(r['significant'] for r in ch_results if r.get('significant') is not None)
        log_fn(f'  P{pid}: {n_valid} trials, {n_sig}/32 channels significant (FDR).')

        # ------------------------------------------------------------------
        # Build participant_mean_norm from sliding-window DataFrame.
        # ------------------------------------------------------------------
        p_sw = sw_df[sw_df['participant'] == pid]

        for w_idx, w_centre in enumerate(centres_ms):
            # Select rows for this window index.
            # window_centre_ms should match centres_ms exactly (float equality
            # may be fragile — use window_index instead for robustness).
            w_rows = p_sw[p_sw['window_index'] == w_idx]

            if len(w_rows) == 0:
                continue

            # Mean across trials for each channel.
            norm_vals = w_rows[norm_cols].values.astype(np.float64)
            participant_mean_norm[p_idx, w_idx, :] = np.nanmean(norm_vals, axis=0)

        # Build participant summary row (filled further after group/cluster tests).
        mean_pre  = np.nanmean(pre_arr)
        mean_post = np.nanmean(post_arr)
        diff_mean = mean_pre - mean_post
        participant_summary.append({
            'participant':       pid,
            'n_valid':           n_valid,
            'skipped':           False,
            'mean_pre_sampen':   float(mean_pre),
            'mean_post_sampen':  float(mean_post),
            'mean_diff_sampen':  float(diff_mean),
            'direction_pre_vs_post': ('higher_pre' if diff_mean > 0
                                      else 'lower_pre'),
        })

    # ------------------------------------------------------------------
    # Direction 1: group-level test
    # ------------------------------------------------------------------
    log_fn('--- Direction 1: group-level scalar pre/post ---')
    group_results = _group_level_scalar(participant_diffs)

    n_group_sig = sum(r['group_significant'] for r in group_results
                      if r.get('group_significant'))
    log_fn(f'  Group level: {n_group_sig}/32 channels significant (FDR).')

    # Merge within-participant (ch-level) and group results into channel_stats.
    # Use the last participant's ch_results as the within-participant template;
    # the actual per-participant results live in participant_ch_results.
    # channel_stats stores GROUP-level results only (one row per channel).
    channel_stats = []
    for ch in range(N_CHANNELS):
        g = group_results[ch]

        # Aggregate within-participant statistics across participants for
        # the channel-level summary: use nanmean of per-participant values.
        pre_means  = [participant_diffs[p, ch] for p in range(N_PARTICIPANTS)
                      if not np.isnan(participant_diffs[p, ch])]
        mean_diff  = float(np.nanmean(participant_diffs[:, ch]))

        channel_stats.append({
            'channel_idx':          ch,
            'channel_name':         EEG_CHANNEL_NAMES[ch],
            'mean_diff':            mean_diff,
            'direction_pre_vs_post': g['group_direction'],
            'group_test_used':      g['test_used'],
            'group_statistic':      g['statistic'],
            'group_p_raw':          g['p_raw'],
            'group_p_fdr':          g['p_fdr'],
            'group_effect_size':    g['group_effect_size'],
            'group_significant':    g['group_significant'],
            'group_direction':      g['group_direction'],
        })

    # ------------------------------------------------------------------
    # Temporal precedence: cluster permutation test
    # ------------------------------------------------------------------
    log_fn('--- Temporal precedence: cluster permutation test ---')
    log_fn(f'  N_permutations={N_PERMUTATIONS}, seed=42, tail=0 (two-sided).')

    (cluster_results,
     first_sig_pre_window_ms,
     recovery_ms,
     T_obs,
     all_cluster_pv) = _cluster_permutation_test(
         participant_mean_norm, centres_ms
     )

    n_sig_clusters = len(cluster_results)
    log_fn(f'  Significant clusters: {n_sig_clusters}')
    for cr in cluster_results:
        log_fn(f'    Cluster {cr["cluster_id"]}: p={cr["cluster_p_value"]:.4f}, '
               f't=[{cr["time_start_ms"]:.0f}, {cr["time_end_ms"]:.0f}] ms, '
               f'epoch={cr["epoch"]}, direction={cr["direction"]}')

    if not np.isnan(first_sig_pre_window_ms):
        log_fn(f'  First significant PRE-onset window: '
               f'{first_sig_pre_window_ms:.1f} ms')
    else:
        log_fn('  No significant pre-onset cluster found.')

    if not np.isnan(recovery_ms):
        log_fn(f'  Recovery (first non-sig post-onset window): {recovery_ms:.1f} ms')

    # Add cluster test results to participant summaries.
    for ps in participant_summary:
        ps['first_sig_pre_window_ms'] = float(first_sig_pre_window_ms) \
            if not np.isnan(first_sig_pre_window_ms) else np.nan
        ps['recovery_ms'] = float(recovery_ms) \
            if not np.isnan(recovery_ms) else np.nan

    # ------------------------------------------------------------------
    # Direction 2: slope permutation test on pre-onset windows
    # ------------------------------------------------------------------
    log_fn('--- Direction 2: pre-onset slope permutation test ---')

    (observed_mean_slope,
     slope_direction,
     p_slope,
     slope_significant,
     slopes) = _slope_permutation_test(participant_mean_norm, centres_ms)

    log_fn(f'  Mean slope: {observed_mean_slope:.6f} SampEn/ms, '
           f'direction: {slope_direction}, '
           f'p={p_slope:.4f}, significant={slope_significant}')

    slope_result = {
        'observed_mean_slope': float(observed_mean_slope)
            if not np.isnan(observed_mean_slope) else np.nan,
        'slope_direction':     slope_direction,
        'p_slope':             float(p_slope)
            if not np.isnan(p_slope) else np.nan,
        'slope_significant':   slope_significant,
        'slopes_per_participant': slopes,
    }

    # Add slope results to participant summaries.
    for p_idx, ps in enumerate(participant_summary):
        ps['mean_slope']       = float(slopes[p_idx]) \
            if not np.isnan(slopes[p_idx]) else np.nan
        ps['slope_direction']  = slope_direction
        ps['slope_p_permutation'] = float(p_slope) \
            if not np.isnan(p_slope) else np.nan
        ps['slope_significant']   = slope_significant

    # ------------------------------------------------------------------
    # Secondary analysis: terminal pre-onset slope (last ~400 ms)
    # Tests whether entropy shows a reliable linear trend in the final
    # four pre-onset windows (centres: -386, -262, -138, -14 ms).
    # Two-tailed sign-flip permutation test, p-value uses (b+1)/(N+1)
    # correction. This is exploratory and post-hoc relative to the
    # cluster permutation test.
    # ------------------------------------------------------------------
    from config import (TERMINAL_WINDOW_CENTRES_MS as _TWC,
                        TERMINAL_WINDOW_RULE as _TWR)

    log_fn('--- Terminal pre-onset slope test (secondary/exploratory) ---')
    log_fn(f'  Window set ({_TWR}): '
           f'{[int(c) for c in _TWC]} ms centres, {len(_TWC)} windows; '
           f'spans [{int(min(_TWC))-250}, '
           f'{int(max(_TWC))+248}] ms relative to EMG onset')
    log_fn('  Two-tailed, N_permutations=10000, seed=42, '
           'p-value correction: (b+1)/(N+1)')

    (terminal_slope,
     terminal_direction,
     p_terminal,
     terminal_significant,
     terminal_slopes) = _terminal_slope_permutation_test(
         participant_mean_norm, centres_ms
     )

    log_fn(f'  Mean slope: {terminal_slope:.6f} SampEn/ms, '
           f'direction: {terminal_direction}, '
           f'p={p_terminal:.4f}, significant={terminal_significant}')

    terminal_slope_result = {
        'observed_mean_slope':      float(terminal_slope)
                                    if not np.isnan(terminal_slope)
                                    else np.nan,
        'slope_direction':          terminal_direction,
        'p_terminal':               float(p_terminal)
                                    if not np.isnan(p_terminal)
                                    else np.nan,
        'slope_significant':        terminal_significant,
        'slopes_per_participant':   terminal_slopes,
        'window_centres_ms':        [-386.0, -262.0, -138.0, -14.0],
        'n_windows':                4,
        'test_type':                'sign_flip_permutation_two_tailed',
        'p_correction':             '(b+1)/(N+1)',
    }

    # ------------------------------------------------------------------
    # Condition generalizability
    # ------------------------------------------------------------------
    log_fn('--- Condition generalizability across weight conditions ---')

    condition_result = _condition_generalizability(
        trial_df, participant_ch_results
    )

    log_fn(f'  Test: {condition_result["test_used"]}, '
           f'stat={condition_result["statistic"]:.4f}, '
           f'p={condition_result["p_value"]:.4f}, '
           f'significant={condition_result["significant"]}')
    for w_g, es in condition_result['effect_sizes_by_condition'].items():
        log_fn(f'    {w_g}g: mean_d={es:.4f}')

    log_fn('STATISTICS: run_all_statistics complete.')
    log_fn('='*60)

    return {
        'participant_ch_results':   participant_ch_results,
        'participant_summary':      participant_summary,
        'channel_stats':            channel_stats,
        'cluster_results':          cluster_results,
        'first_sig_pre_window_ms':  first_sig_pre_window_ms,
        'recovery_ms':              recovery_ms,
        'T_obs':                    T_obs,
        'all_cluster_pv':           all_cluster_pv,
        'slope_result':             slope_result,
        'terminal_slope_result':    terminal_slope_result,
        'condition_result':         condition_result,
        'participant_mean_norm':    participant_mean_norm,
    }


# ---------------------------------------------------------------------------
# Spatial analysis helpers
# ---------------------------------------------------------------------------

def _get_channel_indices(channel_names_subset, all_channel_names):
    """
    Map a list of channel name strings to integer indices into
    all_channel_names. Raises ValueError if any name is not found.
    Must be called at the start of every new spatial analysis
    function to verify channel mapping before any computation.

    Parameters
    ----------
    channel_names_subset : list of str
        Channel names to look up (e.g. SENSORIMOTOR_ROI_CHANNELS).
    all_channel_names : list of str
        The full ordered channel name list (EEG_CHANNEL_NAMES).

    Returns
    -------
    list of int
        Integer indices into all_channel_names, in the same order
        as channel_names_subset.

    Raises
    ------
    ValueError
        If any name in channel_names_subset is not found in
        all_channel_names.
    """
    indices = []
    for name in channel_names_subset:
        if name not in all_channel_names:
            raise ValueError(
                f'Channel "{name}" not found in EEG_CHANNEL_NAMES. '
                f'Check config.py channel group definitions.')
        indices.append(all_channel_names.index(name))
    return indices


def _subset_average(participant_mean_norm, channel_indices):
    """
    Average participant_mean_norm across a channel subset.

    Parameters
    ----------
    participant_mean_norm : np.ndarray, shape (N_participants,
                            N_windows, 32)
        Baseline-normalised SampEn. This array is NEVER modified.
        A view is taken by indexing axis 2 only.
    channel_indices : list of int
        Integer indices into axis 2 (channel axis).

    Returns
    -------
    np.ndarray, shape (N_participants, N_windows)
        Mean across the specified channels at each window for each
        participant.
    """
    return np.nanmean(
        participant_mean_norm[:, :, channel_indices], axis=2)


# ---------------------------------------------------------------------------
# Sensorimotor ROI parallel analysis framework
# ---------------------------------------------------------------------------

def run_roi_analyses(stats_output, trial_df, centres_ms, log_fn,
                     sw_df=None):
    """
    Sensorimotor ROI parallel analysis framework.

    ROI channels: FC1, FC2, C3, Cz, C4, CP1, CP2 (7 channels).
    Defined in config.py as SENSORIMOTOR_ROI_CHANNELS.

    Runs the following tests, all using identical parameters to
    their whole-scalp equivalents:
      1. Cluster permutation test (full time window)
      2. Full pre-onset slope test (sign-flip permutation)
      3. Terminal slope test (sign-flip permutation, 4 windows)
      4. Scalar pre/post comparison (Wilcoxon)
      5. Condition generalizability (rm_anova, 3 weight conditions)

    IMPORTANT: Test 5 derives ROI pre-onset means directly from
    roi_matrix (which is already ROI-channel-averaged and
    trial-averaged), NOT from trial_df['pre_sampen_mean'] which
    is the whole-scalp mean. Using trial_df['pre_sampen_mean']
    would be a silent scientific error. The weight condition
    labels are obtained from trial_df for grouping only.

    participant_mean_norm is read from stats_output and never
    modified.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics. Must contain key
        'participant_mean_norm' with shape (12, N_windows, 32).
    trial_df : pd.DataFrame
        Trial-level entropy data. Used only to obtain weight
        condition labels per participant for Test 5. Must contain
        columns 'participant' and 'weight'.
    centres_ms : list of float
        Sliding window centre times in milliseconds.
    log_fn : callable
        Logging function that writes to pipeline_log.txt.

    Returns
    -------
    dict with keys:
        cluster_results         : list of dicts (significant clusters)
        full_slope_mean         : float
        full_slope_direction    : str ('increasing' or 'decreasing')
        full_slope_p            : float
        terminal_slope_mean     : float
        terminal_slope_direction: str
        terminal_slope_p        : float
        prepost_direction       : str
        prepost_p               : float
        condition_F             : float
        condition_p             : float
        condition_significant   : bool
        roi_matrix              : np.ndarray shape (12, N_windows)
    """
    from config import (
        SENSORIMOTOR_ROI_CHANNELS, EEG_CHANNEL_NAMES,
        TERMINAL_WINDOW_CENTRES_MS, N_PERMUTATIONS, ALPHA,
        N_PARTICIPANTS
    )
    from mne.stats import permutation_cluster_1samp_test

    log_fn('=== Sensorimotor ROI analyses ===')
    log_fn(f'  ROI channels: {SENSORIMOTOR_ROI_CHANNELS}')

    # Verify channel mapping before any computation.
    roi_idx = _get_channel_indices(
        SENSORIMOTOR_ROI_CHANNELS, EEG_CHANNEL_NAMES)
    log_fn(f'  ROI channel indices verified: {roi_idx}')

    # Extract ROI-averaged matrix. participant_mean_norm NOT modified.
    pmn = stats_output['participant_mean_norm']
    roi_matrix = _subset_average(pmn, roi_idx)
    # roi_matrix shape: (N_PARTICIPANTS, N_windows)

    # Shape verification — must pass before any further computation.
    if roi_matrix.ndim != 2 or roi_matrix.shape[0] != N_PARTICIPANTS:
        raise ValueError(
            f'roi_matrix has unexpected shape {roi_matrix.shape}. '
            f'Expected ({N_PARTICIPANTS}, N_windows). '
            f'Check _subset_average and participant_mean_norm.')
    log_fn(f'  roi_matrix shape verified: {roi_matrix.shape}')

    centres_arr = np.array(centres_ms)

    # ----------------------------------------------------------------
    # Test 1: ROI cluster permutation test (full time window)
    # Parameters: N_PERMUTATIONS, seed=42, tail=0 (two-tailed)
    # ----------------------------------------------------------------
    log_fn('--- ROI cluster permutation test ---')
    (roi_T_obs, roi_clusters, roi_cluster_pv,
     roi_H0) = permutation_cluster_1samp_test(
        roi_matrix,
        n_permutations=N_PERMUTATIONS,
        tail=0,
        seed=42,
        verbose=False,
        out_type='mask'
    )
    roi_cluster_results = []
    for k, (cl, pv) in enumerate(
            zip(roi_clusters, roi_cluster_pv)):
        if pv < ALPHA:
            mask = cl[0]
            times = centres_arr[mask]
            epoch = ('pre'  if np.all(times < 0)  else
                     'post' if np.all(times >= 0) else
                     'mixed')
            direction = ('above_baseline'
                         if float(np.mean(roi_T_obs[mask])) > 0
                         else 'below_baseline')
            roi_cluster_results.append({
                'cluster_index':   k,
                'cluster_p_value': float(pv),
                'time_start_ms':   float(times[0]),
                'time_end_ms':     float(times[-1]),
                'epoch':           epoch,
                'direction':       direction,
            })
            log_fn(f'  ROI Cluster {k}: p={pv:.6f}, '
                   f'{times[0]:.0f} to {times[-1]:.0f} ms, '
                   f'{epoch}, {direction}')
    if not roi_cluster_results:
        log_fn('  No significant ROI clusters found.')

    # ----------------------------------------------------------------
    # Test 2: ROI full pre-onset slope test
    # Sign-flip permutation on OLS slope across all pre-onset windows.
    # seed=42, two-tailed, (b+1)/(N_PERMUTATIONS+1) correction.
    # ----------------------------------------------------------------
    log_fn('--- ROI full pre-onset slope test ---')
    pre_mask    = centres_arr < 0
    roi_pre     = roi_matrix[:, pre_mask]
    pre_centres = centres_arr[pre_mask]
    roi_slopes_full = np.array([
        scipy.stats.linregress(
            pre_centres, roi_pre[p]).slope
        for p in range(N_PARTICIPANTS)
    ])
    rng1 = np.random.default_rng(seed=42)
    null_full = np.array([
        float(np.mean(
            roi_slopes_full *
            rng1.choice([-1.0, 1.0], size=N_PARTICIPANTS)))
        for _ in range(N_PERMUTATIONS)
    ])
    b_full     = int(np.sum(
        np.abs(null_full) >= np.abs(np.mean(roi_slopes_full))))
    p_roi_full = float((b_full + 1) / (N_PERMUTATIONS + 1))
    roi_slope_dir_full = (
        'decreasing' if np.mean(roi_slopes_full) < 0
        else 'increasing')
    log_fn(f'  ROI full pre-onset slope: '
           f'{float(np.mean(roi_slopes_full)):.6f} SampEn/ms, '
           f'{roi_slope_dir_full}, p={p_roi_full:.4f}')

    # ----------------------------------------------------------------
    # Test 3: ROI terminal slope test
    # Sign-flip permutation on OLS slope across 4 terminal windows.
    # Windows: centres at -386, -262, -138, -14 ms.
    # seed=42, two-tailed, (b+1)/(N_PERMUTATIONS+1) correction.
    # ----------------------------------------------------------------
    log_fn('--- ROI terminal slope test ---')
    term_mask = np.isin(centres_arr, TERMINAL_WINDOW_CENTRES_MS)
    if int(np.sum(term_mask)) != 4:
        raise ValueError(
            f'Expected 4 terminal windows, found '
            f'{int(np.sum(term_mask))}. '
            f'Check TERMINAL_WINDOW_CENTRES_MS in config.py.')
    roi_term     = roi_matrix[:, term_mask]
    term_centres = centres_arr[term_mask]
    roi_slopes_term = np.array([
        scipy.stats.linregress(
            term_centres, roi_term[p]).slope
        for p in range(N_PARTICIPANTS)
    ])
    rng2 = np.random.default_rng(seed=42)
    null_term = np.array([
        float(np.mean(
            roi_slopes_term *
            rng2.choice([-1.0, 1.0], size=N_PARTICIPANTS)))
        for _ in range(N_PERMUTATIONS)
    ])
    b_term     = int(np.sum(
        np.abs(null_term) >= np.abs(np.mean(roi_slopes_term))))
    p_roi_term = float((b_term + 1) / (N_PERMUTATIONS + 1))
    roi_slope_dir_term = (
        'decreasing' if np.mean(roi_slopes_term) < 0
        else 'increasing')
    log_fn(f'  ROI terminal slope: '
           f'{float(np.mean(roi_slopes_term)):.6f} SampEn/ms, '
           f'{roi_slope_dir_term}, p={p_roi_term:.4f}')

    # ----------------------------------------------------------------
    # Test 4: ROI scalar pre/post comparison
    # Single group-level Wilcoxon on ROI-averaged pre vs post means.
    # No FDR correction — single pre-specified ROI comparison.
    # ----------------------------------------------------------------
    log_fn('--- ROI scalar pre/post comparison ---')
    pre_epoch_mask  = centres_arr < 0
    post_epoch_mask = centres_arr >= 0
    roi_pre_mean  = np.nanmean(
        roi_matrix[:, pre_epoch_mask],  axis=1)
    roi_post_mean = np.nanmean(
        roi_matrix[:, post_epoch_mask], axis=1)
    diff_prepost = roi_pre_mean - roi_post_mean
    stat_pp, p_pp = scipy.stats.wilcoxon(diff_prepost)
    roi_prepost_dir = (
        'lower_pre'  if float(np.mean(diff_prepost)) < 0
        else 'higher_pre')
    log_fn(f'  ROI pre mean: '
           f'{float(np.mean(roi_pre_mean)):.4f}, '
           f'post mean: '
           f'{float(np.mean(roi_post_mean)):.4f}, '
           f'direction: {roi_prepost_dir}, '
           f'Wilcoxon stat={stat_pp:.3f}, p={p_pp:.4f}')

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Test 5: ROI condition generalizability
    # Computed from sw_df when provided.
    #
    # Computation:
    # 1. Filter sw_df to pre-onset windows (window_centre_ms<0)
    # 2. Average 7 ROI sampen_norm_ch{k} columns per row
    # 3. Group by (participant, series, trial_in_series, weight)
    #    → mean across pre-onset windows → 1 scalar per trial
    # 4. Group by (participant, weight)
    #    → mean across trials → 1 scalar per participant/condition
    # 5. Run rm_anova (dv='roi_mean', within='weight',
    #    subject='participant')
    # 6. Store group means AND per-participant means for figure
    #
    # ROI channel columns (indices [8,9,12,13,14,18,19]):
    #   sampen_norm_ch8, sampen_norm_ch9, sampen_norm_ch12,
    #   sampen_norm_ch13, sampen_norm_ch14, sampen_norm_ch18,
    #   sampen_norm_ch19
    #
    # Weight codes in sw_df: [1, 2, 4] → ['165g','330g','660g']
    # in sorted order. Must match Figure 5 label mapping.
    # ----------------------------------------------------------------
    log_fn('--- ROI condition generalizability ---')
    roi_cond_stat = np.nan
    roi_cond_p    = np.nan
    roi_cond_sig  = False

    # CRITICAL: Initialize both before all conditionals so they
    # are always defined for the results dictionary assembly.
    cond_means             = {}
    cond_participant_means = {}

    ROI_SW_COLS = [
        'sampen_norm_ch8',  'sampen_norm_ch9',
        'sampen_norm_ch12', 'sampen_norm_ch13',
        'sampen_norm_ch14', 'sampen_norm_ch18',
        'sampen_norm_ch19',
    ]

    if sw_df is None:
        log_fn('[WARNING] sw_df not passed to run_roi_analyses.'
               ' ROI condition generalizability skipped.')
    else:
        # Verify all required columns exist before proceeding.
        missing_cols = [c for c in ROI_SW_COLS
                        if c not in sw_df.columns]
        if missing_cols:
            log_fn(f'[WARNING] Missing ROI columns in sw_df: '
                   f'{missing_cols}. Skipped.')
        elif 'window_centre_ms' not in sw_df.columns:
            log_fn('[WARNING] window_centre_ms not in sw_df.'
                   ' Skipped.')
        elif 'participant' not in sw_df.columns:
            log_fn('[WARNING] participant not in sw_df.'
                   ' Skipped.')
        elif 'weight' not in sw_df.columns:
            log_fn('[WARNING] weight not in sw_df. Skipped.')
        else:
            try:
                # Step 1: Filter to pre-onset windows.
                pre_sw = sw_df[
                    sw_df['window_centre_ms'] < 0].copy()

                # Step 2: Per-row ROI mean across 7 channels.
                pre_sw['roi_mean'] = pre_sw[
                    ROI_SW_COLS].mean(axis=1)

                # Step 3: Mean across pre-onset windows per trial.
                trial_roi = (
                    pre_sw
                    .groupby(
                        ['participant', 'series',
                         'trial_in_series', 'weight'],
                        as_index=False)['roi_mean']
                    .mean()
                )

                # Step 4: Mean across trials per participant
                # per condition.
                part_cond = (
                    trial_roi
                    .groupby(
                        ['participant', 'weight'],
                        as_index=False)['roi_mean']
                    .mean()
                )

                # Step 5: rm_anova.
                import pingouin as pg
                weights_sorted = sorted(
                    part_cond['weight'].unique())
                if len(weights_sorted) != 3:
                    log_fn(
                        f'[WARNING] Expected 3 weight conditions'
                        f', found {len(weights_sorted)}.'
                        f' Skipped.')
                else:
                    rm = pg.rm_anova(
                        data=part_cond,
                        dv='roi_mean',
                        within='weight',
                        subject='participant',
                        correction=True)
                    p_cond_col = next(
                        (c for c in [
                            'p-GG-corr', 'p_GG_corr',
                            'p-unc', 'p_unc',
                            'pval']
                         if c in rm.columns), None)
                    if p_cond_col and 'F' in rm.columns:
                        roi_cond_stat = float(
                            rm['F'].iloc[0])
                        roi_cond_p = float(
                            rm[p_cond_col].iloc[0])
                        roi_cond_sig = roi_cond_p < ALPHA
                        log_fn(
                            f'  ROI condition: '
                            f'F={roi_cond_stat:.4f}, '
                            f'p={roi_cond_p:.4f}, '
                            f'significant={roi_cond_sig}')

                        # Step 6: Store group means and
                        # per-participant means for figure.
                        # Both use the same weights_sorted
                        # order to ensure alignment.
                        participants_sorted = sorted(
                            part_cond['participant'].unique())
                        for w in weights_sorted:
                            # Group mean for bar height.
                            cond_means[w] = float(
                                part_cond.loc[
                                    part_cond['weight'] == w,
                                    'roi_mean'].mean())
                            # Per-participant list for dots.
                            # Order matches participants_sorted.
                            per_part = []
                            for pid in participants_sorted:
                                val = part_cond.loc[
                                    (part_cond['participant']
                                     == pid) &
                                    (part_cond['weight'] == w),
                                    'roi_mean'].values
                                per_part.append(
                                    float(val[0])
                                    if len(val) > 0
                                    else np.nan)
                            cond_participant_means[w] = per_part
                        log_fn(
                            f'  Per-condition means: '
                            f'{cond_means}')
                        log_fn(
                            f'  N participants per condition: '
                            f'{[len(v) for v in cond_participant_means.values()]}')
                    else:
                        log_fn(
                            '[WARNING] Could not find F or '
                            'p-value in ROI rm_anova output.'
                            ' Flag for manual review.')

            except Exception as e:
                log_fn(f'[WARNING] ROI condition rm_anova '
                       f'failed: {e}. Flag for manual review.')

    # ----------------------------------------------------------------
    # Assemble and return results dictionary.
    # ----------------------------------------------------------------
    roi_results = {
        'cluster_results':          roi_cluster_results,
        'full_slope_mean':          float(np.mean(roi_slopes_full)),
        'full_slope_direction':     roi_slope_dir_full,
        'full_slope_p':             p_roi_full,
        'terminal_slope_mean':      float(np.mean(roi_slopes_term)),
        'terminal_slope_direction': roi_slope_dir_term,
        'terminal_slope_p':         p_roi_term,
        'prepost_direction':        roi_prepost_dir,
        'prepost_p':                float(p_pp),
        'condition_F':              roi_cond_stat,
        'condition_p':              roi_cond_p,
        'condition_significant':    roi_cond_sig,
        'roi_matrix':                 roi_matrix,
        'condition_means':            cond_means,
        'condition_participant_means': cond_participant_means,
    }
    return roi_results


# ---------------------------------------------------------------------------
# Hemisphere comparison analysis
# ---------------------------------------------------------------------------

def run_hemisphere_analyses(stats_output, centres_ms, log_fn):
    """
    Hemisphere comparison analysis.

    Left hemisphere channels (11):
        F7, F3, FC5, FC1, T7, C3, CP5, CP1, P7, P3, O1
    Right hemisphere channels (11):
        F8, F4, FC6, FC2, T8, C4, CP6, CP2, P8, P4, O2
    Excluded from hemisphere comparison (10):
        Fp1, Fp2 (ocular artifact)
        Fz, Cz, Pz, Oz (midline)
        TP9, TP10 (mastoid-adjacent)
        PO9, PO10 (occipital visual)

    Tests:
      1. Cluster permutation test on L-R difference matrix
         (full -2000ms to +1998ms window). Identifies WHEN during
         the epoch hemispheres differ significantly.
         Direction: LEFT minus RIGHT throughout. Do not swap.
         Positive clusters = left greater than right.
         Negative clusters = right greater than left.
      2. Scalar Wilcoxon on L-R terminal means
         (windows at -386, -262, -138, -14 ms).
      3. Sign-flip permutation test on slope of L-R difference
         across the 4 terminal windows.
         seed=42, two-tailed, (b+1)/(N_PERMUTATIONS+1) correction.
         Direction label: 'lr_difference_decreasing' means the
         left-minus-right gap is narrowing or becoming more
         negative over the terminal period (this could reflect
         left declining faster OR right rising faster — do not
         label as hemisphere-specific cause).
         Direction label: 'lr_difference_increasing' means the
         left-minus-right gap is widening or becoming more
         positive over the terminal period.

    participant_mean_norm is read from stats_output and never
    modified. _subset_average and _get_channel_indices are called
    from this module.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics. Must contain key
        'participant_mean_norm' shape (12, N_windows, 32).
    centres_ms : list of float
        Sliding window centre times in milliseconds.
    log_fn : callable
        Logging function.

    Returns
    -------
    dict with keys:
        cluster_results          : list of dicts
        terminal_mean_lr         : float (mean L-R terminal mean)
        terminal_mean_direction  : str ('left_greater' or
                                   'right_greater')
        terminal_mean_p          : float
        terminal_slope_mean      : float
        terminal_slope_direction : str ('lr_difference_decreasing'
                                   or 'lr_difference_increasing')
        terminal_slope_p         : float
        left_matrix              : np.ndarray (12, N_windows)
        right_matrix             : np.ndarray (12, N_windows)
        lr_diff                  : np.ndarray (12, N_windows)
    """
    from config import (
        LEFT_HEMISPHERE_CHANNELS, RIGHT_HEMISPHERE_CHANNELS,
        EEG_CHANNEL_NAMES, TERMINAL_WINDOW_CENTRES_MS,
        N_PERMUTATIONS, ALPHA, N_PARTICIPANTS
    )
    from mne.stats import permutation_cluster_1samp_test

    log_fn('=== Hemisphere comparison analyses ===')
    log_fn(f'  Left channels:  {LEFT_HEMISPHERE_CHANNELS}')
    log_fn(f'  Right channels: {RIGHT_HEMISPHERE_CHANNELS}')

    # Verify channel mapping before any computation.
    left_idx  = _get_channel_indices(
        LEFT_HEMISPHERE_CHANNELS, EEG_CHANNEL_NAMES)
    right_idx = _get_channel_indices(
        RIGHT_HEMISPHERE_CHANNELS, EEG_CHANNEL_NAMES)
    log_fn(f'  Left indices verified:  {left_idx}')
    log_fn(f'  Right indices verified: {right_idx}')

    # Extract hemisphere matrices. participant_mean_norm NOT modified.
    pmn          = stats_output['participant_mean_norm']
    left_matrix  = _subset_average(pmn, left_idx)
    right_matrix = _subset_average(pmn, right_idx)

    # Shape verification.
    for name, mat in [('left_matrix',  left_matrix),
                      ('right_matrix', right_matrix)]:
        if mat.ndim != 2 or mat.shape[0] != N_PARTICIPANTS:
            raise ValueError(
                f'{name} has unexpected shape {mat.shape}. '
                f'Expected ({N_PARTICIPANTS}, N_windows).')
    log_fn(f'  left_matrix shape:  {left_matrix.shape}')
    log_fn(f'  right_matrix shape: {right_matrix.shape}')

    # L-R difference matrix.
    # DIRECTION: LEFT minus RIGHT. Do not swap.
    # Positive values = left greater than right.
    # Negative values = right greater than left.
    lr_diff     = left_matrix - right_matrix
    centres_arr = np.array(centres_ms)

    # ----------------------------------------------------------------
    # Test 1: Hemisphere cluster permutation on L-R difference matrix
    # Full -2000ms to +1998ms window.
    # permutation_cluster_1samp_test computes cluster p-values
    # directly from its internal permutation distribution — no
    # additional manual correction is needed or applied here.
    # seed=42, tail=0 (two-tailed), N_PERMUTATIONS.
    # ----------------------------------------------------------------
    log_fn('--- Hemisphere cluster permutation test (L minus R) ---')
    (hemi_T_obs, hemi_clusters, hemi_cluster_pv,
     hemi_H0) = permutation_cluster_1samp_test(
        lr_diff,
        n_permutations=N_PERMUTATIONS,
        tail=0,
        seed=42,
        verbose=False,
        out_type='mask'
    )
    hemi_cluster_results = []
    for k, (cl, pv) in enumerate(
            zip(hemi_clusters, hemi_cluster_pv)):
        if pv < ALPHA:
            mask  = cl[0]
            times = centres_arr[mask]
            epoch = ('pre'  if np.all(times < 0)  else
                     'post' if np.all(times >= 0) else
                     'mixed')
            direction = ('left_greater'
                         if float(np.mean(hemi_T_obs[mask])) > 0
                         else 'right_greater')
            hemi_cluster_results.append({
                'cluster_index':   k,
                'cluster_p_value': float(pv),
                'time_start_ms':   float(times[0]),
                'time_end_ms':     float(times[-1]),
                'epoch':           epoch,
                'direction':       direction,
            })
            log_fn(f'  Hemisphere Cluster {k}: p={pv:.6f}, '
                   f'{times[0]:.0f} to {times[-1]:.0f} ms, '
                   f'{epoch}, {direction}')
    if not hemi_cluster_results:
        log_fn('  No significant hemisphere clusters found.')

    # ----------------------------------------------------------------
    # Test 2: Hemisphere scalar terminal mean comparison
    # Wilcoxon on L-R terminal window means per participant.
    # Terminal windows: -386, -262, -138, -14 ms.
    # ----------------------------------------------------------------
    log_fn('--- Hemisphere scalar terminal mean comparison ---')
    term_mask = np.isin(centres_arr, TERMINAL_WINDOW_CENTRES_MS)
    if int(np.sum(term_mask)) != 4:
        raise ValueError(
            f'Expected 4 terminal windows, found '
            f'{int(np.sum(term_mask))}. '
            f'Check TERMINAL_WINDOW_CENTRES_MS in config.py.')
    lr_term_means = np.nanmean(lr_diff[:, term_mask], axis=1)
    stat_hemi, p_hemi = scipy.stats.wilcoxon(lr_term_means)
    hemi_term_dir = ('left_greater'
                     if float(np.mean(lr_term_means)) > 0
                     else 'right_greater')
    log_fn(f'  Terminal mean L-R: '
           f'{float(np.mean(lr_term_means)):.4f}, '
           f'{hemi_term_dir}, '
           f'Wilcoxon stat={stat_hemi:.3f}, p={p_hemi:.4f}')

    # ----------------------------------------------------------------
    # Test 3: Hemisphere terminal slope sign-flip permutation test.
    # Slope of the L-R DIFFERENCE trajectory across 4 terminal
    # windows. One slope per participant.
    # seed=42, two-tailed, (b+1)/(N_PERMUTATIONS+1) correction.
    #
    # Direction labels:
    #   'lr_difference_decreasing': mean slope < 0, meaning the
    #     left-minus-right gap narrows or becomes more negative
    #     over the terminal period. This could reflect left
    #     declining faster OR right rising faster — do not
    #     interpret as hemisphere-specific without examining
    #     individual trajectories.
    #   'lr_difference_increasing': mean slope > 0, meaning the
    #     left-minus-right gap widens or becomes more positive.
    #
    # NOTE: This is the slope of lr_diff across terminal windows,
    # NOT the difference of separately computed hemisphere slopes.
    # ----------------------------------------------------------------
    log_fn('--- Hemisphere terminal slope permutation test ---')
    term_centres = centres_arr[term_mask]
    hemi_slopes_term = np.array([
        scipy.stats.linregress(
            term_centres, lr_diff[p, term_mask]).slope
        for p in range(N_PARTICIPANTS)
    ])
    rng_h = np.random.default_rng(seed=42)
    null_h = np.array([
        float(np.mean(
            hemi_slopes_term *
            rng_h.choice([-1.0, 1.0], size=N_PARTICIPANTS)))
        for _ in range(N_PERMUTATIONS)
    ])
    b_h = int(np.sum(
        np.abs(null_h) >= np.abs(np.mean(hemi_slopes_term))))
    p_hemi_slope = float(
        (b_h + 1) / (N_PERMUTATIONS + 1))
    # Direction: describes change in L-R difference, not hemisphere-
    # specific cause. See docstring for interpretation guidance.
    hemi_slope_dir = (
        'lr_difference_decreasing'
        if float(np.mean(hemi_slopes_term)) < 0
        else 'lr_difference_increasing')
    log_fn(f'  Hemisphere terminal slope (L-R diff): '
           f'{float(np.mean(hemi_slopes_term)):.6f} SampEn/ms, '
           f'{hemi_slope_dir}, p={p_hemi_slope:.4f}')

    hemi_results = {
        'cluster_results':          hemi_cluster_results,
        'terminal_mean_lr':         float(np.mean(lr_term_means)),
        'terminal_mean_direction':  hemi_term_dir,
        'terminal_mean_p':          float(p_hemi),
        'terminal_slope_mean':      float(
                                    np.mean(hemi_slopes_term)),
        'terminal_slope_direction': hemi_slope_dir,
        'terminal_slope_p':         p_hemi_slope,
        'left_matrix':              left_matrix,
        'right_matrix':             right_matrix,
        'lr_diff':                  lr_diff,
    }
    return hemi_results


# ---------------------------------------------------------------------------
# Lobe comparison analysis
# ---------------------------------------------------------------------------

def run_lobe_analyses(stats_output, centres_ms, log_fn):
    """
    Lobe comparison analysis.

    Four regions (defined in config.py LOBE_GROUPS):
        Frontal:   F7, F3, Fz, F4, F8, FC5, FC6  (7 channels)
        Central:   FC1, FC2, C3, Cz, C4, CP1, CP2 (7 channels)
        Parietal:  CP5, CP6, P7, P3, Pz, P4, P8   (7 channels)
        Occipital: PO9, O1, Oz, O2, PO10           (5 channels)
    Excluded from all lobe analyses: Fp1, Fp2, T7, T8, TP9, TP10
    Temporal lobe excluded from omnibus by design — not in
    LOBE_GROUPS.

    Tests:
      Option A: rm_anova on terminal window MEANS (4 regions).
                Bonferroni post hoc ONLY IF omnibus significant.
                Primary contrasts: central vs each other region.
                No post hoc if omnibus not significant.
      Option B: rm_anova on terminal window OLS SLOPES (4 regions).
                Exploratory only.
      Option C: Cluster permutation on central-minus-other
                DIFFERENCE TRAJECTORIES. Three pre-specified
                contrasts:
                  central minus frontal
                  central minus parietal
                  central minus occipital
                DIRECTION: central minus other for ALL contrasts.
                Do NOT swap direction for any contrast.
                Positive clusters = central greater than other.
                Negative clusters = other greater than central.
                Each contrast: N_PERMUTATIONS, seed=42, tail=0.
                permutation_cluster_1samp_test computes cluster
                p-values from its own internal permutation
                distribution — no additional manual correction
                is applied. This is different from the sign-flip
                tests above which require manual (b+1)/(N+1)
                correction.

    participant_mean_norm is read from stats_output and never
    modified.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics. Must contain key
        'participant_mean_norm' shape (12, N_windows, 32).
    centres_ms : list of float
        Sliding window centre times in milliseconds.
    log_fn : callable
        Logging function.

    Returns
    -------
    dict with keys:
        lobe_matrices      : dict of str -> np.ndarray (12, N_windows)
        lobe_term_means    : dict of str -> np.ndarray (12,)
        lobe_term_slopes   : dict of str -> np.ndarray (12,)
        optionA_F          : float
        optionA_p          : float
        optionA_significant: bool
        optionA_posthoc    : dict
        optionB_F          : float
        optionB_p          : float
        optionB_significant: bool
        optionC_clusters   : dict of str -> list of dicts
    """
    from config import (
        LOBE_GROUPS, EEG_CHANNEL_NAMES,
        TERMINAL_WINDOW_CENTRES_MS,
        N_PERMUTATIONS, ALPHA, N_PARTICIPANTS
    )
    from mne.stats import permutation_cluster_1samp_test
    import pingouin as pg

    log_fn('=== Lobe comparison analyses ===')
    for lobe, chs in LOBE_GROUPS.items():
        log_fn(f'  {lobe}: {chs}')

    # Verify all channel mappings before any computation.
    lobe_indices = {}
    for lobe, chs in LOBE_GROUPS.items():
        lobe_indices[lobe] = _get_channel_indices(
            chs, EEG_CHANNEL_NAMES)
        log_fn(f'  {lobe} indices verified: {lobe_indices[lobe]}')

    # Extract lobe matrices. participant_mean_norm NOT modified.
    pmn         = stats_output['participant_mean_norm']
    centres_arr = np.array(centres_ms)
    term_mask   = np.isin(centres_arr, TERMINAL_WINDOW_CENTRES_MS)
    if int(np.sum(term_mask)) != 4:
        raise ValueError(
            f'Expected 4 terminal windows, found '
            f'{int(np.sum(term_mask))}. '
            f'Check TERMINAL_WINDOW_CENTRES_MS in config.py.')
    term_centres = centres_arr[term_mask]
    lobe_names   = list(LOBE_GROUPS.keys())

    lobe_matrices    = {}
    lobe_term_means  = {}
    lobe_term_slopes = {}
    for lobe, idx in lobe_indices.items():
        mat = _subset_average(pmn, idx)
        if mat.ndim != 2 or mat.shape[0] != N_PARTICIPANTS:
            raise ValueError(
                f'lobe_matrices[{lobe}] has unexpected shape '
                f'{mat.shape}. Expected ({N_PARTICIPANTS}, '
                f'N_windows).')
        lobe_matrices[lobe]    = mat
        lobe_term_means[lobe]  = np.nanmean(
            mat[:, term_mask], axis=1)
        lobe_term_slopes[lobe] = np.array([
            scipy.stats.linregress(
                term_centres, mat[p, term_mask]).slope
            for p in range(N_PARTICIPANTS)
        ])
    log_fn('  All lobe matrices computed and shape-verified.')

    # ----------------------------------------------------------------
    # Option A: rm_anova on terminal window means
    # Post hoc comparisons run ONLY IF omnibus is significant.
    # If omnibus is not significant, no post hoc tests are run.
    # ----------------------------------------------------------------
    log_fn('--- Lobe Option A: rm_anova on terminal means ---')
    optA_df = pd.DataFrame(
        {lobe: lobe_term_means[lobe] for lobe in lobe_names})
    optA_df['participant'] = range(1, N_PARTICIPANTS + 1)
    optA_long = optA_df.melt(
        id_vars='participant',
        var_name='lobe',
        value_name='terminal_mean')
    optA_F   = np.nan
    optA_p   = np.nan
    optA_sig = False
    posthoc_results = {}
    try:
        rmA = pg.rm_anova(
            data=optA_long,
            dv='terminal_mean',
            within='lobe',
            subject='participant',
            correction=True)
        p_col_A = next(
            (c for c in ['p-GG-corr', 'p_GG_corr', 'p-unc',
                          'p_unc', 'pval']
             if c in rmA.columns), None)
        if 'F' in rmA.columns:
            optA_F = float(rmA['F'].iloc[0])
        if p_col_A:
            optA_p   = float(rmA[p_col_A].iloc[0])
            optA_sig = optA_p < ALPHA
        log_fn(f'  Option A omnibus: F={optA_F:.4f}, '
               f'p={optA_p:.4f}, significant={optA_sig}')
        # Bonferroni post hoc ONLY if omnibus is significant.
        # Primary contrasts: central vs each other region.
        if optA_sig:
            log_fn('  Post hoc (Bonferroni, central vs others):')
            n_comp = len(lobe_names) - 1
            for lobe in lobe_names:
                if lobe == 'central':
                    continue
                diff = (lobe_term_means['central'] -
                        lobe_term_means[lobe])
                stat_ph, p_ph = scipy.stats.wilcoxon(diff)
                p_ph_bonf = min(p_ph * n_comp, 1.0)
                posthoc_results[f'central_vs_{lobe}'] = {
                    'stat':          float(stat_ph),
                    'p_uncorrected': float(p_ph),
                    'p_bonferroni':  float(p_ph_bonf),
                    'significant':   p_ph_bonf < ALPHA,
                    'direction':     ('central_higher'
                                      if np.mean(diff) > 0
                                      else 'other_higher'),
                }
                log_fn(f'    central vs {lobe}: '
                       f'p_bonf={p_ph_bonf:.4f}, '
                       f'sig={p_ph_bonf < ALPHA}')
        else:
            log_fn('  Omnibus not significant — '
                   'no post hoc comparisons run.')
    except Exception as e:
        log_fn(f'[WARNING] Lobe Option A rm_anova failed: {e}. '
               f'Flag for manual review.')

    # ----------------------------------------------------------------
    # Option B: rm_anova on terminal window slopes (exploratory)
    # ----------------------------------------------------------------
    log_fn('--- Lobe Option B: rm_anova on terminal slopes '
           '(exploratory) ---')
    optB_df = pd.DataFrame(
        {lobe: lobe_term_slopes[lobe] for lobe in lobe_names})
    optB_df['participant'] = range(1, N_PARTICIPANTS + 1)
    optB_long = optB_df.melt(
        id_vars='participant',
        var_name='lobe',
        value_name='terminal_slope')
    optB_F   = np.nan
    optB_p   = np.nan
    optB_sig = False
    try:
        rmB = pg.rm_anova(
            data=optB_long,
            dv='terminal_slope',
            within='lobe',
            subject='participant',
            correction=True)
        p_col_B = next(
            (c for c in ['p-GG-corr', 'p_GG_corr', 'p-unc',
                          'p_unc', 'pval']
             if c in rmB.columns), None)
        if 'F' in rmB.columns:
            optB_F = float(rmB['F'].iloc[0])
        if p_col_B:
            optB_p   = float(rmB[p_col_B].iloc[0])
            optB_sig = optB_p < ALPHA
        log_fn(f'  Option B omnibus: F={optB_F:.4f}, '
               f'p={optB_p:.4f}, significant={optB_sig}')
    except Exception as e:
        log_fn(f'[WARNING] Lobe Option B rm_anova failed: {e}. '
               f'Flag for manual review.')

    # ----------------------------------------------------------------
    # Option C: Cluster permutation on central-minus-other
    # DIRECTION: central minus other for ALL three contrasts.
    # Do NOT swap this direction for any contrast.
    # Positive clusters = central greater than other lobe.
    # Negative clusters = other lobe greater than central.
    #
    # permutation_cluster_1samp_test computes cluster p-values
    # from its own internal permutation distribution. No additional
    # manual (b+1)/(N+1) correction is applied here — that
    # correction is only needed for sign-flip tests where p-values
    # are computed manually (as in Tests 1-3 above in the
    # hemisphere function). These are different mechanisms.
    # ----------------------------------------------------------------
    log_fn('--- Lobe Option C: cluster permutation '
           'central-minus-other ---')
    other_lobes  = [l for l in lobe_names if l != 'central']
    central_mat  = lobe_matrices['central']
    optC_results = {}
    for lobe in other_lobes:
        # DIRECTION: central minus other. Do not swap.
        diff_mat = central_mat - lobe_matrices[lobe]
        log_fn(f'  central minus {lobe}:')
        (cT, cClusters, cPV,
         cH0) = permutation_cluster_1samp_test(
            diff_mat,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            seed=42,
            verbose=False,
            out_type='mask'
        )
        contrast_clusters = []
        for k, (cl, pv) in enumerate(zip(cClusters, cPV)):
            if pv < ALPHA:
                mask  = cl[0]
                times = centres_arr[mask]
                epoch = ('pre'  if np.all(times < 0)  else
                         'post' if np.all(times >= 0) else
                         'mixed')
                direction = ('central_higher'
                             if float(np.mean(cT[mask])) > 0
                             else 'other_higher')
                contrast_clusters.append({
                    'cluster_index':   k,
                    'cluster_p_value': float(pv),
                    'time_start_ms':   float(times[0]),
                    'time_end_ms':     float(times[-1]),
                    'epoch':           epoch,
                    'direction':       direction,
                })
                log_fn(f'    Cluster {k}: p={pv:.6f}, '
                       f'{times[0]:.0f} to {times[-1]:.0f} ms, '
                       f'{epoch}, {direction}')
        if not contrast_clusters:
            log_fn(f'    No significant clusters for '
                   f'central minus {lobe}.')
        optC_results[f'central_vs_{lobe}'] = contrast_clusters

    lobe_results = {
        'lobe_matrices':       lobe_matrices,
        'lobe_term_means':     lobe_term_means,
        'lobe_term_slopes':    lobe_term_slopes,
        'optionA_F':           optA_F,
        'optionA_p':           optA_p,
        'optionA_significant': optA_sig,
        'optionA_posthoc':     posthoc_results,
        'optionB_F':           optB_F,
        'optionB_p':           optB_p,
        'optionB_significant': optB_sig,
        'optionC_clusters':    optC_results,
    }
    return lobe_results
