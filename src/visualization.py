"""
visualization.py — All figures for the WAY-EEG-GAL entropy pipeline.

Public functions
----------------
plot_qc_erp_sanity_check        QC Figure A — grand-average ERP at C3/Cz/C4
plot_qc_emg_onset_participant   QC Figure B — EMG onset detection per participant
plot_fig1_pre_post              Figure 1   — scalar pre/post SampEn bar chart
plot_fig2_topomap               Figure 2   — topographic t-statistic map
plot_fig3_grand_average         Figure 3   — grand-average entropy trajectory
plot_fig4_per_participant        Figure 4   — per-participant trajectories
plot_fig5_conditions            Figure 5   — condition generalizability
generate_all_figures            Wrapper that calls all figure functions

Conventions
-----------
- EEG amplitudes displayed in µV: multiply raw 10-µV units by EEG_UV_SCALE (0.1)
- Publication figures: 300 dpi. QC figures: 150 dpi.
- matplotlib only — no seaborn.
- All file paths from config. No hardcoded strings.
- Every saved file confirmed with a print statement.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne

from config import (
    FIGURES_DIR, CHAN_FILE,
    EEG_UV_SCALE, EEG_CHANNEL_NAMES,
    WEIGHT_MAP, N_PARTICIPANTS, N_CHANNELS,
    ALPHA,
)

# Motor-channel indices (0-based) from EEG_CHANNEL_NAMES
_C3_IDX  = EEG_CHANNEL_NAMES.index('C3')    # = 12
_CZ_IDX  = EEG_CHANNEL_NAMES.index('Cz')   # = 13
_C4_IDX  = EEG_CHANNEL_NAMES.index('C4')   # = 14


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, filename, dpi):
    """Save figure to FIGURES_DIR and print confirmation."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _load_montage():
    """
    Load the 32-channel electrode montage from CHAN_FILE.

    File format: one row per channel, space-separated:
        index  X  Y  Z  label
    No header line. Coordinates are in centimetres.
    CRITICAL: multiply all XYZ by 0.01 before passing to MNE.
    MNE uses metres internally; failure to convert places electrodes
    100× too far from head centre, producing silently wrong topomaps.

    Returns
    -------
    montage : mne.channels.DigMontage
        32-channel montage in the 'head' coordinate frame, metres.
    ch_names_file : list of str
        Channel names in the order they appear in CHAN_FILE (should match
        EEG_CHANNEL_NAMES).
    """
    with open(CHAN_FILE, 'r') as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    ch_names_file = []
    xyz_cm        = []

    for line in lines:
        parts = line.split()
        # Format: index X Y Z label
        xyz_cm.append([float(parts[1]), float(parts[2]), float(parts[3])])
        ch_names_file.append(parts[4])

    xyz_cm = np.array(xyz_cm)          # shape (32, 3)
    xyz_m  = xyz_cm * 0.01             # centimetres → metres (MNE uses metres)

    ch_pos   = {ch_names_file[i]: xyz_m[i] for i in range(len(ch_names_file))}
    montage  = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    return montage, ch_names_file


# ---------------------------------------------------------------------------
# QC Figure A — ERP sanity check
# ---------------------------------------------------------------------------

def plot_qc_erp_sanity_check(erp_data):
    """
    Grand-average ERP at C3, Cz, C4 time-locked to EMG onset.

    Must be generated and inspected BEFORE running statistics. If the expected
    Bereitschaftspotential / MRCP pattern (slow negative ramp from ~-1500 ms)
    is absent, the EEG preprocessing or onset alignment is incorrect.

    Parameters
    ----------
    erp_data : dict with keys:
        'grand_avg'   : numpy.ndarray, shape (N_samples, 32)
            Grand-average EEG (mean across all valid trials and all participants),
            in raw 10-µV units. Time axis is combined_t_ms.
        'combined_t_ms' : numpy.ndarray, shape (N_samples,)
            Time in ms relative to EMG onset.
        'sem'         : numpy.ndarray, shape (N_samples, 32)
            SEM across participants at each time × channel.

    Saves
    -----
    figures/qc_erp_sanity_check.png  (150 dpi)
    """
    t        = erp_data['combined_t_ms']
    avg      = erp_data['grand_avg']    # (N_samples, 32), raw 10-µV units
    sem      = erp_data['sem']          # (N_samples, 32)

    # Restrict plot window to -2000 to +1000 ms as specified.
    mask = (t >= -2000) & (t <= 1000)
    t_plot   = t[mask]
    avg_plot = avg[mask, :]
    sem_plot = sem[mask, :]

    fig, ax = plt.subplots(figsize=(10, 5))

    for ch_idx, color, label in [
        (_C3_IDX,  'steelblue', 'C3'),
        (_CZ_IDX,  'black',     'Cz'),
        (_C4_IDX,  'firebrick', 'C4'),
    ]:
        # Convert raw 10-µV units → µV for display.
        y    = avg_plot[:, ch_idx] * EEG_UV_SCALE
        y_se = sem_plot[:, ch_idx] * EEG_UV_SCALE
        ax.plot(t_plot, y, color=color, linewidth=1.8, label=label)
        ax.fill_between(t_plot, y - y_se, y + y_se,
                        color=color, alpha=0.15)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.2,
               label='EMG onset (t=0)')

    # Informational text box reminding the user what to look for.
    textstr = (
        "Expected: slow negative ramp (Bereitschaftspotential)\n"
        "from ~\u22121500 ms, peaking near t=0.\n"
        "If absent: check filtering and onset alignment."
    )
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='grey', alpha=0.8))

    ax.set_xlabel('Time relative to EMG onset (ms)', fontsize=11)
    ax.set_ylabel('EEG amplitude (\u00b5V)', fontsize=11)
    ax.set_title('ERP at Motor Channels \u2014 Sanity Check', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, 'qc_erp_sanity_check.png', dpi=150)


# ---------------------------------------------------------------------------
# QC Figure B — EMG onset detection per participant
# ---------------------------------------------------------------------------

def plot_qc_emg_onset_participant(participant_id, trial_records,
                                   emg_data_list, emg_t_list,
                                   smoothed_list, channel_names):
    """
    Show 12 randomly selected valid trials for one participant, each subplot
    displaying the smoothed RMS EMG envelope, onset threshold, detected onset,
    LEDon, and tHandStart.

    Parameters
    ----------
    participant_id : int
        Participant number (1–12).
    trial_records : list of dict
        Valid trial records for this participant (from onset_detection_results).
        Each dict must have: trial_in_series, weight, emg_onset_s,
        emg_onset_ms_post_led, tHandStart_s, threshold (float).
    emg_data_list : list of numpy.ndarray
        Smoothed RMS EMG envelope for each valid trial, shape (N_emg, 5).
    emg_t_list : list of numpy.ndarray
        EMG time vector for each valid trial, shape (N_emg,).
    smoothed_list : list of numpy.ndarray
        Same as emg_data_list (alias accepted for compatibility).
    channel_names : list of str
        5 EMG channel names from ws.names.emg.

    Saves
    -----
    figures/qc_emg_onset_P{participant_id}.png  (150 dpi)
    """
    n_available = len(trial_records)
    n_show      = min(12, n_available)

    if n_show == 0:
        print(f'  P{participant_id}: no valid trials for QC EMG plot — skipping.')
        return

    # Reproducible random trial selection (seed=42, as specified).
    rng          = np.random.default_rng(seed=42)
    indices      = rng.choice(n_available, size=n_show, replace=False)
    indices      = np.sort(indices)   # sort for interpretable ordering

    # 3-row × 4-col grid for 12 subplots.
    n_cols = 4
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(16, 4 * n_rows),
                             sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing

    # One colour per EMG channel (5 channels).
    emg_colors = ['tab:blue', 'tab:orange', 'tab:green',
                  'tab:red', 'tab:purple']

    for plot_i, trial_i in enumerate(indices):
        ax      = axes[plot_i]
        rec     = trial_records[trial_i]
        emg_env = emg_data_list[trial_i]    # (N_emg, 5)
        emg_t   = emg_t_list[trial_i]       # (N_emg,)

        for ch_idx, (ch_name, color) in enumerate(
                zip(channel_names, emg_colors)):
            ax.plot(emg_t, emg_env[:, ch_idx],
                    color=color, linewidth=0.7,
                    label=ch_name if plot_i == 0 else '_nolegend_')

        # Onset threshold — horizontal black dashed line.
        threshold = rec.get('threshold', np.nan)
        if not np.isnan(threshold):
            ax.axhline(threshold, color='black', linestyle='--',
                       linewidth=1.0, label='Threshold' if plot_i == 0
                       else '_nolegend_')

        # Detected EMG onset — vertical red solid line.
        onset_s = rec.get('emg_onset_s', np.nan)
        if not np.isnan(onset_s):
            ax.axvline(onset_s, color='red', linestyle='-',
                       linewidth=1.4, label='EMG onset' if plot_i == 0
                       else '_nolegend_')

        # LEDon — vertical orange dashed line (always at 2.0 s).
        ax.axvline(2.0, color='darkorange', linestyle='--',
                   linewidth=1.0, label='LEDon' if plot_i == 0
                   else '_nolegend_')

        # tHandStart — vertical grey dashed line (sanity check only).
        t_hs = rec.get('tHandStart_s', np.nan)
        if not np.isnan(t_hs):
            ax.axvline(t_hs, color='grey', linestyle='--',
                       linewidth=1.0, label='tHandStart' if plot_i == 0
                       else '_nolegend_')

        # Title with trial metadata.
        w_code   = int(rec.get('weight', 0))
        w_g      = WEIGHT_MAP.get(w_code, '?')
        latency  = rec.get('emg_onset_ms_post_led', np.nan)
        ax.set_title(
            f"Trial {rec.get('trial_in_series', '?')} | "
            f"{w_g}g | Onset={latency:.0f}ms post-LED"
            if not np.isnan(latency) else
            f"Trial {rec.get('trial_in_series', '?')} | {w_g}g | No onset",
            fontsize=7
        )
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('EMG (V)', fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots.
    for ax in axes[n_show:]:
        ax.set_visible(False)

    # Single shared legend from first subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'P{participant_id} — EMG Onset Detection QC '
                 f'({n_show} randomly selected valid trials)',
                 fontsize=11, y=1.01)
    fig.tight_layout()

    _save(fig, f'qc_emg_onset_P{participant_id}.png', dpi=150)


# ---------------------------------------------------------------------------
# Figure 1 — Scalar pre vs. post SampEn
# ---------------------------------------------------------------------------

def plot_fig1_pre_post(stats_output, trial_df):
    """
    Bar chart comparing mean pre-onset and post-onset SampEn, averaged across
    all 32 channels and all 12 participants.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics (stats.py).
    trial_df : pandas.DataFrame
        Trial-level entropy data with columns pre_sampen_mean, post_sampen_mean,
        participant.

    Saves
    -----
    figures/fig1_pre_post_comparison.png  (300 dpi)
    """
    # Per-participant means (unit of error bars = SEM across participants).
    part_pre  = []
    part_post = []

    for pid in range(1, N_PARTICIPANTS + 1):
        p_df = trial_df[trial_df['participant'] == pid]
        if len(p_df) == 0:
            continue
        part_pre.append(float(np.nanmean(p_df['pre_sampen_mean'].values)))
        part_post.append(float(np.nanmean(p_df['post_sampen_mean'].values)))

    part_pre  = np.array(part_pre)
    part_post = np.array(part_post)

    grand_pre   = np.nanmean(part_pre)
    grand_post  = np.nanmean(part_post)
    sem_pre     = np.nanstd(part_pre,  ddof=1) / np.sqrt(len(part_pre))
    sem_post    = np.nanstd(part_post, ddof=1) / np.sqrt(len(part_post))

    mean_diff   = grand_pre - grand_post
    sem_diff    = np.nanstd(part_pre - part_post, ddof=1) / \
                  np.sqrt(len(part_pre))
    direction   = 'higher' if mean_diff > 0 else 'lower'

    # Pull group-level statistic for annotation from channel_stats (channel-avg).
    ch_stats  = stats_output.get('channel_stats', [])
    sig_chs   = [r for r in ch_stats if r.get('group_significant')]
    # Use a representative channel (Cz) for annotation if available.
    cz_stat   = next((r for r in ch_stats
                      if r['channel_name'] == 'Cz'), None)
    if cz_stat:
        test_used = cz_stat.get('group_test_used', 'n/a')
        stat_val  = cz_stat.get('group_statistic', np.nan)
        p_fdr_val = cz_stat.get('group_p_fdr', np.nan)
    else:
        test_used, stat_val, p_fdr_val = 'n/a', np.nan, np.nan

    fig, ax = plt.subplots(figsize=(6, 8))

    x_pos  = [0, 1]
    colors = ['#4878CF', '#D65F5F']   # blue for pre, red for post
    bars = ax.bar(x_pos,
                  [grand_pre, grand_post],
                  yerr=[sem_pre, sem_post],
                  color=colors, width=0.5,
                  capsize=6, error_kw={'linewidth': 1.5},
                  zorder=2)

    # Overlay individual participant dots connected by lines.
    jitter = np.linspace(-0.08, 0.08, len(part_pre))
    for i, (pp, po) in enumerate(zip(part_pre, part_post)):
        j = jitter[i]
        ax.plot([0 + j, 1 + j], [pp, po],
                color='grey', linewidth=0.6, alpha=0.5, zorder=3)
        ax.scatter([0 + j], [pp], color='grey', s=18, zorder=4, alpha=0.7)
        ax.scatter([1 + j], [po], color='grey', s=18, zorder=4, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Pre-onset\n(-2000 to 0 ms)',
                         'Post-onset\n(0 to +2000 ms)'], fontsize=11)
    ax.set_ylabel('Sample Entropy (SampEn)', fontsize=11)
    ax.set_title('Pre vs. Post EMG Onset SampEn\n'
                 '(grand mean \u00b1 SEM across participants, all 32 channels;\n'
                 'annotation cites Cz channel group-level test as representative)',
                 fontsize=11)

    # Annotation text box.
    p_fdr_str = f'{p_fdr_val:.3f}' if not np.isnan(p_fdr_val) else 'n/a'
    stat_str  = f'{stat_val:.2f}' if not np.isnan(stat_val) else 'n/a'

    # Compute direction counts dynamically from ch_stats.
    n_lower  = sum(1 for r in ch_stats
                   if r.get('group_direction') == 'lower_pre')
    n_higher = sum(1 for r in ch_stats
                   if r.get('group_direction') == 'higher_pre')

    # Determine grand mean direction from actual computed values.
    # grand_pre and grand_post are already computed above as np.nanmean
    # of part_pre and part_post respectively.
    if grand_post > grand_pre:
        direction_str = 'post-onset SampEn slightly higher than pre-onset'
    else:
        direction_str = 'pre-onset SampEn slightly higher than post-onset'

    ann_text = (
        f'Direction: {direction_str}\n'
        f'Magnitude: {abs(mean_diff):.3f} \u00b1 {sem_diff:.3f} '
        f'(|pre\u2212post|, mean \u00b1 SEM)\n'
        f'Channel directions: {n_higher} higher_pre, '
        f'{n_lower} lower_pre (of 32 channels)\n'
        f'Group test (Cz): {test_used}, stat={stat_str}, '
        f'p(FDR)={p_fdr_str}\n'
        f'Significant channels: {len(sig_chs)}/32 (group-level FDR)'
    )
    fig.text(0.5, 0.02, ann_text, fontsize=8,
             va='bottom', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='grey', alpha=0.85),
             transform=fig.transFigure)

    ax.grid(axis='y', alpha=0.3, zorder=1)
    fig.tight_layout(rect=[0, 0.25, 1, 1])
    _save(fig, 'fig1_pre_post_comparison.png', dpi=300)


# ---------------------------------------------------------------------------
# Figure 2 — Topographic map
# ---------------------------------------------------------------------------

def plot_fig2_topomap(stats_output):
    """
    Topographic map of the group-level t-statistic (or z-statistic for
    Wilcoxon) at each of the 32 EEG channels.

    Channels with group-level p_fdr < ALPHA are marked with white dots.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics.

    Saves
    -----
    figures/fig2_topomap.png  (300 dpi)
    """
    ch_stats = stats_output['channel_stats']

    # sig_mask is derived from group_significant — significance criterion
    # is independent of which quantity is plotted on the map.
    sig_mask = np.array([r['group_significant'] for r in ch_stats], dtype=bool)

    # Compute signed effect size per channel.
    # Sign convention: positive = higher pre-onset entropy (group_direction ==
    # 'higher_pre'), negative = lower pre-onset entropy ('lower_pre').
    # This is on a consistent scale across all channels regardless of whether
    # Cohen's d (t-test channels) or rank-biserial r (Wilcoxon channels) was used.
    effect_vals = np.array([
        r['group_effect_size'] * (1.0 if r.get('group_direction') ==
        'higher_pre' else -1.0)
        for r in ch_stats
    ], dtype=np.float64)

    # Channels with insufficient data have NaN effect size.
    # Replace with 0.0 (baseline level) before plotting, identical to
    # how NaN statistics are currently handled in this function.
    effect_plot = np.where(np.isnan(effect_vals), 0.0, effect_vals)

    # Load montage.
    montage, ch_names_file = _load_montage()

    # Build minimal MNE Info object for plot_topomap.
    info = mne.create_info(ch_names=ch_names_file,
                           sfreq=500.0,
                           ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Determine colour limits symmetrically around zero.
    vmax = np.nanmax(np.abs(effect_plot))
    vmax = vmax if vmax > 0 else 1.0

    im, _ = mne.viz.plot_topomap(
        effect_plot,
        info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        vlim=(-vmax, vmax),
        contours=6,
        sensors=True,
        sphere='auto',
    )

    # Mark FDR-significant channels with white dots.
    # We need to re-project the 3-D electrode positions onto the 2-D topomap.
    # MNE's plot_topomap uses a nose-up convention; we recover 2-D positions
    # by calling _get_pos_outlines which is internal — instead we use the
    # montage positions directly via mne.channels.layout.find_layout.
    # A simpler approach: iterate channel positions and use the same projection
    # that MNE uses (azimuthal equidistant projection).
    try:
        from mne.channels.layout import _find_topomap_coords
        pos2d = _find_topomap_coords(info, picks='eeg', sphere='auto')
        for ch_i in range(len(ch_names_file)):
            if sig_mask[ch_i]:
                ax.plot(pos2d[ch_i, 0], pos2d[ch_i, 1],
                        'o', color='white', markersize=6,
                        markeredgecolor='black', markeredgewidth=0.8,
                        zorder=5)
    except Exception as e:
        # If MNE internal API changes, skip white dot overlay gracefully.
        print(f'  [WARNING] Could not overlay significant-channel dots: {e}')

    plt.colorbar(im, ax=ax,
                 label='Signed effect size\n(Cohen\'s d or rank-biserial r)',
                 fraction=0.046, pad=0.04)
    ax.set_title(
        f'Group-Level Effect Size (pre vs. post SampEn)\n'
        f'Positive = higher pre-onset | Negative = lower pre-onset\n'
        f'White dots = FDR-significant (p<{ALPHA:.2f})\n'
        f'Note: Cohen\'s d used for t-test channels, '
        f'rank-biserial r for Wilcoxon channels',
        fontsize=10
    )

    fig.tight_layout()
    _save(fig, 'fig2_topomap.png', dpi=300)


# ---------------------------------------------------------------------------
# Figure 3 — Grand-average entropy trajectory
# ---------------------------------------------------------------------------

def plot_fig3_grand_average(stats_output, sw_df, centres_ms, digit_contact_ms,
                             led_mean_ms=np.nan,
                             led_sd_ms=np.nan,
                             liftoff_contact_ms=np.nan):
    """
    Grand-average baseline-normalised SampEn trajectory across -2000 to +1998 ms,
    with cluster significance bars, onset markers, and slope annotation.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics (contains cluster_results, T_obs,
        slope_result, first_sig_pre_window_ms, recovery_ms).
    sw_df : pandas.DataFrame
        Sliding-window entropy data. Columns: participant, window_index,
        window_centre_ms, sampen_norm_ch0 ... sampen_norm_ch31.
    centres_ms : list of float
        Window centre times in ms (from epoching.py).
    digit_contact_ms : float or NaN
        Mean tFirstDigitTouch − EMG onset, in milliseconds.
        NaN if unavailable.

    Saves
    -----
    figures/fig3_grand_average_timeseries.png  (300 dpi)
    """
    centres_ms_arr = np.array(centres_ms)
    N_windows      = len(centres_ms)
    norm_cols      = [f'sampen_norm_ch{ch}' for ch in range(N_CHANNELS)]

    # ------------------------------------------------------------------
    # Grand average: mean across participants at each window.
    # participant_mean_norm already computed in stats — reuse from stats_output.
    # ------------------------------------------------------------------
    part_norm = stats_output['participant_mean_norm']   # (12, N_windows, 32)
    # Channel-average then participant-average.
    part_norm_ch_avg = np.nanmean(part_norm, axis=2)   # (12, N_windows)
    grand_avg = np.nanmean(part_norm_ch_avg, axis=0)   # (N_windows,)
    sem_vals  = np.nanstd(part_norm_ch_avg, axis=0, ddof=1) / \
                np.sqrt(np.sum(~np.isnan(part_norm_ch_avg), axis=0))

    slope_result = stats_output['slope_result']
    cluster_results = stats_output['cluster_results']
    first_sig = stats_output['first_sig_pre_window_ms']
    rec_ms    = stats_output['recovery_ms']

    fig, ax = plt.subplots(figsize=(13, 7))

    # Background shading: light grey pre-onset, light blue post-onset.
    x_min, x_max = centres_ms_arr[0], centres_ms_arr[-1]
    ax.axvspan(x_min, 0.0, color='lightgrey', alpha=0.35, zorder=0,
               label='Pre-onset region')
    ax.axvspan(0.0, x_max, color='lightblue', alpha=0.25, zorder=0,
               label='Post-onset region')

    # SEM shading and grand-average line.
    ax.fill_between(centres_ms_arr,
                    grand_avg - sem_vals,
                    grand_avg + sem_vals,
                    color='steelblue', alpha=0.25, zorder=2)
    ax.plot(centres_ms_arr, grand_avg,
            color='steelblue', linewidth=2.0,
            label='Grand average (\u00b11 SEM)', zorder=3)

    # EMG onset vertical line.
    ax.axvline(0.0, color='black', linewidth=1.5, zorder=4, label='EMG onset')

    # Mean digit contact line.
    if not np.isnan(digit_contact_ms):
        ax.axvline(digit_contact_ms, color='darkgreen',
                   linestyle='--', linewidth=1.2, zorder=4,
                   label=f'Mean digit contact ({digit_contact_ms:.0f} ms)')

    # Cluster significance bars at bottom of plot.
    y_range   = np.nanmax(grand_avg + sem_vals) - np.nanmin(grand_avg - sem_vals)
    y_bottom  = np.nanmin(grand_avg - sem_vals) - 0.08 * y_range
    bar_height = 0.03 * y_range

    pre_color  = '#e6194b'   # red for pre-onset clusters
    post_color = '#3cb44b'   # green for post-onset clusters
    span_color = '#f58231'   # orange for spanning clusters

    for cr in cluster_results:
        c_color = {'pre': pre_color, 'post': post_color,
                   'spanning': span_color}.get(cr['epoch'], 'purple')
        ax.barh(y=y_bottom, width=cr['time_end_ms'] - cr['time_start_ms'],
                left=cr['time_start_ms'], height=bar_height,
                color=c_color, alpha=0.85, zorder=3)
        ax.text(
            (cr['time_start_ms'] + cr['time_end_ms']) / 2,
            y_bottom + bar_height * 1.1,
            f"p={cr['cluster_p_value']:.3f}",
            ha='center', va='bottom', fontsize=7, color=c_color
        )

    # First significant pre-onset window marker.
    if not np.isnan(first_sig):
        ax.axvline(first_sig, color=pre_color, linestyle='--',
                   linewidth=1.2, zorder=4)
        ax.text(first_sig, ax.get_ylim()[1] * 0.98,
                f'Pre-onset cluster start: {first_sig:.0f} ms',
                color=pre_color, fontsize=8,
                ha='left' if first_sig < -500 else 'right',
                va='top')

    # Recovery marker.
    if not np.isnan(rec_ms):
        ax.axvline(rec_ms, color=post_color, linestyle=':',
                   linewidth=1.2, zorder=4)
        ax.text(rec_ms + 20, np.nanmin(grand_avg),
                f'Recovery: {rec_ms:.0f} ms',
                color=post_color, fontsize=8, va='bottom')

    # Slope direction annotation — placed in the lower-left figure margin
    # (pre-onset side, x≈0.12) so it does not overlap the trajectory line,
    # and remains visually associated with the pre-onset (left) region.
    slope_dir = slope_result.get('slope_direction', 'n/a')
    p_slope   = slope_result.get('p_slope', np.nan)
    p_sl_str  = f'{p_slope:.3f}' if not np.isnan(p_slope) else 'n/a'
    fig.text(0.12, 0.01, f'Pre-onset: {slope_dir}\n(p={p_sl_str})',
             fontsize=9, color='navy', va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='navy', alpha=0.85),
             transform=fig.transFigure)

    # LED cue mean vertical dashed line with ±1 SD shaded band.
    # MANDATORY GUARD: only plot if led_mean_ms is a valid negative
    # number. The LED cue always precedes EMG onset and must therefore
    # be negative in EMG-onset-aligned time. If led_mean_ms >= 0 or
    # is NaN, log a warning and skip — do NOT plot a cue line in the
    # post-onset region.
    if not np.isnan(led_mean_ms):
        if led_mean_ms >= 0:
            print('[WARNING] led_mean_ms is not negative — LED cue '
                  'marker skipped to avoid misleading placement.')
        else:
            ax.axvline(led_mean_ms, color='darkorange',
                       linestyle='--', linewidth=1.2, zorder=4,
                       label=f'Mean LED cue ({led_mean_ms:.0f} ms)')
            if not np.isnan(led_sd_ms):
                ax.axvspan(led_mean_ms - led_sd_ms,
                           led_mean_ms + led_sd_ms,
                           color='darkorange', alpha=0.10, zorder=1,
                           label='\u00b11 SD reaction time'
                                 ' (pooled)')

    # tLiftOff mean vertical dotted line.
    # Color: teal — distinct from existing darkgreen digit contact line.
    # Only drawn if the mean falls within the visible x-axis range;
    # otherwise the legend entry would be misleading (line not visible).
    # centres_ms_arr is already computed earlier in this function.
    x_min = float(centres_ms_arr[0])
    x_max = float(centres_ms_arr[-1])
    if not np.isnan(liftoff_contact_ms) and \
            x_min <= liftoff_contact_ms <= x_max:
        ax.axvline(liftoff_contact_ms, color='teal',
                   linestyle=':', linewidth=1.2, zorder=4,
                   label=f'Mean liftoff ({liftoff_contact_ms:.0f} ms)')
    elif not np.isnan(liftoff_contact_ms):
        print(f'[INFO] Mean liftoff ({liftoff_contact_ms:.0f} ms) '
              f'falls outside visible range [{x_min:.0f}, {x_max:.0f}]'
              f' ms — liftoff marker not shown in Figure 3.')

    ax.set_xlabel('Time relative to EMG onset (ms)', fontsize=11)
    ax.set_ylabel('Baseline-normalised SampEn', fontsize=11)
    ax.set_title('Grand-Average Movement-Locked Entropy Trajectory\n'
                 '(all participants, all 32 channels, ±1 SEM)', fontsize=11)
    ax.set_xlim(centres_ms_arr[0], centres_ms_arr[-1])
    # Compute ncol at runtime to match the actual number of legend
    # entries. This avoids gaps from over-specifying columns or
    # overflow from under-specifying. Cap at 4 columns maximum
    # to keep the legend compact.
    _handles, _labels = ax.get_legend_handles_labels()
    _n_entries = len(_handles)
    _ncol = min(4, max(1, _n_entries))
    ax.legend(handles=_handles, labels=_labels,
              fontsize=9,
              bbox_to_anchor=(0.5, -0.18),
              loc='upper center',
              borderaxespad=0,
              frameon=True,
              ncol=_ncol)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0.26, 1, 1])
    _save(fig, 'fig3_grand_average_timeseries.png', dpi=300)


# ---------------------------------------------------------------------------
# Figure 4 — Per-participant trajectories
# ---------------------------------------------------------------------------

def plot_fig4_per_participant(stats_output, centres_ms, trial_df, sw_df,
                              per_participant_led_ms=None):
    """
    3×4 grid of per-participant channel-averaged, baseline-normalised SampEn
    trajectories from -2000 to +1998 ms.

    Parameters
    ----------
    stats_output : dict
        Output of run_all_statistics (participant_mean_norm, participant_summary).
    centres_ms : list of float
        Window centre times in ms.
    trial_df : pandas.DataFrame
        Trial-level entropy data (for participant trial counts).
    sw_df : pandas.DataFrame
        Sliding-window entropy data (unused directly — data from stats_output).

    Saves
    -----
    figures/fig4_per_participant_timeseries.png  (300 dpi)
    """
    centres_ms_arr = np.array(centres_ms)
    part_norm      = stats_output['participant_mean_norm']  # (12, N_windows, 32)

    # Channel-average per participant → (12, N_windows).
    part_ch_avg = np.nanmean(part_norm, axis=2)

    # Compute SEM across trials for shading.
    # We need per-trial sliding-window data; approximate SEM from the
    # inter-window variability stored in sw_df.
    # Build per-participant (N_windows,) SEM arrays.
    norm_cols = [f'sampen_norm_ch{ch}' for ch in range(N_CHANNELS)]

    part_sems = np.full(part_ch_avg.shape, np.nan)
    for p_idx in range(N_PARTICIPANTS):
        pid   = p_idx + 1
        p_sw  = sw_df[sw_df['participant'] == pid]
        if len(p_sw) == 0:
            continue
        for w_idx in range(len(centres_ms)):
            w_rows = p_sw[p_sw['window_index'] == w_idx]
            if len(w_rows) < 2:
                continue
            vals = np.nanmean(w_rows[norm_cols].values.astype(np.float64),
                              axis=1)   # mean across channels per trial
            part_sems[p_idx, w_idx] = (np.nanstd(vals, ddof=1) /
                                        np.sqrt(np.sum(~np.isnan(vals))))

    # Compute shared Y-axis limits across all 12 participants.
    all_y = (part_ch_avg + part_sems).flatten()
    all_y = np.append(all_y, (part_ch_avg - part_sems).flatten())
    y_min = np.nanmin(all_y)
    y_max = np.nanmax(all_y)
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_lim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for p_idx in range(N_PARTICIPANTS):
        ax  = axes_flat[p_idx]
        pid = p_idx + 1

        y    = part_ch_avg[p_idx, :]
        se   = part_sems[p_idx, :]

        n_valid = len(trial_df[trial_df['participant'] == pid])

        # Pre/post background shading.
        ax.axvspan(centres_ms_arr[0], 0.0, color='lightgrey',
                   alpha=0.3, zorder=0)
        ax.axvspan(0.0, centres_ms_arr[-1], color='lightblue',
                   alpha=0.2, zorder=0)

        # SEM shading.
        valid_w = ~np.isnan(y) & ~np.isnan(se)
        if np.any(valid_w):
            ax.fill_between(centres_ms_arr[valid_w],
                            (y - se)[valid_w],
                            (y + se)[valid_w],
                            color='steelblue', alpha=0.25, zorder=2)
        ax.plot(centres_ms_arr, y, color='steelblue',
                linewidth=1.4, zorder=3)

        ax.axvline(0.0, color='black', linewidth=1.0, zorder=4)

        # Per-participant LED cue line — only if value is negative
        # (cue must precede onset in EMG-onset-aligned time).
        if per_participant_led_ms is not None:
            p_led = per_participant_led_ms.get(pid, np.nan)
            if not np.isnan(p_led) and p_led < 0:
                ax.axvline(p_led, color='darkorange', linestyle='--',
                           linewidth=0.9, zorder=4,
                           label='LED cue' if p_idx == 0
                           else '_nolegend_')

        ax.set_ylim(y_lim)
        ax.set_title(f'P{pid} (N={n_valid} trials)', fontsize=9)
        ax.grid(True, alpha=0.2)

    # Shared axis labels on outer subplots only.
    for ax in axes[-1, :]:
        ax.set_xlabel('Time re. EMG onset (ms)', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('Norm. SampEn', fontsize=8)

    fig.suptitle('Per-Participant Baseline-Normalised SampEn Trajectories\n'
                 '(channel average ± 1 SEM across trials)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig4_per_participant_timeseries.png', dpi=300)


# ---------------------------------------------------------------------------
# Figure 5 — Condition generalizability
# ---------------------------------------------------------------------------

def plot_fig5_conditions(stats_output):
    """
    Grouped bar chart of channel-averaged Cohen's d (or rank-biserial r)
    per weight condition (165 g, 330 g, 660 g).

    Parameters
    ----------
    stats_output : dict
        Must contain 'condition_result' key (from _condition_generalizability).

    Saves
    -----
    figures/fig5_condition_generalizability.png  (300 dpi)
    """
    cond_result = stats_output['condition_result']
    part_es     = cond_result['participant_condition_es']  # (12, 3)
    es_by_cond  = cond_result['effect_sizes_by_condition']  # {g: float}
    test_used   = cond_result['test_used']
    stat_val    = cond_result['statistic']
    p_val       = cond_result['p_value']

    weight_labels = ['165g', '330g', '660g']
    weight_gs     = [165, 330, 660]
    x_pos         = np.arange(3)
    bar_width     = 0.45
    colors        = ['#4878CF', '#6ACC65', '#D65F5F']

    # Grand means and SEMs per condition.
    means = [np.nanmean(part_es[:, c]) for c in range(3)]
    sems  = [np.nanstd(part_es[:, c], ddof=1) /
             np.sqrt(np.sum(~np.isnan(part_es[:, c])))
             for c in range(3)]

    fig, ax = plt.subplots(figsize=(7, 7.5))

    bars = ax.bar(x_pos, means, width=bar_width, color=colors,
                  yerr=sems, capsize=6,
                  error_kw={'linewidth': 1.5}, zorder=2)

    # Overlay individual participant dots.
    jitter = np.linspace(-0.12, 0.12, N_PARTICIPANTS)
    for c_idx in range(3):
        for p_idx in range(N_PARTICIPANTS):
            val = part_es[p_idx, c_idx]
            if np.isnan(val):
                continue
            ax.scatter(x_pos[c_idx] + jitter[p_idx], val,
                       color='grey', s=20, alpha=0.7, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(weight_labels, fontsize=12)
    ax.set_xlabel('Weight Condition', fontsize=11)
    ax.set_ylabel('Effect size (Cohen\'s d)', fontsize=11)
    ax.set_title('Pre vs. Post SampEn Effect Size by Weight Condition\n'
                 '(channel-averaged per participant, ±1 SEM)',
                 fontsize=11)

    # Annotate with omnibus test result — placed in the bottom figure margin
    # so it does not overlap the bars or participant dots.
    stat_str = f'{stat_val:.3f}' if not np.isnan(stat_val) else 'n/a'
    p_str    = f'{p_val:.3f}' if not np.isnan(p_val) else 'n/a'
    sig_str  = 'significant' if cond_result['significant'] else 'not significant'
    fig.text(0.5, 0.02,
             f'Test: {test_used}\nstat={stat_str}, p={p_str}\n({sig_str})',
             fontsize=9, va='bottom', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='grey', alpha=0.85),
             transform=fig.transFigure)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _save(fig, 'fig5_condition_generalizability.png', dpi=300)


# ---------------------------------------------------------------------------
# Master wrapper
# ---------------------------------------------------------------------------

def generate_all_figures(stats_output, trial_df, sw_df, centres_ms,
                          erp_data, digit_contact_ms,
                          led_mean_ms=np.nan,
                          led_sd_ms=np.nan,
                          per_participant_led_ms=None,
                          liftoff_contact_ms=np.nan,
                          per_participant_liftoff_ms=None,
                          spatial_results=None):
    """
    Call all figure-generation functions in the order specified by the pipeline.

    QC Figure A (ERP sanity check) is generated first so it can be inspected
    before interpreting any results.

    Parameters
    ----------
    stats_output : dict
        Output of stats.run_all_statistics.
    trial_df : pandas.DataFrame
        Trial-level entropy data (results/trial_level_entropy.csv).
    sw_df : pandas.DataFrame
        Sliding-window entropy data (results/sliding_window_entropy.csv).
    centres_ms : list of float
        Window centre times in ms (from epoching.build_sliding_windows).
    erp_data : dict
        Grand-average ERP data for QC Figure A. Keys: 'grand_avg',
        'combined_t_ms', 'sem'. See plot_qc_erp_sanity_check.
    digit_contact_ms : float or NaN
        Mean tFirstDigitTouch − EMG onset in ms for Figure 3 annotation.

    Notes
    -----
    QC Figure B (EMG onset per participant) is generated in Phase 1 of
    main.py immediately after onset detection, NOT here. It is listed here
    for documentation completeness only.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('\n--- Generating figures ---')

    # QC Figure A — ERP sanity check (must be first).
    print('  QC Figure A: ERP sanity check...')
    plot_qc_erp_sanity_check(erp_data)
    print('  Inspect figures/qc_erp_sanity_check.png before interpreting results.')

    # Figure 1 — Scalar pre/post bar chart.
    print('  Figure 1: pre vs. post SampEn...')
    plot_fig1_pre_post(stats_output, trial_df)

    # Figure 2 — Topomap.
    print('  Figure 2: topographic map...')
    plot_fig2_topomap(stats_output)

    # Figure 3 — Grand-average entropy trajectory.
    print('  Figure 3: grand-average entropy trajectory...')
    plot_fig3_grand_average(stats_output, sw_df, centres_ms, digit_contact_ms,
                            led_mean_ms=led_mean_ms,
                            led_sd_ms=led_sd_ms,
                            liftoff_contact_ms=liftoff_contact_ms)

    # Figure 4 — Per-participant trajectories.
    print('  Figure 4: per-participant trajectories...')
    plot_fig4_per_participant(stats_output, centres_ms, trial_df, sw_df,
                              per_participant_led_ms=per_participant_led_ms)

    # Figure 5 — Condition generalizability.
    print('  Figure 5: condition generalizability...')
    plot_fig5_conditions(stats_output)

    if spatial_results is not None:
        roi_r  = spatial_results.get('roi',        None)
        hemi_r = spatial_results.get('hemisphere', None)
        if roi_r is not None:
            print('  Figure ROI-A: ROI grand average...')
            plot_fig_roi_grand_average(
                roi_r, centres_ms,
                digit_contact_ms=digit_contact_ms,
                led_mean_ms=led_mean_ms,
                led_sd_ms=led_sd_ms,
                liftoff_contact_ms=liftoff_contact_ms)
            print('  Figure ROI-B: ROI per-participant...')
            plot_fig_roi_per_participant(
                roi_r, centres_ms,
                per_participant_led_ms=per_participant_led_ms)
            print('  Figure ROI-C: ROI conditions...')
            plot_fig_roi_conditions(roi_r)
        if hemi_r is not None:
            print('  Figure Hemi-A: hemisphere timeseries...')
            plot_fig_hemisphere_timeseries(
                hemi_r, centres_ms,
                digit_contact_ms=digit_contact_ms,
                led_mean_ms=led_mean_ms,
                led_sd_ms=led_sd_ms,
                liftoff_contact_ms=liftoff_contact_ms)
        lobe_r = spatial_results.get('lobe', None)
        if lobe_r is not None:
            print('  Figure Lobe-A: lobe terminal means...')
            plot_fig_lobe_terminal_means(lobe_r)
            print('  Figure Lobe-B: lobe terminal slopes...')
            plot_fig_lobe_terminal_slopes(lobe_r)
            print('  Figure Lobe-C: lobe timeseries...')
            plot_fig_lobe_timeseries(
                lobe_r, centres_ms,
                digit_contact_ms=digit_contact_ms,
                led_mean_ms=led_mean_ms,
                led_sd_ms=led_sd_ms,
                liftoff_contact_ms=liftoff_contact_ms)

    print('--- All figures complete ---\n')


# ---------------------------------------------------------------------------
# ROI grand average figure (analog of Figure 3)
# ---------------------------------------------------------------------------

def plot_fig_roi_grand_average(
        roi_results, centres_ms,
        digit_contact_ms=np.nan,
        led_mean_ms=np.nan,
        led_sd_ms=np.nan,
        liftoff_contact_ms=np.nan):
    """
    Grand average ROI entropy trajectory figure.
    Analog of Figure 3 using sensorimotor ROI data only
    (FC1, FC2, C3, Cz, C4, CP1, CP2 — 7 channels).
    Saves to: fig_roi_grand_average_timeseries.png
    Uses _save helper — do not use plt.savefig directly.
    """
    roi_matrix  = roi_results['roi_matrix']   # (12, N_windows)
    grand_avg   = np.nanmean(roi_matrix, axis=0)
    sem         = np.nanstd(roi_matrix, axis=0, ddof=1) / np.sqrt(12)
    centres_arr = np.array(centres_ms)
    x_min       = float(centres_arr[0])
    x_max       = float(centres_arr[-1])

    fig, ax = plt.subplots(figsize=(13, 7))

    # Background shading.
    ax.axvspan(x_min, 0.0, color='lightgrey', alpha=0.35, zorder=0,
               label='Pre-onset region')
    ax.axvspan(0.0, x_max, color='lightblue', alpha=0.25, zorder=0,
               label='Post-onset region')

    # SEM band and grand average line.
    ax.fill_between(centres_arr, grand_avg - sem, grand_avg + sem,
                    color='blue', alpha=0.2, zorder=2)
    ax.plot(centres_arr, grand_avg, color='blue', linewidth=1.5,
            label='Grand average (\u00b11 SEM)', zorder=3)

    # EMG onset line.
    ax.axvline(0.0, color='black', linewidth=1.5, zorder=4,
               label='EMG onset')

    # Cluster significance bars — dynamic position below data.
    # Matches Figure 3 (plot_fig3_grand_average) style exactly.
    y_range    = (np.nanmax(grand_avg + sem)
                  - np.nanmin(grand_avg - sem))
    y_bottom   = np.nanmin(grand_avg - sem) - 0.08 * y_range
    bar_height = 0.03 * y_range

    for cr in roi_results['cluster_results']:
        epoch = cr.get('epoch', 'pre')
        if epoch == 'pre':
            c_color = '#e6194b'
        elif epoch == 'post':
            c_color = '#3cb44b'
        else:
            c_color = '#f58231'
        ax.barh(
            y=y_bottom,
            width=cr['time_end_ms'] - cr['time_start_ms'],
            left=cr['time_start_ms'],
            height=bar_height,
            color=c_color, alpha=0.85, zorder=3)
        ax.text(
            (cr['time_start_ms'] + cr['time_end_ms']) / 2,
            y_bottom + bar_height * 1.1,
            f"p={cr['cluster_p_value']:.3f}",
            ha='center', va='bottom', fontsize=7,
            color=c_color)

    # Ensure bars are visible after dynamic positioning.
    if ax.get_ylim()[0] > y_bottom - bar_height:
        ax.set_ylim(bottom=y_bottom - bar_height * 0.5)

    # Pre-onset cluster start marker.
    pre_clusters = [cr for cr in roi_results['cluster_results']
                    if cr['epoch'] == 'pre']
    if pre_clusters:
        first_pre = min(cr['time_start_ms'] for cr in pre_clusters)
        ax.axvline(first_pre, color='crimson', linestyle='--',
                   linewidth=1.2, zorder=4,
                   label=f'Pre-onset cluster start: {first_pre:.0f} ms')

    # Digit contact marker.
    if not np.isnan(digit_contact_ms):
        ax.axvline(digit_contact_ms, color='darkgreen',
                   linestyle='--', linewidth=1.2, zorder=4,
                   label=f'Mean digit contact ({digit_contact_ms:.0f} ms)')

    # LED cue mean vertical dashed line with ±1 SD shaded band.
    # Guard: only plot if led_mean_ms is valid and negative.
    if not np.isnan(led_mean_ms):
        if led_mean_ms >= 0:
            print('[WARNING] led_mean_ms is not negative — LED cue '
                  'marker skipped (ROI figure).')
        else:
            ax.axvline(led_mean_ms, color='darkorange',
                       linestyle='--', linewidth=1.2, zorder=4,
                       label=f'Mean LED cue ({led_mean_ms:.0f} ms)')
            if not np.isnan(led_sd_ms):
                ax.axvspan(led_mean_ms - led_sd_ms,
                           led_mean_ms + led_sd_ms,
                           color='darkorange', alpha=0.10, zorder=1,
                           label='\u00b11 SD reaction time (pooled)')

    # Liftoff marker — only if within visible x range.
    if not np.isnan(liftoff_contact_ms) and \
            x_min <= liftoff_contact_ms <= x_max:
        ax.axvline(liftoff_contact_ms, color='teal',
                   linestyle=':', linewidth=1.2, zorder=4,
                   label=f'Mean liftoff ({liftoff_contact_ms:.0f} ms)')
    elif not np.isnan(liftoff_contact_ms):
        print(f'[INFO] Mean liftoff ({liftoff_contact_ms:.0f} ms) '
              f'falls outside visible range [{x_min:.0f}, {x_max:.0f}]'
              f' ms — liftoff marker not shown in ROI figure.')

    # Slope annotation boxes — outside data area, never inside axes.
    # Box 1: full pre-onset slope.
    full_dir = roi_results.get('full_slope_direction', 'n/a')
    full_p   = roi_results.get('full_slope_p', np.nan)
    full_p_s = f'{full_p:.3f}' if not np.isnan(full_p) else 'n/a'
    fig.text(0.12, 0.01,
             f'Pre-onset: {full_dir}\n(p={full_p_s})',
             fontsize=9, color='navy', va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='navy', alpha=0.85),
             transform=fig.transFigure)

    # Box 2: terminal slope.
    term_dir = roi_results.get('terminal_slope_direction', 'n/a')
    term_p   = roi_results.get('terminal_slope_p', np.nan)
    term_p_s = f'{term_p:.3f}' if not np.isnan(term_p) else 'n/a'
    fig.text(0.35, 0.01,
             f'Terminal: {term_dir}\n(p={term_p_s})',
             fontsize=9, color='darkred', va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='darkred', alpha=0.85),
             transform=fig.transFigure)

    ax.set_xlabel('Time relative to EMG onset (ms)', fontsize=11)
    ax.set_ylabel('Baseline-normalised SampEn', fontsize=11)
    ax.set_title(
        'Sensorimotor ROI Movement-Locked Entropy Trajectory\n'
        '(FC1, FC2, C3, Cz, C4, CP1, CP2 \u2014 7 channels, \u00b11 SEM)',
        fontsize=11)
    ax.set_xlim(centres_arr[0], centres_arr[-1])
    ax.grid(True, alpha=0.25)

    _handles, _labels = ax.get_legend_handles_labels()
    _n_entries = len(_handles)
    _ncol = min(4, max(1, _n_entries))
    ax.legend(handles=_handles, labels=_labels,
              fontsize=9,
              bbox_to_anchor=(0.5, -0.18),
              loc='upper center',
              borderaxespad=0,
              frameon=True,
              ncol=_ncol)

    fig.tight_layout(rect=[0, 0.26, 1, 1])
    _save(fig, 'fig_roi_grand_average_timeseries.png', dpi=300)


# ---------------------------------------------------------------------------
# ROI per-participant figure (analog of Figure 4)
# ---------------------------------------------------------------------------

def plot_fig_roi_per_participant(
        roi_results, centres_ms,
        per_participant_led_ms=None):
    """
    Per-participant ROI entropy trajectory figure.
    Analog of Figure 4 using sensorimotor ROI data.
    Saves to: fig_roi_per_participant_timeseries.png
    Uses _save helper — do not use plt.savefig directly.

    NOTE ON SEM: roi_matrix contains trial-averaged means per
    participant (shape 12 x N_windows). Per-trial ROI SEM
    is not available without sw_df (skipped pending Stage 7).
    This figure shows per-participant ROI means without a
    SEM band. This limitation is documented here and must
    be noted in any figure caption.
    """
    roi_matrix  = roi_results['roi_matrix']   # (12, N_windows)
    centres_arr = np.array(centres_ms)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for p_idx in range(N_PARTICIPANTS):
        ax  = axes_flat[p_idx]
        pid = p_idx + 1
        y   = roi_matrix[p_idx, :]

        ax.axvspan(centres_arr[0], 0.0, color='lightgrey',
                   alpha=0.3, zorder=0)
        ax.axvspan(0.0, centres_arr[-1], color='lightblue',
                   alpha=0.2, zorder=0)

        # Single line per participant — no SEM band (see docstring).
        ax.plot(centres_arr, y, color='steelblue',
                linewidth=1.4, zorder=3)
        ax.axvline(0.0, color='black', linewidth=1.0, zorder=4)

        # Per-participant LED cue line (orange dashed).
        # Guard: only plot if value is negative.
        if per_participant_led_ms is not None:
            p_led = per_participant_led_ms.get(pid, np.nan)
            if not np.isnan(p_led) and p_led < 0:
                ax.axvline(p_led, color='darkorange',
                           linestyle='--', linewidth=0.9, zorder=4,
                           label='LED cue' if p_idx == 0
                           else '_nolegend_')

        ax.set_title(f'P{pid} (ROI)', fontsize=9)
        ax.grid(True, alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel('Time re. EMG onset (ms)', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('Norm. SampEn', fontsize=8)

    fig.suptitle(
        'Per-Participant Sensorimotor ROI SampEn Trajectories\n'
        '(trial-averaged, no SEM \u2014 see note)',
        fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig_roi_per_participant_timeseries.png', dpi=300)


# ---------------------------------------------------------------------------
# ROI condition generalizability figure (analog of Figure 5)
# ---------------------------------------------------------------------------

def plot_fig_roi_conditions(roi_results):
    """
    ROI condition generalizability bar chart.
    Analog of Figure 5 using ROI condition results.
    Saves to: fig_roi_condition_generalizability.png
    Uses _save helper — do not use plt.savefig directly.

    NOTE: ROI condition test is currently skipped (NaN values)
    pending Stage 7 implementation. Figure is generated
    regardless to show the placeholder structure.
    """
    condition_F    = roi_results['condition_F']
    condition_p    = roi_results['condition_p']
    condition_sig  = roi_results['condition_significant']
    condition_means = roi_results.get('condition_means', {})
    condition_participant_means = roi_results.get(
        'condition_participant_means', {})

    # Weight code to label mapping — must match Figure 5
    # exactly. Sorted weight codes [1,2,4] map to labels
    # ['165g','330g','660g'] in order.
    WEIGHT_LABELS = ['165g', '330g', '660g']
    WEIGHT_CODES  = [1, 2, 4]

    # Show real bar chart only when condition_F is valid AND
    # condition_means is populated. Show placeholder otherwise.
    show_real = (
        bool(condition_means) and
        not np.isnan(condition_F)
    )

    fig, ax = plt.subplots(figsize=(7, 7.5))

    if not show_real:
        # Placeholder — pending data or Stage 7 not complete.
        ax.text(0.5, 0.5,
            'ROI condition generalizability\n'
            'not yet computed\n'
            '(pending Stage 7 implementation)',
            ha='center', va='center',
            transform=ax.transAxes, fontsize=11)
    else:
        # Real bar chart — matches Figure 5 style exactly.
        x_pos    = [0, 1, 2]
        bar_width = 0.6
        colors   = ['#4878CF', '#6ACC65', '#D65F5F']

        # Group means and SEMs for bars.
        means = [condition_means.get(w, np.nan)
                 for w in WEIGHT_CODES]

        # Per-participant values for dot overlay.
        # condition_participant_means[w] is a list of
        # N_PARTICIPANTS values in participant order.
        part_vals = [
            condition_participant_means.get(w, [])
            for w in WEIGHT_CODES
        ]
        # Compute SEM from per-participant values.
        sems = []
        for pv in part_vals:
            arr = np.array([v for v in pv
                            if not np.isnan(v)])
            if len(arr) > 1:
                sems.append(
                    float(np.nanstd(arr, ddof=1)
                          / np.sqrt(len(arr))))
            else:
                sems.append(0.0)

        # Draw bars with error bars matching Figure 5.
        ax.bar(x_pos, means, width=bar_width,
               color=colors, yerr=sems, capsize=6,
               error_kw={'linewidth': 1.5}, zorder=2)

        # Overlay individual participant dots matching
        # Figure 5 exactly:
        # jitter = np.linspace(-0.12, 0.12, N_participants)
        # color='grey', s=20, alpha=0.7, zorder=3
        n_parts = max(
            (len(pv) for pv in part_vals if pv), default=0)
        if n_parts > 0:
            jitter = np.linspace(-0.12, 0.12, n_parts)
            for c_idx, pv in enumerate(part_vals):
                for p_idx, val in enumerate(pv):
                    if np.isnan(val):
                        continue
                    ax.scatter(
                        x_pos[c_idx] + jitter[p_idx],
                        val,
                        color='grey', s=20,
                        alpha=0.7, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(WEIGHT_LABELS, fontsize=12)
        ax.set_xlabel('Weight Condition', fontsize=11)
        ax.set_ylabel(
            'Mean ROI pre-onset SampEn\n'
            '(baseline-normalised)', fontsize=11)
        ax.axhline(0, color='black',
                   linewidth=0.8, linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3, zorder=1)

    ax.set_title('Sensorimotor ROI Condition Generalizability',
                 fontsize=11)

    # Annotation outside data area — always shown.
    # Must use fig.text with transform=fig.transFigure.
    # Must not overlap data area.
    if np.isnan(condition_F) or not condition_means:
        annot_text = ('Test: rm_anova\n'
                      'Result: pending Stage 7')
    else:
        sig_str = ('significant'
                   if condition_sig else 'not significant')
        annot_text = (f'Test: rm_anova\n'
                      f'F={condition_F:.3f}, '
                      f'p={condition_p:.3f}\n'
                      f'({sig_str})')
    fig.text(0.5, 0.02, annot_text, fontsize=9,
        ha='center', va='bottom',
        transform=fig.transFigure,
        bbox=dict(boxstyle='round,pad=0.4',
        facecolor='lightyellow',
        edgecolor='grey', alpha=0.85))

    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _save(fig, 'fig_roi_condition_generalizability.png', dpi=300)


# ---------------------------------------------------------------------------
# Hemisphere timeseries figure
# ---------------------------------------------------------------------------

def plot_fig_hemisphere_timeseries(
        hemi_results, centres_ms,
        digit_contact_ms=np.nan,
        led_mean_ms=np.nan,
        led_sd_ms=np.nan,
        liftoff_contact_ms=np.nan):
    """
    Left and right hemisphere grand average trajectories
    overlaid on the same plot with cluster significance bars.
    Saves to: fig_hemisphere_timeseries.png
    Uses _save helper — do not use plt.savefig directly.
    """
    left_matrix  = hemi_results['left_matrix']   # (12, N_windows)
    right_matrix = hemi_results['right_matrix']  # (12, N_windows)
    left_avg     = np.nanmean(left_matrix,  axis=0)
    right_avg    = np.nanmean(right_matrix, axis=0)
    left_sem     = np.nanstd(left_matrix,  axis=0, ddof=1) / np.sqrt(12)
    right_sem    = np.nanstd(right_matrix, axis=0, ddof=1) / np.sqrt(12)
    centres_arr  = np.array(centres_ms)
    x_min        = float(centres_arr[0])
    x_max        = float(centres_arr[-1])

    fig, ax = plt.subplots(figsize=(13, 7))

    # Background shading.
    ax.axvspan(x_min, 0.0, color='lightgrey', alpha=0.35, zorder=0,
               label='Pre-onset region')
    ax.axvspan(0.0, x_max, color='lightblue', alpha=0.25, zorder=0,
               label='Post-onset region')

    # Left hemisphere trajectory (blue).
    ax.fill_between(centres_arr, left_avg - left_sem,
                    left_avg + left_sem,
                    color='blue', alpha=0.2, zorder=2)
    ax.plot(centres_arr, left_avg, color='blue', linewidth=1.5,
            label='Left hemisphere (\u00b11 SEM)', zorder=3)

    # Right hemisphere trajectory (crimson).
    ax.fill_between(centres_arr, right_avg - right_sem,
                    right_avg + right_sem,
                    color='crimson', alpha=0.2, zorder=2)
    ax.plot(centres_arr, right_avg, color='crimson', linewidth=1.5,
            label='Right hemisphere (\u00b11 SEM)', zorder=3)

    # EMG onset line.
    ax.axvline(0.0, color='black', linewidth=1.5, zorder=4,
               label='EMG onset')

    # Cluster significance bars — dynamic position below data.
    # Matches Figure 3 style. Combined range from both hemispheres.
    _all_avg   = np.concatenate([[left_avg], [right_avg]])
    _all_sem   = np.concatenate([[left_sem], [right_sem]])
    y_range    = (np.nanmax(_all_avg + _all_sem)
                  - np.nanmin(_all_avg - _all_sem))
    y_bottom   = np.nanmin(_all_avg - _all_sem) - 0.08 * y_range
    bar_height = 0.03 * y_range

    # Track which direction labels have been added to the legend
    # to prevent duplicate legend entries.
    _hemi_legend_added = set()

    for cr in hemi_results['cluster_results']:
        direction = cr.get('direction', 'left_greater')
        if direction == 'left_greater':
            c_color   = 'purple'
            dir_label = 'L>R cluster'
        else:
            c_color   = 'darkorchid'
            dir_label = 'R>L cluster'
        # Only add legend label for first occurrence of each direction.
        leg_label = (dir_label
                     if dir_label not in _hemi_legend_added
                     else '_nolegend_')
        _hemi_legend_added.add(dir_label)
        ax.barh(
            y=y_bottom,
            width=cr['time_end_ms'] - cr['time_start_ms'],
            left=cr['time_start_ms'],
            height=bar_height,
            color=c_color, alpha=0.85, zorder=3,
            label=leg_label)
        ax.text(
            (cr['time_start_ms'] + cr['time_end_ms']) / 2,
            y_bottom + bar_height * 1.1,
            f"p={cr['cluster_p_value']:.3f}",
            ha='center', va='bottom', fontsize=7,
            color=c_color)

    # Ensure bars are visible after dynamic positioning.
    if ax.get_ylim()[0] > y_bottom - bar_height:
        ax.set_ylim(bottom=y_bottom - bar_height * 0.5)

    # Digit contact marker.
    if not np.isnan(digit_contact_ms):
        ax.axvline(digit_contact_ms, color='darkgreen',
                   linestyle='--', linewidth=1.2, zorder=4,
                   label=f'Mean digit contact ({digit_contact_ms:.0f} ms)')

    # LED cue marker (orange) with ±1 SD band.
    # Guard: only plot if led_mean_ms is valid and negative.
    if not np.isnan(led_mean_ms):
        if led_mean_ms >= 0:
            print('[WARNING] led_mean_ms is not negative — LED cue '
                  'marker skipped (hemisphere figure).')
        else:
            ax.axvline(led_mean_ms, color='darkorange',
                       linestyle='--', linewidth=1.2, zorder=4,
                       label=f'Mean LED cue ({led_mean_ms:.0f} ms)')
            if not np.isnan(led_sd_ms):
                ax.axvspan(led_mean_ms - led_sd_ms,
                           led_mean_ms + led_sd_ms,
                           color='darkorange', alpha=0.10, zorder=1,
                           label='\u00b11 SD reaction time (pooled)')

    # Liftoff marker — only if within visible x range.
    if not np.isnan(liftoff_contact_ms) and \
            x_min <= liftoff_contact_ms <= x_max:
        ax.axvline(liftoff_contact_ms, color='teal',
                   linestyle=':', linewidth=1.2, zorder=4,
                   label=f'Mean liftoff ({liftoff_contact_ms:.0f} ms)')
    elif not np.isnan(liftoff_contact_ms):
        print(f'[INFO] Mean liftoff ({liftoff_contact_ms:.0f} ms) '
              f'falls outside visible range [{x_min:.0f}, {x_max:.0f}]'
              f' ms — liftoff marker not shown in hemisphere figure.')

    # Terminal results annotation outside data area.
    t_mean_p   = hemi_results.get('terminal_mean_p',  np.nan)
    t_slope_p  = hemi_results.get('terminal_slope_p', np.nan)
    t_mean_ps  = f'{t_mean_p:.3f}'  if not np.isnan(t_mean_p)  else 'n/a'
    t_slope_ps = f'{t_slope_p:.3f}' if not np.isnan(t_slope_p) else 'n/a'
    fig.text(0.12, 0.01,
             f'Terminal mean L-R: p={t_mean_ps}'
             f'\nTerminal slope L-R: p={t_slope_ps}',
             fontsize=9, va='bottom', ha='left',
             transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='navy', alpha=0.85))

    ax.set_xlabel('Time relative to EMG onset (ms)', fontsize=11)
    ax.set_ylabel('Baseline-normalised SampEn', fontsize=11)
    ax.set_title(
        'Left vs Right Hemisphere Entropy Trajectory\n'
        '(Grand average \u00b11 SEM, cluster permutation on L\u2212R)',
        fontsize=11)
    ax.set_xlim(centres_arr[0], centres_arr[-1])
    ax.grid(True, alpha=0.25)

    _handles, _labels = ax.get_legend_handles_labels()
    _n_entries = len(_handles)
    _ncol = min(4, max(1, _n_entries))
    ax.legend(handles=_handles, labels=_labels,
              fontsize=9,
              bbox_to_anchor=(0.5, -0.18),
              loc='upper center',
              borderaxespad=0,
              frameon=True,
              ncol=_ncol)

    fig.tight_layout(rect=[0, 0.26, 1, 1])
    _save(fig, 'fig_hemisphere_timeseries.png', dpi=300)


# ---------------------------------------------------------------------------
# Lobe terminal means bar chart
# ---------------------------------------------------------------------------

def plot_fig_lobe_terminal_means(lobe_results):
    """
    Bar chart of terminal window mean SampEn per lobe region.
    Shows mean ± 1 SEM across 12 participants for each of
    four regions: frontal, central, parietal, occipital.
    Lobe Option A rm_anova result annotated outside data area.
    If omnibus significant, post hoc contrasts annotated with
    asterisk markers above relevant bars (no bbox).
    Analog of Figure 5 layout for lobe comparison.
    Saves to: fig_lobe_terminal_means.png
    Uses _save helper — do not use plt.savefig directly.
    """
    lobe_term_means = lobe_results['lobe_term_means']
    lobe_names      = ['frontal', 'central', 'parietal', 'occipital']
    lobe_colors     = {
        'frontal':   '#E69F00',
        'central':   '#0072B2',
        'parietal':  '#009E73',
        'occipital': '#CC79A7',
    }
    x_pos     = np.arange(len(lobe_names))
    bar_width = 0.6

    means  = [np.nanmean(lobe_term_means[l]) for l in lobe_names]
    sems   = [np.nanstd(lobe_term_means[l], ddof=1) /
              np.sqrt(N_PARTICIPANTS) for l in lobe_names]
    colors = [lobe_colors[l] for l in lobe_names]

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.bar(x_pos, means, width=bar_width, color=colors,
           yerr=sems, capsize=5, alpha=0.8,
           error_kw={'linewidth': 1.5, 'ecolor': 'black'},
           zorder=2)

    # Individual participant dots overlaid on each bar.
    jitter = np.linspace(-0.15, 0.15, N_PARTICIPANTS)
    for i, lobe in enumerate(lobe_names):
        for p_idx in range(N_PARTICIPANTS):
            val = lobe_term_means[lobe][p_idx]
            if np.isnan(val):
                continue
            ax.scatter(x_pos[i] + jitter[p_idx], val,
                       color='grey', s=20, alpha=0.5, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Frontal', 'Central', 'Parietal', 'Occipital'],
                       fontsize=11)
    ax.set_xlabel('Brain region', fontsize=11)
    ax.set_ylabel('Mean terminal SampEn (baseline-normalised)',
                  fontsize=11)
    ax.set_title(
        'Regional Terminal Pre-Onset SampEn\n'
        '(Mean of -386 to -14 ms windows, \u00b11 SEM)',
        fontsize=11)

    # Omnibus result annotation outside data area using fig.text.
    optA_F   = lobe_results['optionA_F']
    optA_p   = lobe_results['optionA_p']
    optA_sig = lobe_results['optionA_significant']
    if pd.isna(optA_F):
        annot_text = 'rm_anova: result unavailable'
    else:
        sig_str    = 'significant' if optA_sig else 'not significant'
        annot_text = (f'rm_anova: F={float(optA_F):.3f}, '
                      f'p={float(optA_p):.3f}\n({sig_str})')
    fig.text(0.5, 0.01, annot_text, fontsize=9, ha='center',
             va='bottom', transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='lightyellow',
                       edgecolor='grey', alpha=0.85))

    # Post hoc significance markers — ax.text only, no bbox.
    # Asterisks are standard bar chart markers, placed inside axes.
    posthoc = lobe_results.get('optionA_posthoc', {})
    if posthoc and optA_sig:
        for key, ph in posthoc.items():
            if ph.get('significant', False):
                partner = key.replace('central_vs_', '')
                if partner in lobe_names:
                    p_bar = lobe_names.index(partner)
                    bar_top = means[p_bar] + sems[p_bar]
                    ax.text(x_pos[p_bar], bar_top + 0.002, '*',
                            ha='center', va='bottom', fontsize=14,
                            color='black')

    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _save(fig, 'fig_lobe_terminal_means.png', dpi=300)


# ---------------------------------------------------------------------------
# Lobe terminal slopes bar chart
# ---------------------------------------------------------------------------

def plot_fig_lobe_terminal_slopes(lobe_results):
    """
    Bar chart of terminal window OLS slope per lobe region.
    Exploratory — analog of plot_fig_lobe_terminal_means
    but using OLS slope values instead of means.
    No post hoc markers — Option B has no post hoc tests.
    Saves to: fig_lobe_terminal_slopes.png
    Uses _save helper — do not use plt.savefig directly.
    """
    lobe_term_slopes = lobe_results['lobe_term_slopes']
    lobe_names       = ['frontal', 'central', 'parietal', 'occipital']
    lobe_colors      = {
        'frontal':   '#E69F00',
        'central':   '#0072B2',
        'parietal':  '#009E73',
        'occipital': '#CC79A7',
    }
    x_pos     = np.arange(len(lobe_names))
    bar_width = 0.6

    means  = [np.nanmean(lobe_term_slopes[l]) for l in lobe_names]
    sems   = [np.nanstd(lobe_term_slopes[l], ddof=1) /
              np.sqrt(N_PARTICIPANTS) for l in lobe_names]
    colors = [lobe_colors[l] for l in lobe_names]

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.bar(x_pos, means, width=bar_width, color=colors,
           yerr=sems, capsize=5, alpha=0.8,
           error_kw={'linewidth': 1.5, 'ecolor': 'black'},
           zorder=2)

    # Individual participant dots overlaid on each bar.
    jitter = np.linspace(-0.15, 0.15, N_PARTICIPANTS)
    for i, lobe in enumerate(lobe_names):
        for p_idx in range(N_PARTICIPANTS):
            val = lobe_term_slopes[lobe][p_idx]
            if np.isnan(val):
                continue
            ax.scatter(x_pos[i] + jitter[p_idx], val,
                       color='grey', s=20, alpha=0.5, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Frontal', 'Central', 'Parietal', 'Occipital'],
                       fontsize=11)
    ax.set_xlabel('Brain region', fontsize=11)
    ax.set_ylabel('Terminal slope SampEn/ms (OLS)', fontsize=11)
    ax.set_title(
        'Regional Terminal Pre-Onset SampEn Slope\n'
        '(Exploratory \u2014 OLS slope across -386 to -14 ms, '
        '\u00b11 SEM)',
        fontsize=11)

    # Omnibus result annotation outside data area using fig.text.
    optB_F   = lobe_results['optionB_F']
    optB_p   = lobe_results['optionB_p']
    optB_sig = lobe_results['optionB_significant']
    if pd.isna(optB_F):
        annot_text = 'rm_anova (exploratory): result unavailable'
    else:
        sig_str    = 'significant' if optB_sig else 'not significant'
        annot_text = (f'rm_anova (exploratory): '
                      f'F={float(optB_F):.3f}, '
                      f'p={float(optB_p):.3f}\n({sig_str})')
    fig.text(0.5, 0.01, annot_text, fontsize=9, ha='center',
             va='bottom', transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='lightyellow',
                       edgecolor='grey', alpha=0.85))

    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _save(fig, 'fig_lobe_terminal_slopes.png', dpi=300)


# ---------------------------------------------------------------------------
# Lobe overlaid timeseries with Option C cluster bars
# ---------------------------------------------------------------------------

def plot_fig_lobe_timeseries(
        lobe_results, centres_ms,
        digit_contact_ms=np.nan,
        led_mean_ms=np.nan,
        led_sd_ms=np.nan,
        liftoff_contact_ms=np.nan):
    """
    Overlaid full entropy trajectories for all four lobe
    regions across the full -2000ms to +1998ms window.
    Shows grand average ± 1 SEM per region.
    Lobe Option C cluster permutation results shown as
    staggered significance bars at the bottom of the axes.
    Cluster bars show PAIRWISE CONTRAST significance
    (when central differs from each other region) —
    NOT individual region significance vs baseline.
    Saves to: fig_lobe_timeseries.png
    Uses _save helper — do not use plt.savefig directly.
    """
    lobe_matrices = lobe_results['lobe_matrices']
    lobe_names    = ['frontal', 'central', 'parietal', 'occipital']
    lobe_colors   = {
        'frontal':   '#E69F00',
        'central':   '#0072B2',
        'parietal':  '#009E73',
        'occipital': '#CC79A7',
    }
    lobe_labels = {
        'frontal':   'Frontal (\u00b11 SEM)',
        'central':   'Central (\u00b11 SEM)',
        'parietal':  'Parietal (\u00b11 SEM)',
        'occipital': 'Occipital (\u00b11 SEM)',
    }
    centres_arr = np.array(centres_ms)
    x_min       = float(centres_arr[0])
    x_max       = float(centres_arr[-1])

    fig, ax = plt.subplots(figsize=(13, 7))

    # Background shading.
    ax.axvspan(x_min, 0.0, color='lightgrey', alpha=0.35, zorder=0,
               label='Pre-onset region')
    ax.axvspan(0.0, x_max, color='lightblue', alpha=0.25, zorder=0,
               label='Post-onset region')

    # Four overlaid lobe trajectories.
    for lobe in lobe_names:
        grand_avg = np.nanmean(lobe_matrices[lobe], axis=0)
        sem       = (np.nanstd(lobe_matrices[lobe], axis=0, ddof=1) /
                     np.sqrt(N_PARTICIPANTS))
        c         = lobe_colors[lobe]
        ax.fill_between(centres_arr, grand_avg - sem, grand_avg + sem,
                        color=c, alpha=0.15, zorder=2)
        ax.plot(centres_arr, grand_avg, color=c, linewidth=1.5,
                label=lobe_labels[lobe], zorder=3)

    # EMG onset line.
    ax.axvline(0.0, color='black', linewidth=1.5, zorder=4,
               label='EMG onset')

    # Option C cluster significance bars (staggered y positions).
    # Do NOT use orange — orange is reserved for LED cue marker.
    # Contrast colors match the non-central lobe color.
    optC = lobe_results['optionC_clusters']
    contrast_config = [
        ('central_vs_frontal',   -0.025, '#E69F00', 'F>C cluster'),
        ('central_vs_parietal',  -0.033, '#009E73', 'P>C cluster'),
        ('central_vs_occipital', -0.041, '#CC79A7', 'O>C cluster'),
    ]
    _legend_added = set()
    for contrast_key, y_bar, bar_color, bar_legend in contrast_config:
        cluster_list = optC.get(contrast_key, [])
        for cluster in cluster_list:
            _lbl = (bar_legend if contrast_key not in _legend_added
                    else '_nolegend_')
            ax.hlines(y=y_bar,
                      xmin=cluster['time_start_ms'],
                      xmax=cluster['time_end_ms'],
                      colors=bar_color, linewidth=4,
                      label=_lbl)
            _legend_added.add(contrast_key)
            midpoint = ((cluster['time_start_ms'] +
                         cluster['time_end_ms']) / 2)
            ax.text(midpoint, y_bar + 0.001,
                    f"p={cluster['cluster_p_value']:.3f}",
                    ha='center', va='bottom', fontsize=7,
                    color=bar_color)

    # Cluster bar visibility rule — adjusted for three staggered bars
    # reaching down to y=-0.041. Do NOT use standard -0.025/-0.030.
    current_ylim = ax.get_ylim()
    if current_ylim[0] > -0.041:
        ax.set_ylim(bottom=-0.048)

    # Digit contact marker.
    if not np.isnan(digit_contact_ms):
        ax.axvline(digit_contact_ms, color='darkgreen',
                   linestyle='--', linewidth=1.2, zorder=4,
                   label=f'Mean digit contact ({digit_contact_ms:.0f} ms)')

    # LED cue marker (orange) with ±1 SD band.
    # Guard: only plot if led_mean_ms is valid and negative.
    if not np.isnan(led_mean_ms):
        if led_mean_ms >= 0:
            print('[WARNING] led_mean_ms is not negative — LED cue '
                  'marker skipped (lobe timeseries figure).')
        else:
            ax.axvline(led_mean_ms, color='darkorange',
                       linestyle='--', linewidth=1.2, zorder=4,
                       label=f'Mean LED cue ({led_mean_ms:.0f} ms)')
            if not np.isnan(led_sd_ms):
                ax.axvspan(led_mean_ms - led_sd_ms,
                           led_mean_ms + led_sd_ms,
                           color='darkorange', alpha=0.10, zorder=1,
                           label='\u00b11 SD reaction time (pooled)')

    # Liftoff marker — only if within visible x range.
    if not np.isnan(liftoff_contact_ms) and \
            x_min <= liftoff_contact_ms <= x_max:
        ax.axvline(liftoff_contact_ms, color='teal',
                   linestyle=':', linewidth=1.2, zorder=4,
                   label=f'Mean liftoff ({liftoff_contact_ms:.0f} ms)')
    elif not np.isnan(liftoff_contact_ms):
        print(f'[INFO] Mean liftoff ({liftoff_contact_ms:.0f} ms) '
              f'falls outside visible range [{x_min:.0f}, {x_max:.0f}]'
              f' ms — liftoff marker not shown in lobe figure.')

    # Option C annotation outside data area using fig.text.
    # Format: 'Central vs {Lobe}: {n} significant cluster(s)' per line.
    _summary_lines = []
    for contrast_key, _, _, _ in contrast_config:
        n = len(optC.get(contrast_key, []))
        lobe_label = contrast_key.replace('central_vs_', '').capitalize()
        _summary_lines.append(
            f'Central vs {lobe_label}: {n} significant cluster(s)')
    summary_text = '\n'.join(_summary_lines)
    fig.text(0.12, 0.01, summary_text, fontsize=8,
             va='bottom', ha='left', transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='navy', alpha=0.85))

    ax.set_xlabel('Time relative to EMG onset (ms)', fontsize=11)
    ax.set_ylabel('Baseline-normalised SampEn', fontsize=11)
    ax.set_title(
        'Regional Entropy Trajectories\n'
        '(Frontal, Central, Parietal, Occipital \u2014 \u00b11 SEM;\n'
        'bars = Option C pairwise cluster significance)',
        fontsize=11)
    ax.set_xlim(centres_arr[0], centres_arr[-1])
    ax.grid(True, alpha=0.25)

    _handles, _labels = ax.get_legend_handles_labels()
    _n_entries = len(_handles)
    _ncol = min(4, max(1, _n_entries))
    ax.legend(handles=_handles, labels=_labels,
              fontsize=9,
              bbox_to_anchor=(0.5, -0.18),
              loc='upper center',
              borderaxespad=0,
              frameon=True,
              ncol=_ncol)

    fig.tight_layout(rect=[0, 0.26, 1, 1])
    _save(fig, 'fig_lobe_timeseries.png', dpi=300)
