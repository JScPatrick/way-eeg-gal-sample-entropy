"""
Path resolution for the figure and statistics scripts in this directory.

These scripts read the per-participant sliding-window entropy tables and the
EMG onset table that `src/main.py` writes. By default they look in `results/`
at the repository root, which is where the pipeline puts them. Each location
can be overridden from the environment:

    WEG_RESULTS    directory holding the per-participant entropy tables
                   and onset_detection_results.csv  (default: <root>/results)
    WEG_ONSET      explicit path to the onset table, if it lives elsewhere
    WEG_PERWINDOW  optional legacy per-window-tolerance dataset, used only
                   for the CF1 comparison trajectory in sfn_verify.py
                   (default: <root>/results/per_window; absence is fine)
    WEG_OUT        directory for figures and intermediate JSON
                   (default: <root>/output, created on import)

Example:

    WEG_RESULTS=/data/run2 python figures/verify_topo.py

----------------------------------------------------------------------------
Why resolution is strict
----------------------------------------------------------------------------
`main.py` writes two tables per participant into the same directory:

    sliding_window_entropy_P<n>.csv   29 windows x 32 channels per trial
    trial_level_entropy_P<n>.csv      one row per trial

They have different shapes, and loading the wrong one would corrupt every
downstream statistic without necessarily raising. Earlier versions of these
scripts selected the input with `glob.glob(...)[0]`, whose ordering is not
guaranteed, so this module instead checks an explicit list of accepted names
and raises if more than one of them exists. A clear error is preferable to a
silently wrong dataset.

The shortened `sw_P<n>.csv` name is still accepted, because some staged
copies of the intermediate data use it, but it is never preferred over the
canonical name and never resolved alongside it.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

#: directory src/main.py writes its per-participant tables into
RESULTS = os.environ.get(
    'WEG_RESULTS',
    os.environ.get('WEG_DERIVED',            # accepted for backward compat
                   os.path.join(ROOT, 'results')))

#: alias; several scripts import this module's DERIVED name
DERIVED = RESULTS

#: optional legacy comparison dataset (SAMPEN_TOLERANCE_MODE = 'per_window').
#: Not produced by the default configuration and not required by any figure.
PERWINDOW = os.environ.get('WEG_PERWINDOW',
                           os.path.join(RESULTS, 'per_window'))

#: figures and intermediate JSON
OUT = os.environ.get('WEG_OUT', os.path.join(ROOT, 'output'))
os.makedirs(OUT, exist_ok=True)


def _resolve(directory, names, what, required=True):
    """Return the one existing file among ``names`` in ``directory``.

    Raises on ambiguity -- always, whether or not the file is required --
    because picking arbitrarily from several candidates is how the wrong
    dataset gets loaded without anyone noticing. Returns ``None`` when
    nothing matches and ``required`` is False.
    """
    hits = [os.path.join(directory, n) for n in names
            if os.path.isfile(os.path.join(directory, n))]
    if len(hits) > 1:
        raise SystemExit(
            '\nAmbiguous input: found more than one candidate for '
            f'{what} in\n    {directory}\n\n'
            + ''.join(f'    {os.path.basename(h)}\n' for h in hits)
            + '\nRefusing to guess which to load, because these tables have\n'
              'different shapes and the wrong one would corrupt the results.\n'
              'Move or remove the extras so exactly one remains.\n')
    if hits:
        return hits[0]
    if not required:
        return None
    raise SystemExit(
        f'\nCannot find {what} in\n    {directory}\n\n'
        f'Looked for: {", ".join(names)}\n\n'
        'Run the pipeline first (`python src/main.py` -- see the README), or\n'
        'point the figure scripts at an existing directory:\n'
        '    WEG_RESULTS=/path/to/results python figures/<script>.py\n')


def sliding_window_file(participant, directory=None, required=True):
    """Resolve one participant's sliding-window entropy table.

    Canonical name first, then the shortened staging name. Both plain and
    gzipped are accepted; pandas decompresses by extension.
    """
    d = RESULTS if directory is None else directory
    p = participant
    return _resolve(d,
                    [f'sliding_window_entropy_P{p}.csv',
                     f'sliding_window_entropy_P{p}.csv.gz',
                     f'sw_P{p}.csv',
                     f'sw_P{p}.csv.gz'],
                    f'the sliding-window entropy table for P{p}',
                    required=required)


def onset_file(required=True):
    """Resolve the EMG onset table."""
    env = os.environ.get('WEG_ONSET')
    if env:
        if not os.path.isfile(env):
            raise SystemExit(
                f'\nWEG_ONSET is set but points at a missing file:\n    {env}\n')
        return env
    return _resolve(RESULTS,
                    ['onset_detection_results.csv',
                     'onset_detection_results.csv.gz'],
                    'the EMG onset table (onset_detection_results.csv)',
                    required=required)


def __getattr__(name):
    """Resolve ``ONSET`` on first access rather than at import time.

    Scripts do ``from _paths import ONSET``; deferring the lookup keeps the
    error message attached to the script that actually needs the file.
    """
    if name == 'ONSET':
        return onset_file()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def require(path, what):
    """Fail with a useful message rather than a bare FileNotFoundError."""
    if not os.path.exists(path):
        raise SystemExit(
            f'\nMissing {what}:\n    {path}\n\n'
            'See the README for the script that generates it.\n')
    return path
