# Raw data

The raw dataset is public and is **not** redistributed here.

## WAY-EEG-GAL

Luciw, M. D., Jarocka, E., & Edin, B. B. (2014). Multi-channel EEG recordings
during 3,936 grasp and lift trials with varying weight and friction.
*Scientific Data*, 1, 140047. https://doi.org/10.1038/sdata.2014.47

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The raw data is **not** redistributed in this repository — download it from the
original source. No processed or derived data is redistributed either: this
repository is code only, and `src/main.py` regenerates the intermediate
tables into `results/`.

The data records are hosted on figshare:
https://doi.org/10.6084/m9.figshare.988376

Download the `.mat` files and place them **flat** in `data/mat/`:

```
data/mat/P1_AllLifts.mat
data/mat/WS_P1_S1.mat
...
```

`src/verify_setup.py` checks that the layout and channel metadata are as the
pipeline expects. Run it before `main.py`.

## Channel coordinates

`chanlabels_32channel.xyz` is part of the WAY-EEG-GAL dataset and is **not
redistributed here**. It is **required** to complete `src/main.py`.

**Where to get it.** It is in **Supplementary File 2, `Utilities.zip`**, on the
dataset's figshare record (https://doi.org/10.6084/m9.figshare.988376). The
data descriptor describes it as the file "used to localize the electrode
positions in EEGLab".

**Where to put it.** Extract `Utilities.zip` and copy the file, keeping its
original name, to:

```
data/chanlabels_32channel.xyz
```

This is `CHAN_FILE` in `src/config.py`. `src/visualization.py` reads it in
`_load_montage()` to build the MNE montage for the Phase 5 topographic map;
Phase 5 aborts with `FileNotFoundError` if it is absent.
`src/verify_setup.py` reports `FOUND` or `MISSING` and warns — check that
before a long run.

Format: one row per channel, space-separated, no header, `index X Y Z label`,
coordinates in **centimetres** (`_load_montage()` multiplies by 0.01 for MNE,
which uses metres). 32 rows, labels matching `EEG_CHANNEL_NAMES`.

The three poster figures produced by `figures/` do **not** use this file —
`figures/topo_final.py` uses MNE's `standard_1020` template (see limitation 5
in the main README).
