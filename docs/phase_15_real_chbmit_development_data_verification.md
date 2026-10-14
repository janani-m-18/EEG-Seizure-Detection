# Phase 15: Real CHB-MIT Development-Data Verification

## Dataset-root configuration

`CHBMIT_DATASET_ROOT` must be the corpus directory that contains patient
folders, rather than a patient folder itself.  The verified local setting was:

```powershell
$env:CHBMIT_DATASET_ROOT = 'D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0'
```

No download was started or resumed.

## Local development subset verified

The local `chb01` folder contains exactly 13 EDF files:
`chb01_01.edf` through `chb01_13.edf`.  There is no local `chb01_14.edf`.
Each available EDF has 23 channels, a 256 Hz sampling frequency, 921,600
samples, and a 3,600-second duration.

The supplied `chb01-summary.txt` identifies one seizure in each of the two
available seizure recordings: `chb01_03.edf` (2996--3036 seconds) and
`chb01_04.edf` (1467--1494 seconds).  The remaining available recordings are
listed with zero seizures.  `chb01_03.edf` and `chb01_04.edf` also have
adjacent `.edf.seizures` files.

## Pipeline verification

`chb01_03.edf` was loaded through the existing MNE loader, processed with the
existing default preprocessing configuration (0.5--70 Hz band-pass, 50 Hz
notch, 500 microvolt peak-to-peak rejection, z-score normalization), then
segmented into non-overlapping four-second windows.  It produced 900 windows
of 1,024 samples at 256 Hz.

The current 500 microvolt quality-control threshold rejected 21 of the 23 EEG
channels in this recording, retaining only `P7-O1` and `T7-FT9`.  This is an
observed preprocessing result, not a claim about data quality or a clinical
judgement; the threshold and channel policy need study-specific review before
any training is considered.

## Annotation limitations

The project parser intentionally reads only standard `*-summary.txt` seizure
start/end entries.  MNE reported no embedded annotations in `chb01_03.edf`,
and the parser does not read the adjacent binary `.edf.seizures` files.  The
summary file lists additional CHB01 recordings that are not present in this
local development subset; discovery uses only EDF files actually on disk, so
no missing recording or label is fabricated.  Summary-based intervals are
recording-relative seconds and should be independently validated if a future
study requires a different annotation source or boundary convention.
