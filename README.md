# EEG Time–Frequency (MNE Sample) — Research-level Mini Pipeline

- Two-condition TFR (Auditory Left vs Right) on MNE sample EEG
- Morlet time–frequency power with baseline log-ratio (ERSP-style)
- Condition difference (Right − Left)
- Cluster-based permutation test with significance contour

## Run
python src/tfr_mne_sample.py

## Output
![TFR Diff](results/tfr_diff_with_stats.png)
![ERP + TFR](results/erp_tfr_figure.png)
