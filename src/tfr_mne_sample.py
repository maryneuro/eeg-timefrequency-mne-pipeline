import os
import numpy as np
import matplotlib.pyplot as plt
import mne

def main():
    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # Load MNE sample dataset
    # -----------------------------
    data_path = mne.datasets.sample.data_path(verbose=False)
    raw_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    event_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw-eve.fif")

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(1., 40., verbose=False)

    events = mne.read_events(event_fname)

    # Two auditory conditions exist in the sample events: 1 (Left), 2 (Right)
    event_id = {"Auditory/Left": 1, "Auditory/Right": 2}

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=-0.2, tmax=0.8,
        baseline=None,          # we'll baseline-correct at TFR stage
        preload=True, verbose=False,
        reject_by_annotation=True,
    )

    # Speed vs quality knobs (research-level but still reasonable)
    epochs = epochs.resample(250)
    epochs_L = epochs["Auditory/Left"][:80]
    epochs_R = epochs["Auditory/Right"][:80]

    # -----------------------------
    # Compute TFR (Morlet)
    # -----------------------------
    freqs = np.linspace(4, 40, 50)     # denser freq grid
    n_cycles = freqs / 2.0

    tfr_L = epochs_L.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=False, decim=1
    )
    tfr_R = epochs_R.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=False, decim=1
    )

    # Baseline correction (ERSP-like): logratio relative to baseline window
    baseline = (-0.2, 0.0)
    tfr_L.apply_baseline(baseline=baseline, mode="logratio")
    tfr_R.apply_baseline(baseline=baseline, mode="logratio")

    # -----------------------------
    # Choose channel (or average a few)
    # -----------------------------
    # Use a stable channel name if present; otherwise fall back to first
    preferred = "EEG 014"
    ch = preferred if preferred in tfr_L.ch_names else tfr_L.ch_names[0]

    # Extract single-channel trial-wise data: (n_epochs, n_freqs, n_times)
    X_L = tfr_L.copy().pick(ch).data[:, 0, :, :]
    X_R = tfr_R.copy().pick(ch).data[:, 0, :, :]

    # Condition difference per trial (Right - Left)
    # If trial counts differ, match the minimum
    n = min(X_L.shape[0], X_R.shape[0])
    X = X_R[:n] - X_L[:n]   # shape: (n_trials, n_freqs, n_times)

    times = tfr_L.times
    freqs = tfr_L.freqs

    # -----------------------------
    # Simple cluster permutation test (1-sample on difference)
    # -----------------------------
    # H0: mean difference = 0
    # Returns mask of significant clusters
    from mne.stats import permutation_cluster_1samp_test

    # reshape to (n_trials, n_features)
    X_2d = X.reshape(n, -1)

    T_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        X_2d,
        n_permutations=512,   # increase to 1024 if you want more stable p-values
        threshold=None,
        tail=0,
        out_type="mask",
        verbose=False,
        seed=42
    )

    # Build significance mask in (freq, time)
    sig_mask = np.zeros(X_2d.shape[1], dtype=bool)
    for cl, p in zip(clusters, cluster_pv):
        if p < 0.05:
            sig_mask |= cl
    sig_mask = sig_mask.reshape(len(freqs), len(times))

    # Mean difference map (freq, time)
    diff_mean = X.mean(axis=0)

    # Robust color scaling
    vmax = np.nanpercentile(np.abs(diff_mean), 98)
    vmin = -vmax

    # -----------------------------
    # Plot research-level figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(
        diff_mean,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        interpolation="bicubic"
    )

    # Overlay significance contour
    if sig_mask.any():
        ax.contour(
            times, freqs, sig_mask.astype(int),
            levels=[0.5], linewidths=1.5
        )

    ax.axvline(0, linewidth=1)  # stimulus onset
    ax.set_title(f"TFR Difference (Right − Left), logratio baseline — {ch}", fontsize=13)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power difference (logratio)")

    plt.tight_layout()
    out_png = os.path.join("results", "tfr_diff_with_stats.png")
    fig.savefig(out_png, dpi=400)
    plt.close(fig)

    # -----------------------------
    # Save a compact report
    # -----------------------------
    report_path = os.path.join("results", "run_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEG Time–Frequency (research-level mini pipeline)\n")
        f.write(f"Channel: {ch}\n")
        f.write(f"Epochs Left/Right used: {len(epochs_L)} / {len(epochs_R)} (matched n={n})\n")
        f.write(f"Baseline: {baseline}, mode=logratio\n")
        f.write(f"Freqs: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz (n={len(freqs)})\n")
        f.write("Stats: permutation_cluster_1samp_test on (Right-Left), alpha=0.05\n")
        f.write(f"Output figure: {out_png}\n")

    print("Done. Saved:", out_png)

if __name__ == "__main__":
    main()