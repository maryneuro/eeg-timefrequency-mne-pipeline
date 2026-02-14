import os
import numpy as np
import matplotlib.pyplot as plt
import mne

def main():
    os.makedirs("results", exist_ok=True)

    # Load dataset
    data_path = mne.datasets.sample.data_path(verbose=False)
    raw_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    event_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw-eve.fif")

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(1., 40., verbose=False)

    events = mne.read_events(event_fname)

    event_id = {"Auditory/Left": 1}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        preload=True,
        verbose=False,
    )

    epochs = epochs[:60].resample(300)

    freqs = np.linspace(4, 40, 60)
    n_cycles = freqs / 2.

    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
    )

    ch_name = power.ch_names[0]
    power_data = power.copy().pick(ch_name).data[0]  # shape: freqs x times

    times = power.times
    freqs = power.freqs

    # --- Custom matplotlib plot (publication style) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        power_data,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        interpolation="bicubic",   # smoother
        cmap="RdBu_r"
    )

    ax.set_title(f"Time–Frequency (Morlet) — {ch_name}", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (log ratio)", fontsize=11)

    plt.tight_layout()

    out_png = os.path.join("results", "time_frequency_eeg.png")
    fig.savefig(out_png, dpi=400)
    plt.close(fig)

    print("Saved high-quality figure to:", out_png)


if __name__ == "__main__":
    main()