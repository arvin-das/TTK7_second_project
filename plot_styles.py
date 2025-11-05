
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt


def compute_hht(imfs, fs=250, smooth_sigma=1):
    """
    Compute instantaneous amplitude and frequency for IMFs via Hilbert transform.
    Args:
        imfs: (K, R, T)
        fs: sampling frequency (Hz)
        smooth_sigma: Gaussian smoothing for inst. frequency
    Returns:
        inst_amp, inst_freq: both (K, R, T)
    """
    analytic = hilbert(imfs, axis=2)
    inst_amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=2)
    inst_freq = np.diff(phase, prepend=phase[:, :, :1], axis=2) * (fs / (2*np.pi))
    if smooth_sigma > 0:
        inst_freq = gaussian_filter1d(inst_freq, sigma=smooth_sigma, axis=2)
    return inst_amp, inst_freq




# ---------------------------------------------------------------
# IMF and frequency plots
# ---------------------------------------------------------------


def plot_imfs(X, imfs, roi_idx=0, freqs=None, subj_label="", method_name="VLMD", save_path=None):
    """
    Plot original signal and its IMFs for a single ROI.
    Args:
        X: (R, T)
        imfs: (K, R, T)
    """
    K, R, T = imfs.shape
    t = np.arange(T)
    fig, axs = plt.subplots(K + 1, 1, figsize=(12, 1.6 * (K + 1)), sharex=True)

    axs[0].plot(t, X[roi_idx], color='k', lw=1.2)
    axs[0].set_ylabel("Original", rotation=0, labelpad=25)
    axs[0].set_title(f"{method_name} – channel {roi_idx + 1} – {subj_label}")
    axs[0].grid(alpha=0.3)

    for k in range(K):
        axs[k + 1].plot(t, imfs[k, roi_idx, :], lw=0.9)
        label = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            label += f"\n{freqs[k]:.3f} Hz"
        axs[k + 1].set_ylabel(label, rotation=0, labelpad=30)
        axs[k + 1].grid(alpha=0.2)

    axs[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_imfs_with_spectrum(X, imfs, fs=1.25, roi_idx=0, freqs=None,
                            subj_label="", method_name="VLMD", fmax=0.3, save_path=None):
    """
    Plot the original signal and each IMF with its corresponding PSD side by side,
    preserving the vertical structure of the pure IMF plot.
    
    Args:
        X: (R, T) array of signals
        imfs: (K, T, R) array of IMFs
        fs: sampling frequency
        roi_idx: ROI index
    """
    K, R, T = imfs.shape
    t = np.arange(T) / fs

    # +1 row for original signal
    fig, axs = plt.subplots(K + 1, 2, figsize=(12, 1.6 * (K + 1)), sharex='col')

    # --- Row 0: Original signal ---
    axs[0, 0].plot(t, X[roi_idx], color='k', lw=1.2)
    axs[0, 0].set_ylabel("Original", rotation=0, labelpad=25)
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].set_title(f"{method_name} – channel {roi_idx + 1}")

    # Power spectrum of original signal
    f, Pxx = welch(X[roi_idx], fs=fs, nperseg=min(256, T))
    axs[0, 1].semilogy(f, Pxx, color='darkorange', lw=0.8)
    axs[0, 1].set_xlim(0, fmax)
    axs[0, 1].set_ylabel("PSD", rotation=0, labelpad=20)
    axs[0, 1].grid(alpha=0.3)

    # --- Each IMF row ---
    for k in range(K):
        # Time-domain IMF
        axs[k + 1, 0].plot(t, imfs[k, roi_idx, :], lw=0.9)
        label = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            label += f"\n{freqs[k]:.3f} Hz"
        axs[k + 1, 0].set_ylabel(label, rotation=0, labelpad=30)
        axs[k + 1, 0].grid(alpha=0.3)

        # Frequency-domain IMF
        f, Pxx = welch(imfs[k, roi_idx,: ], fs=fs, nperseg=min(256, T))
        axs[k + 1, 1].semilogy(f, Pxx, color='darkorange', lw=0.8)
        axs[k + 1, 1].set_xlim(0, fmax)
        axs[k + 1, 1].grid(alpha=0.3)

    # Axis labels
    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Frequency (Hz)")

    # Clean layout
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



# ---------------------------------------------------------------------
# HILBERT–HUANG TRANSFORMS (IMPROVED)
# ---------------------------------------------------------------------

def plot_imfs_with_hht(imfs, inst_amp, inst_freq, fs=1.25, roi_idx=0,
                       freqs=None, subj_label="", method_name="VLMD",
                       fmax=0.3, smooth_sigma=1, cmap="turbo", save_path=None):
    """
    Plot each IMF (time-domain) next to its Hilbert–Huang spectrum.
    """
    K, R, T = imfs.shape
    t = np.arange(T) / fs
    freq_bins = np.linspace(0, fmax, 200)

    fig, axs = plt.subplots(K, 2, figsize=(12, 1.8 * K), gridspec_kw={'width_ratios': [1.2, 2]}, sharex='col')
    axs = np.atleast_2d(axs)

    for k in range(K):
        # --- IMF signal ---
        axs[k, 0].plot(t, imfs[k, roi_idx,: ], lw=0.8, color="navy")
        label = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            label += f"\n{freqs[k]:.3f} Hz"
        axs[k, 0].set_ylabel(label, rotation=0, labelpad=30)
        axs[k, 0].grid(alpha=0.2)

        # --- HHT Spectrum ---
        f = inst_freq[k, roi_idx,: ]
        a = inst_amp[k, roi_idx, : ]
        inds = np.digitize(f, freq_bins) - 1
        H = np.zeros((len(freq_bins)-1, T))
        np.add.at(H, (inds.clip(0, len(freq_bins)-2), np.arange(T)), a)
        if smooth_sigma > 0:
            H = gaussian_filter1d(H, sigma=smooth_sigma, axis=1)

        pcm = axs[k, 1].pcolormesh(t, freq_bins[:-1], H, shading="auto", cmap=cmap)
        axs[k, 1].set_ylim(0, fmax)
        axs[k, 1].grid(False)

        if k == 0:
            axs[k, 0].set_title("IMF Signal", fontsize=10)
            axs[k, 1].set_title("Hilbert–Huang Spectrum", fontsize=10)

    # --- Axis labels ---
    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    for ax in axs[:, 1]:
        ax.set_ylabel("Frequency (Hz)")

    # --- Shared colorbar to the right ---
    fig.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.75])  # [left, bottom, width, height]
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Amplitude", rotation=270, labelpad=15)

    # --- Title ---
    fig.suptitle(f"{method_name} – channel {roi_idx +1} ", fontsize=13, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# def plot_combined_hht(inst_amp, inst_freq, fs, roi_idx=0,
#                       method_name="VLMD", subj_label="", fmax=0.3, smooth_sigma=1,
#                       save_path=None):
#     """
#     Combined Hilbert–Huang spectrum:
#     Sum amplitude contributions from all IMFs.
#     """
#     K, R, T  = inst_amp.shape
#     t = np.arange(T) / fs
#     freq_bins = np.linspace(0, fmax, 200)
#     H = np.zeros((len(freq_bins)-1, T))

#     for k in range(K):
#         f = inst_freq[k, roi_idx, :]
#         a = inst_amp[k, roi_idx, :]
#         inds = np.digitize(f, freq_bins) - 1
#         np.add.at(H, (inds.clip(0, len(freq_bins)-2), np.arange(T)), a)

#     if smooth_sigma > 0:
#         H = gaussian_filter1d(H, sigma=smooth_sigma, axis=1)

#     plt.figure(figsize=(10, 5))
#     plt.pcolormesh(t, freq_bins[:-1], H, shading='auto', cmap='turbo')
#     plt.xlabel("Time (s)")
#     plt.ylabel("Frequency (Hz)")
#     plt.title(f"Combined Hilbert–Huang Spectrum – channel {roi_idx + 1 }, {method_name}")
#     plt.colorbar(label="Amplitude (summed across IMFs)")
#     plt.ylim(0, fmax)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()

def plot_combined_hht(
    inst_amp, inst_freq, fs, roi_idx=0,
    method_name="MEMD", subj_label="",
    fmax=None, f_bin_hz=0.5,
    imf_keep=None,                 # e.g. (0, M-1) to drop residue
    min_f_hz=0.5,                  # ignore < 0.5 Hz
    amp_thresh_pct=None,           # e.g. 50 keeps top half of amplitudes
    smooth_sigma=(0.8, 1.5),       # (freq_bins, time_samples)
    dynamic_range_db=60,
    cmap="magma"
):
    """
    inst_amp, inst_freq: shape = (K, R, T)
    """
    K, R, T = inst_amp.shape
    if fmax is None:
        fmax = fs/2

    # choose IMFs (drop residue by default)
    if imf_keep is None:
        imf_keep = (0, K-1)   # keep 0..K-2; excludes k=K-1 (residue) below

    k0, k1 = imf_keep
    ks = range(k0, min(k1, K-1))   # stop before residue

    # frequency bin edges
    nfbins = int(np.ceil(fmax / f_bin_hz)) + 1
    f_edges = np.linspace(0, fmax, nfbins)
    H = np.zeros((nfbins-1, T), dtype=float)

    for k in ks:
        f = inst_freq[k, roi_idx].astype(float)
        a = inst_amp[k, roi_idx].astype(float)

        # optional IF denoising (helps with spurious spikes)
        f = medfilt(f, kernel_size=5)

        # masks: finite, within range, above minimum frequency
        mask = np.isfinite(f) & np.isfinite(a) & (f >= min_f_hz) & (f < fmax)
        if amp_thresh_pct is not None:
            thr = np.percentile(a[mask], amp_thresh_pct) if mask.any() else 0.0
            mask &= (a >= thr)

        if not mask.any():
            continue

        # use power (a^2) for weighting
        w = (a[mask] ** 2)

        # linear binning: distribute to two nearest bins
        f_idx = np.interp(f[mask], f_edges, np.arange(nfbins)) - 1.0
        i0 = np.floor(f_idx).astype(int)
        frac = f_idx - i0
        valid = (i0 >= 0) & (i0 < nfbins-1)

        t_idx = np.nonzero(mask)[0][valid]
        i0 = i0[valid]; i1 = i0 + 1
        frac = frac[valid]; wv = w[valid]

        np.add.at(H, (i0, t_idx), (1.0 - frac) * wv)
        np.add.at(H, (i1, t_idx), frac * wv)

    # optional smoothing (freq, time)
    if smooth_sigma is not None:
        H = gaussian_filter(H, sigma=smooth_sigma, mode="nearest")

    # convert to dB
    H /= (H.max() + 1e-12)
    HdB = 10*np.log10(H + 1e-12)
    vmax = HdB.max()
    vmin = vmax - dynamic_range_db

    t = np.arange(T) / fs
    plt.figure(figsize=(11, 5))
    plt.pcolormesh(t, f_edges[:-1], HdB, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.ylim(0, fmax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Combined Hilbert–Huang Spectrum – channel {roi_idx+1} ({method_name})")
    cbar = plt.colorbar()
    cbar.set_label("Power (dB, summed across IMFs)")
    plt.tight_layout()
    plt.show()


def plot_marginal_spectrum(inst_amp, inst_freq, roi_idx=0, fmax=0.3, bins=200, save_path=None):
    """
    Marginal Hilbert amplitude spectrum (integrated over time).
    """
    K, T, R = inst_amp.shape
    freq_bins = np.linspace(0, fmax, bins)
    spec = np.zeros(bins - 1)
    for k in range(K):
        f = inst_freq[k, :, roi_idx]
        a = inst_amp[k, :, roi_idx]
        inds = np.digitize(f, freq_bins) - 1
        np.add.at(spec, inds.clip(0, bins - 2), a)

    plt.figure(figsize=(8, 4))
    plt.plot(freq_bins[:-1], spec, color="steelblue")
    add_freq_bands(plt.gca(), alpha=0.1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(f"Marginal Hilbert Spectrum – ROI {roi_idx}")
    plt.xlim(0, fmax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_example_subjects(results, plot_hht=True, plot_psd=True, save_figs=False):
    """
    Plot example decompositions (IMFs, spectra, and HHTs) for one MDD and one HC subject.
    Automatically computes Hilbert–Huang transforms before plotting.

    Args:
        results: list of subject results dicts (with keys like 'group', 'subject', 'imfs', 'freqs', etc.)
        plot_hht: bool, whether to plot Hilbert–Huang spectra
        plot_psd: bool, whether to include PSD plots
        save_figs: bool, whether to save figures to file
    """

    mdd_data = next((r for r in results if r["group"] == "MDD"), None)
    hc_data  = next((r for r in results if r["group"] == "HC"), None)

    for group, data in [("MDD", mdd_data), ("HC", hc_data)]:
        if data is None:
            print(f"No data found for group {group}")
            continue

        subj_label = f"{data['subject']} ({group}) run {data['run_idx']}"
        X = load_bold_matrix(data["run_file"])
        fs = 1 / 0.8
        
        # --- Compute HHT on the fly ---
        inst_amp, inst_freq = compute_hht(data["imfs"], fs=fs, smooth_sigma=1)

        # --- 1. Time-domain IMFs ---
        plot_imfs(
            X=X,
            imfs=data["imfs"],
            roi_idx=0,
            freqs=data["freqs"],
            subj_label=subj_label,
            method_name="VLMD",
            save_path=f"{data['subject']}_imfs.png" if save_figs else None,
        )

        # --- 2. IMFs with spectra (Welch PSDs) ---
        if plot_psd:
            plot_imfs_with_spectrum(
                X=X,
                imfs=data["imfs"],
                fs=fs,
                roi_idx=0,
                freqs=data["freqs"],
                subj_label=subj_label,
                method_name="VLMD",
                fmax=0.3,
                save_path=f"{data['subject']}_imfs_psd.png" if save_figs else None,
            )

        # --- 3. Hilbert–Huang per IMF ---
        if plot_hht:
            plot_imfs_with_hht(
                imfs=data["imfs"],
                inst_amp=inst_amp,
                inst_freq=inst_freq,
                fs=fs,
                roi_idx=0,
                freqs=data["freqs"],
                subj_label=subj_label,
                method_name="VLMD",
                fmax=0.3,
                smooth_sigma=1,
                save_path=f"{data['subject']}_hht_per_imf.png" if save_figs else None,
            )

            # --- 4. Combined HHT ---
            plot_combined_hht(
                inst_amp=inst_amp,
                inst_freq=inst_freq,
                fs=fs,
                roi_idx=0,
                method_name="VLMD",
                subj_label=subj_label,
                fmax=0.3,
                smooth_sigma=1,
                save_path=f"{data['subject']}_hht_combined.png" if save_figs else None,
            )