
def plot_imfs_and_spectrum(imfs, fs=1.0, signal=None, n_freq_bins=200):
    """
    Plot imfs + Hilbert spectrum per IMF in Plotly.
   
    inputs:
    - imfs: array of imfs. From PyEMD.EMD().emd(signal) (Or EEMD etc)
    - fs: sampling frequency of the original signal
    - signal: original signal (for top plot, 1D array)
    - n_freq_bins: number of frequency bins for HHT (200 ok)
 
    example:
    import numpy as np
    from PyEMD import EMD
    from scipy.signal import hilbert
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
 
    # signal: 1d array
    emd = EMD()
    imfs = emd.emd(signal)
    plot_imfs_and_spectrum(imfs, fs=fs, signal=signal)
    """
    n_imfs = len(imfs)
    t = np.arange(imfs.shape[1]) / fs
 
    # Hilbert transform
    analytic = hilbert(imfs, axis=1)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase, axis=1) * fs / (2*np.pi)
    inst_freq = np.abs(inst_freq)
    t_if = t[1:]
 
    # Hilbert-Huang Spectrum per IMF
    max_freq = np.percentile(inst_freq, 99)
    freq_bins = np.linspace(0, max_freq, n_freq_bins)
 
    hht_maps = []
    for i in range(n_imfs):
        hht = np.zeros((n_freq_bins, len(t_if)))
        for j in range(len(t_if)):
            f = inst_freq[i, j]
            a = amp[i, j+1]
            if 0 <= f < max_freq:
                idx = np.searchsorted(freq_bins, f)
                hht[idx, j] += a
        hht = gaussian_filter(hht, sigma=1.0)
        hht_maps.append(hht)
 
    # Plotly nice looking plot
    total_rows = n_imfs + (1 if signal is not None else 0)
    fig = make_subplots(
        rows=total_rows,
        cols=2,
        shared_xaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.04,
        column_widths=[0.45, 0.55],
        specs=[
            [{"colspan": 2}, None] if (signal is not None and r == 0)
            else [{}, {}]
            for r in range(total_rows)
        ],
        subplot_titles=[
            "Original signal" if signal is not None and i == 0 else
            (f"IMF {i if signal is not None else i+1}" if j == 0 else
            f"Hilbert–Huang Spectrum IMF {i if signal is not None else i+1}")
            for i in range(total_rows)
            for j in range(2)
            if not (signal is not None and i == 0 and j == 1)
        ]
    )
 
    row_offset = 1 if signal is not None else 0
 
    # original signal on top
    if signal is not None:
        fig.add_trace(
            go.Scatter(x=t, y=signal, mode='lines',
                    line=dict(color='black', width=1.5),
                    name='Original signal'),
            row=1, col=1
        )
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
 
    # imfs + hht
    for i in range(n_imfs):
        r = i + row_offset + 1
        fig.add_trace(
            go.Scatter(x=t, y=imfs[i], mode='lines',
                    line=dict(color='royalblue'),
                    name=f'IMF {i+1}'),
            row=r, col=1
        )
        fig.add_trace(
            go.Heatmap(
                z=hht_maps[i],
                x=t_if,
                y=freq_bins,
                colorscale='turbo',
                showscale=(i == 0)
            ),
            row=r, col=2
        )
 
        fig.update_yaxes(title_text="Amplitude", row=r, col=1)
        fig.update_yaxes(title_text="Freq [Hz]", row=r, col=2)
 
    # plotly layout
    fig.update_xaxes(title_text="Time [s]", row=total_rows, col=1)
    fig.update_xaxes(title_text="Time [s]", row=total_rows, col=2)
    fig.update_layout(
        height=260 * total_rows,
        width=1600,
        showlegend=False,
        title="Hilbert–Huang Transform per IMF",
        template="plotly_white",
        margin=dict(t=80, l=50, r=50, b=50),
    )
    fig.show()
 