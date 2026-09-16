import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_r_peaks(df: pd.DataFrame, percentile: float = 95, prom_frac: float = 0.30,
                    min_rr_ms: float = 400.0, window_s: float = 10.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Detektiert R_peak+ im A_abs-Signal mit adaptiver, fensterbasierter Schwelle.

    Parameters
    ----------
    window_s : float
        Breite des gleitenden Fensters in Sekunden für lokale Normierung.
        Typisch: 5–15 s. Kleiner = reaktiver, größer = stabiler.

    Returns
    -------
    r_peak_times  : np.ndarray  –  Zeitpunkte [s]
    r_peak_values : np.ndarray  –  Amplitudenwerte
    """
    t   = df['Time'].values
    sig = df['A_abs'].values.astype(float)
    fs  = 1.0 / np.median(np.diff(t))

    win_samples = max(1, int(window_s * fs))

    # ── Lokale Statistiken per gleitendem Fenster ─────────────────────────
    # Echtes lokales Perzentil via pandas rolling().quantile() (robust).
    # Hinweis Migration: Im Original (detect_r_peaks.py) wurde hier zuerst
    # eine schnellere, aber ungenutzte Approximation über uniform_filter1d
    # berechnet, deren Ergebnis sofort durch die untenstehende
    # rolling().quantile()-Variante überschrieben wurde, ohne je verwendet
    # zu werden. Dieser tote Code wurde entfernt.
    local_height = (
        pd.Series(sig)
        .rolling(win_samples, center=True, min_periods=win_samples // 2)
        .quantile(percentile / 100.0)
        .bfill().ffill()
        .values
    )
    local_prom = prom_frac * local_height

    # ── Peak-Detektion mit sample-genauen Schwellen ───────────────────────
    peaks, _ = find_peaks(
        sig,
        distance  = int(min_rr_ms / 1000.0 * fs),
        height    = local_height,   # array → pro Sample adaptiv
        prominence= local_prom,     # array → pro Sample adaptiv
    )

    rr_ms = np.diff(t[peaks]) * 1000
    print(f"Sampling-Rate:     {fs:.1f} Hz")
    print(f"Fensterbreite:     {window_s:.1f} s  ({win_samples} Samples)")
    print(f"Detektierte Peaks: {len(peaks)}")
    print(f"Ø RR-Abstand:      {np.mean(rr_ms):.1f} ms")
    print(f"Ø HF:              {60 / (np.mean(rr_ms) / 1000):.1f} bpm")
    print(f"RR-Bereich:        {np.min(rr_ms):.0f} – {np.max(rr_ms):.0f} ms")

    return t[peaks], sig[peaks]
