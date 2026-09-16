# ══════════════════════════════════════════════════════════════════════════
#  HRV-HELFER  –  Kernmetriken, Normwerte, PSD
#
#  Migriert aus HRV_helpers.py ("BLOCK HRV_HELPERS").
#
#  Hinweis Migration: Das Original druckte beim Import ("✅ HRV_HELPERS —
#  alle Funktionen und Konstanten geladen.") eine Diagnosemeldung als
#  Notebook-Zellen-Seiteneffekt. Das ist für ein importierbares Modul einer
#  Bibliothek unerwünscht (jeder `import vcgsuite` würde sonst drucken) und
#  wurde daher ersatzlos entfernt — die Funktionen/Konstanten selbst sind
#  unverändert migriert.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, filtfilt


# ── Norm-Parameter (Kubios-orientiert) ───────────────────────────────────
PNS_REF = {'mean_rr': 1.0,  'rmssd': 0.035, 'sd1_pct': 25}
PNS_STD = {'mean_rr': 0.15, 'rmssd': 0.015, 'sd1_pct': 10}
SNS_REF = {'mean_hr': 65,   'SI': 6,         'sd2_pct': 45}
SNS_STD = {'mean_hr': 10,   'SI': 2,         'sd2_pct': 15}


# ── PSD-Frequenzbänder für Dashboard-Plot ────────────────────────────────
# Format: {name: (f_low, f_high, fillcolor, linecolor)}
BANDS = {
    'VLF': (0.003, 0.04,  'rgba(180,180,255,0.35)', '#8888ff'),
    'LF' : (0.04,  0.15,  'rgba(255,200,100,0.35)', '#ffaa00'),
    'HF' : (0.15,  0.40,  'rgba(100,220,180,0.35)', '#00cc96'),
}


def compute_sd1_sd2(rr: np.ndarray) -> tuple[float, float]:
    """Poincaré SD1/SD2 (kurz-/langfristige RR-Variabilität)."""
    diffs = np.diff(rr)
    SD1 = np.sqrt(np.var(diffs, ddof=1) / 2)
    SD2 = np.sqrt(2 * np.var(rr, ddof=1) - np.var(diffs, ddof=1) / 2)
    return SD1, SD2


def compute_baevsky_stress_index(rr_s: np.ndarray) -> float:
    """Baevsky Stress-Index aus dem RR-Histogramm."""
    bins = np.linspace(0.3, 2.0, 50)
    hist, bin_edges = np.histogram(rr_s, bins=bins)
    hist_smooth = np.convolve(hist, np.ones(3) / 3, mode='same')
    mode_bin = np.argmax(hist_smooth)
    mode = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
    amo = 100.0 * hist_smooth[mode_bin] / len(rr_s)
    range_rr = np.ptp(np.clip(rr_s, 0.3, 2.0))
    return np.log(amo / (2 * mode * range_rr + 1e-6) + 1)


def compute_psd_welch(rr_times: np.ndarray, rr_intervals: np.ndarray,
                      fs_interp: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Kubisches Interpolations-Resampling → Welch-PSD aus ungleichmäßigen RR-Daten."""
    duration = rr_times[-1] - rr_times[0]
    n_samples = int(duration * fs_interp)
    nperseg = min(256, n_samples // 2)
    if nperseg < 16:
        return np.array([]), np.array([])
    t_uni = np.arange(rr_times[0], rr_times[-1], 1 / fs_interp)
    rr_uni = interp1d(rr_times, rr_intervals, kind='cubic',
                      bounds_error=False, fill_value='extrapolate')(t_uni)
    rr_uni -= np.mean(rr_uni)
    freqs, psd = welch(rr_uni, fs=fs_interp, nperseg=nperseg,
                       window='hann', scaling='density')
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Bandleistung via Trapezregel."""
    m = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[m], freqs[m]) if m.sum() >= 2 else 0.0


def compute_hrv_full(rr) -> dict:
    """
    Berechnet alle HRV-Kennwerte.

    Parameter
    ---------
    rr : array-like
        RR-Intervalle in Sekunden ODER Millisekunden.
        Automatische Erkennung: Median > 3 → ms → wird in s konvertiert.

    Returns
    -------
    dict mit: mean_rr, mean_hr, sdnn, rmssd, pnn50,
              SD1, SD2, SI (Baevsky),
              PNS_index, SNS_index, pns_zs, sns_zs
    """
    rr = np.asarray(rr, dtype=float)
    if np.median(rr) > 3:          # ms → s
        rr = rr / 1000.0

    mean_rr = np.mean(rr)
    mean_hr = 60.0 / mean_rr
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    pnn50 = 100.0 * np.mean(np.abs(np.diff(rr)) > 0.05)
    SD1, SD2 = compute_sd1_sd2(rr)
    sd1_pct = (SD1 / mean_rr) * 100
    sd2_pct = (SD2 / mean_rr) * 100
    SI = compute_baevsky_stress_index(rr)

    pns_zs = [
        (mean_rr - PNS_REF['mean_rr']) / PNS_STD['mean_rr'],
        (rmssd - PNS_REF['rmssd']) / PNS_STD['rmssd'],
        (sd1_pct - PNS_REF['sd1_pct']) / PNS_STD['sd1_pct'],
    ]
    sns_zs = [
        (mean_hr - SNS_REF['mean_hr']) / SNS_STD['mean_hr'],
        (SI - SNS_REF['SI']) / SNS_STD['SI'],
        (sd2_pct - SNS_REF['sd2_pct']) / SNS_STD['sd2_pct'],
    ]

    return dict(
        mean_rr=mean_rr, mean_hr=mean_hr,
        sdnn=sdnn, rmssd=rmssd, pnn50=pnn50,
        SD1=SD1, SD2=SD2, SI=SI,
        PNS_index=float(np.mean(pns_zs)),
        SNS_index=float(np.mean(sns_zs)),
        pns_zs=pns_zs, sns_zs=sns_zs,
    )


def compute_rsa_amplitude(df_rr_w, f_resp: float | None,
                          fs_interp: float = 4.0, bw: float = 0.025) -> float | None:
    """
    RSA-Amplitude direkt aus der EDR-Atemfrequenz:
    RR → Resampling → schmaler Bandpass (f_resp ± bw Hz)
    → Halbamplitude (Peak-to-Peak / 2) in ms.

    Unabhängig von fixen HF-Bandgrenzen — nutzt die tatsächlich
    detektierte Atemrate des EDR-Algorithmus.
    SD1 im Poincaré-Plot ≈ RSA-Amplitude (beide ∝ Vagusaktivität).
    """
    if f_resp is None or f_resp <= 0 or len(df_rr_w) < 15:
        return None

    t_b = df_rr_w["Time"].values
    rr = df_rr_w["RR_interval"].values
    t_u = np.arange(t_b[0], t_b[-1], 1.0 / fs_interp)
    rr_u = interp1d(t_b, rr, kind="cubic",
                    bounds_error=False, fill_value="extrapolate")(t_u)
    rr_u -= np.mean(rr_u)

    nyq = fs_interp / 2.0
    flo = max(0.005, f_resp - bw)
    fhi = min(nyq * 0.95, f_resp + bw)
    if flo >= fhi:
        return None

    b, a = butter(4, [flo / nyq, fhi / nyq], btype="band")
    rr_bp = filtfilt(b, a, rr_u)

    return (np.max(rr_bp) - np.min(rr_bp)) / 2.0 * 1000.0  # → ms


def compute_rsa_hf_power(df_rr_w, f_resp: float | None,
                         fs_interp: float = 4.0, bw: float = 0.025) -> float | None:
    """
    RSA als ln(HF-Power) in ms² — direkt vergleichbar mit Normwert-Studien.
    Bandpass identisch zu compute_rsa_amplitude, aber RMS² statt Peak-to-Peak.
    """
    if f_resp is None or f_resp <= 0 or len(df_rr_w) < 15:
        return None

    t_b = df_rr_w["Time"].values
    rr = df_rr_w["RR_interval"].values
    t_u = np.arange(t_b[0], t_b[-1], 1.0 / fs_interp)
    rr_u = interp1d(t_b, rr, kind="cubic",
                    bounds_error=False, fill_value="extrapolate")(t_u)
    rr_u -= np.mean(rr_u)

    nyq = fs_interp / 2.0
    flo = max(0.005, f_resp - bw)
    fhi = min(nyq * 0.95, f_resp + bw)
    if flo >= fhi:
        return None

    b, a = butter(4, [flo / nyq, fhi / nyq], btype="band")
    rr_bp = filtfilt(b, a, rr_u)

    # RMS² = mittlere Leistung im Band → in ms² (rr_u ist bereits in ms)
    power_ms2 = np.mean(rr_bp ** 2)

    if power_ms2 <= 0:
        return None

    return float(np.log(power_ms2))   # ln(ms²)
