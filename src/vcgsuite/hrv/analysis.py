# ══════════════════════════════════════════════════════════════════════════
#  HRV_ANALYSIS — EDR-Signale + CWT-Scalogram
#
#  Migriert aus HRV_Analysis.py ("BLOCK HRV_ANALYSIS").
#
#  Abhängigkeit: df_rr aus `vcgsuite.hrv.prep.build_rr_dataframe()`
#
#  EDR_CONFIG:  kanalspezifische Filterparameter
#  Funktionen:  extract_edr_signal()   → EDR + Atemfrequenz
#               compute_rsa_coherence()→ RR↔EDR Kreuzkoherenz
#               _compute_cwt()         → Wavelet-Scalogram
#
#  Hinweis Migration: Der "── Ausführung ──"-Abschnitt am Ende des
#  Originalskripts (der beim Import sofort extract_edr_signal(),
#  compute_rsa_coherence() und _compute_cwt() auf einem Notebook-globalen
#  df_rr ausführte) ist entfernt — Bibliotheksmodule dürfen beim Import
#  keine Berechnungen auf Nutzdaten auslösen. Diese Aufrufe obliegen jetzt
#  explizit dem Aufrufer (z. B. `vcgsuite.pipeline` oder Nutzer-Code).
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, coherence as sp_coherence, welch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as _sgf

FS_INTERP = 4.0   # Hz — Resampling-Frequenz

# ── Kanalspezifische Filterparameter ─────────────────────────────────────
#
#  'type':  'hp'  → Hochpass (DC + Trend entfernen)
#           'bp'  → Bandpass (spektrale Selektion)
#  'fc':    Grenzfrequenz [Hz]  (hp) oder (flo, fhi) (bp)
#  'order': Butterworth-Ordnung — hier grundsätzlich 1
#
EDR_CONFIG = {
    'Rpeak_ca': {
        'type': 'hp',
        'fc': 0.03,          # nur DC/Baselinewander entfernen
        'order': 1,
        'label': 'Rpeak_ca — Herzrotation (mechanisch)',
        'color': '#e07030',
    },
    'Rpeak_r': {
        'type': 'hp',
        'fc': 0.03,          # nur DC/Baselinewander entfernen
        'order': 1,
        'label': 'Rpeak_r — Amplitude (Vorlast)',
        'color': '#ab63fa',
    },
    'RR_derived': {
        'type': 'hp',
        'fc': 0.15,          # Sympathikus (<0.15 Hz) entfernen
        'order': 1,             # 1. Ordnung = flache Flanke, kein Überschwingen
        'label': 'RR-Intervall — RSA direkt (neuro-vegetativ)',
        'color': '#00cc96',
    },
}
RESP_COL = 'Rpeak_ca'


def extract_edr_signal(df_rr: pd.DataFrame, resp_col: str | None = None,
                       fs_interp: float = FS_INTERP):
    """
    Extrahiert EDR aus einem der drei Kanäle.

    Filterlogik (aus EDR_CONFIG):
      Rpeak_ca / Rpeak_r → Butterworth HP 1. Ord. 0.03 Hz
          Nur DC + Baselinewander entfernen, Atemsignal vollständig erhalten.
      RR_derived         → Butterworth HP 1. Ord. 0.15 Hz
          Klassische HRV: Sympathikus-Anteil (<0.15 Hz) wegfiltern,
          nur HF-Band (RSA) erhalten.

    Returns
    -------
    t_edr  : gleichmäßige Zeitachse [s]
    edr    : gefiltertes EDR-Signal
    f_resp : dominante Atemfrequenz [Hz]  (None wenn nicht detektierbar)
    rpm    : Atemzüge/min                 (None wenn nicht detektierbar)
    (f_w, p_w) : Welch-PSD des EDR-Signals (für Diagnose)
    """
    if resp_col is None:
        resp_col = RESP_COL
    cfg = EDR_CONFIG[resp_col]
    order = cfg['order']

    raw = df_rr[resp_col].values.astype(float)
    t_raw = df_rr['Time'].values.astype(float)
    valid = np.isfinite(raw) & np.isfinite(t_raw)
    t_raw, raw = t_raw[valid], raw[valid]

    if len(t_raw) < 10:
        raise ValueError(f"Zu wenige valide Werte in '{resp_col}': {len(t_raw)}")

    # Gleichmäßig resamplen (cubic spline)
    t_edr = np.arange(t_raw[0], t_raw[-1], 1.0 / fs_interp)
    y_int = interp1d(t_raw, raw, kind='cubic',
                     bounds_error=False,
                     fill_value='extrapolate')(t_edr)

    # ── Filter: immer Hochpass 1. Ordnung ────────────────────────────────
    nyq = fs_interp / 2.0
    fc = cfg['fc']           # Grenzfrequenz [Hz]
    b_hp, a_hp = butter(order, fc / nyq, btype='high')
    edr = filtfilt(b_hp, a_hp, y_int)

    # Dominante Atemfrequenz via Welch
    # Suchbereich: oberhalb der Grenzfrequenz bis 0.5 Hz
    f_search_lo = max(fc * 1.5, 0.05)   # etwas Abstand zur Grenzfreq.
    f_search_hi = 0.5
    nperseg = min(256, len(edr) // 2)
    f_w, p_w = welch(edr, fs=fs_interp, nperseg=nperseg)
    rm = (f_w >= f_search_lo) & (f_w <= f_search_hi)
    f_resp = float(f_w[rm][np.argmax(p_w[rm])]) if rm.any() else None
    rpm = f_resp * 60 if f_resp else None

    return t_edr, edr, f_resp, rpm, (f_w, p_w)


def compute_rsa_coherence(df_rr: pd.DataFrame, t_edr: np.ndarray, edr_signal: np.ndarray,
                          fs_interp: float = FS_INTERP, resp_col: str | None = None):
    """Kreuzkoherenz RR ↔ EDR im physiologisch relevanten Atemband."""
    if resp_col is None:
        resp_col = RESP_COL
    cfg = EDR_CONFIG[resp_col]

    # Kohärenz-Suchbereich: immer Atemband 0.05–0.5 Hz
    # (unabhängig vom Filter — wir wollen die Kopplung im Atemband sehen)
    band_lo = max(cfg['fc'] * 1.5, 0.05)
    band_hi = 0.5

    rr_int = interp1d(df_rr['Time'].values,
                       df_rr['RR_interval'].values,
                       kind='cubic',
                       bounds_error=False,
                       fill_value='extrapolate')(t_edr)
    rr_int -= np.mean(rr_int)

    n = min(len(rr_int), len(edr_signal))
    nperseg = min(256, n // 2)
    f_coh, coh = sp_coherence(rr_int[:n], edr_signal[:n],
                               fs=fs_interp, nperseg=nperseg)

    rm = (f_coh >= band_lo) & (f_coh <= band_hi)
    rsa_coh = float(np.mean(coh[rm])) if rm.any() else 0.0
    f_rsa_pk = float(f_coh[rm][np.argmax(coh[rm])]) if rm.any() else None

    return f_coh, coh, f_rsa_pk, rsa_coh


def _compute_cwt(t_sig: np.ndarray, edr_sig: np.ndarray, fs_interp: float = FS_INTERP,
                 freqmin: float = 0.01, freqmax: float = 0.5, nscales: int = 150):
    """Morlet-CWT + geglättete Instantanfrequenz-Ridge."""
    wavelet = 'cmor2.5-1.5'
    freqs_cw = np.linspace(freqmin, freqmax, nscales)
    scales = pywt.central_frequency(wavelet) * fs_interp / freqs_cw

    y = edr_sig - np.mean(edr_sig)
    coeffs, _ = pywt.cwt(y, scales, wavelet,
                          sampling_period=1.0 / fs_interp)
    power = np.abs(coeffs) ** 2

    imax = power.argmax(axis=0)
    ridge = freqs_cw[imax]
    wl_r = min(51, len(ridge) - (1 - len(ridge) % 2))
    if wl_r >= 5:
        ridge = _sgf(ridge, window_length=wl_r, polyorder=1)

    return freqs_cw, power, ridge
