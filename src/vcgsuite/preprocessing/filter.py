# ============================================================
#  SIGNALFILTERUNG
#  Pipeline:  1. ZapLine  (50 Hz + alle Harmonischen bis Nyquist)
#             2. FIR Hochpass  (Baseline Wander)
#             3. FIR Tiefpass  (Hochfrequentes Rauschen)
# ============================================================

import numpy as np
from meegkit import dss
from scipy.signal import firwin, filtfilt

from ..config import FS

# ── Filterparameter  (zentral anpassbar)
FLINE        = 50.0   # Netzfrequenz in Hz
NREMOVE_ZAP  = 1      # DSS-Komponenten für ZapLine
HP_CUTOFF    = 1.5    # Hochpass-Grenzfrequenz in Hz
HP_TAPS      = 201    # FIR-Taps Hochpass
LP_CUTOFF    = 37.5   # Tiefpass-Grenzfrequenz in Hz
LP_TAPS      = 21     # FIR-Taps Tiefpass


def zapline(signals: list, fs: int = FS,
            fline: float = FLINE,
            nremove: int = NREMOVE_ZAP) -> list:
    """
    DSS ZapLine: entfernt Netzfrequenz und alle Harmonischen bis Nyquist.

    Parameters
    ----------
    signals : list of np.ndarray
        Liste von 1D-Signalarrays, beliebige Kanalanzahl.

    Returns
    -------
    list of np.ndarray, gleiche Reihenfolge wie Eingabe.
    """
    raw = np.column_stack(signals)
    clean, _ = dss.dss_line(
        raw,
        fline=fline,
        sfreq=fs,
        nremove=nremove,
        nfft=4 * fs,
        nkeep=None,
        show=False,
    )
    n_harm = fs / 2 // fline
    print(f"ZapLine: {fline} Hz, bis {n_harm:.0f}. Harmonische, nremove={nremove}")
    return [clean[:, i] for i in range(clean.shape[1])]


def fir_bandpass(signals: list, fs: int = FS,
                 hp_cutoff: float = HP_CUTOFF, hp_taps: int = HP_TAPS,
                 lp_cutoff: float = LP_CUTOFF, lp_taps: int = LP_TAPS) -> list:
    """
    Zweistufiger FIR-Filter: Hochpass → Tiefpass (zero-phase via filtfilt).

    Returns
    -------
    list of np.ndarray, gleiche Reihenfolge wie Eingabe.
    """
    fir_hp = firwin(hp_taps, hp_cutoff, pass_zero=False, fs=fs)
    fir_lp = firwin(lp_taps, lp_cutoff, pass_zero=True,  fs=fs)
    out = []
    for s in signals:
        s_hp   = filtfilt(fir_hp, [1], s)
        s_filt = filtfilt(fir_lp, [1], s_hp)
        out.append(s_filt)
    print(f"FIR: HP {hp_cutoff} Hz ({hp_taps} Taps)  →  LP {lp_cutoff} Hz ({lp_taps} Taps)")
    return out


def filter_pipeline(signals: list, fs: int = FS,
                    fline: float = FLINE, nremove_zap: int = NREMOVE_ZAP,
                    hp_cutoff: float = HP_CUTOFF, hp_taps: int = HP_TAPS,
                    lp_cutoff: float = LP_CUTOFF, lp_taps: int = LP_TAPS,
                    use_zapline: bool = True) -> list:
    """
    Vollständige Filterpipeline: (optional ZapLine) → FIR HP → FIR LP.

    Parameters
    ----------
    signals : list[np.ndarray]
        Rohe Signalarrays, beliebige Kanalanzahl.
    use_zapline : bool
        True (Default) = ZapLine (Netzfrequenz-Entfernung via DSS) läuft
        vor dem FIR-Bandpass, wie ursprünglich. False = ZapLine wird
        komplett übersprungen, es läuft nur noch der FIR-Bandpass. Sinnvoll
        z. B. wenn die Aufnahme kaum Netzbrumm enthält oder `meegkit`/DSS
        bei einer bestimmten Aufnahme Artefakte statt Verbesserung bringt.

    Returns
    -------
    list[np.ndarray], gefiltert, gleiche Reihenfolge.
    """
    if use_zapline:
        signals = zapline(signals, fs=fs, fline=fline, nremove=nremove_zap)
    signals = fir_bandpass(signals, fs=fs,
                           hp_cutoff=hp_cutoff, hp_taps=hp_taps,
                           lp_cutoff=lp_cutoff, lp_taps=lp_taps)
    return signals
