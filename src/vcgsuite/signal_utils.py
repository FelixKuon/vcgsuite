# ══════════════════════════════════════════════════════════════════════════
#  ALLGEMEINE SIGNAL-HILFSFUNKTIONEN
#
#  Konsolidiert die im ursprünglichen Notebook-Export mehrfach leicht
#  unterschiedlich implementierte Savitzky-Golay-Glättung:
#    - `apply_savgol`  in plot_loop_features.py (3× nahezu identische Kopien)
#    - `_savgol`       in beat_to_beat_VCG_rotation.py
#    - `_savgol_safe`  in ComplexAmplitude_r_peak_edr.py
#    - `_sg`           in ActivationTimeMapping.py
#
#  Die Varianten unterschieden sich in zwei Punkten:
#    1. NaN-Behandlung: `apply_savgol` interpoliert NaNs linear vor der
#       Filterung und setzt sie danach zurück (Zeitbezug bleibt erhalten).
#       `_savgol` / `_savgol_safe` / `_sg` haben NaNs stattdessen einfach
#       herausgeschnitten ("kompaktiert") und den Filter nur auf die
#       verbleibenden, nicht mehr äquidistanten Werte angewendet – das
#       verzerrt die Glättung, sobald NaN-Lücken vorkommen.
#    2. Fensterlänge/Polynomgrad-Clipping bei zu kurzen Arrays bzw. geradem
#       Fenster: leicht unterschiedliche, aber im Kern gleichwertige
#       Rundungslogik.
#
#  `savgol_smooth()` unten übernimmt die robustere NaN-Behandlung
#  (Interpolation + Rücksetzen) aus `apply_savgol` und vereinheitlicht das
#  Clipping von Fensterlänge/Polynomgrad für zu kurze Arrays und gerade
#  Fensterlängen.
# ══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def savgol_smooth(arr, window: int, polyorder: int) -> np.ndarray:
    """
    Robuste Savitzky-Golay-Glättung mit einheitlicher NaN- und Kantenbehandlung.

    Parameters
    ----------
    arr       : np.ndarray | pd.Series  –  Eingangssignal, darf NaNs enthalten.
    window    : int  –  Gewünschte Fensterlänge (wird intern auf eine
                        ungerade Zahl <= len(arr) reduziert, falls nötig).
    polyorder : int  –  Gewünschter Polynomgrad (wird intern auf < window
                        begrenzt).

    Verhalten
    ---------
    - NaNs werden vor der Filterung linear interpoliert (np.interp) und nach
      der Filterung an ihren ursprünglichen Positionen wieder auf NaN
      gesetzt, damit die Filterung nicht durch fehlende Werte abbricht und
      der zeitliche Bezug zwischen den Samples erhalten bleibt.
    - Ist das Array komplett NaN, wird es unverändert zurückgegeben.
    - Ist das (nach NaN-Interpolation) verfügbare Array zu kurz für eine
      sinnvolle Glättung (Fensterlänge < polyorder + 1, oder < 3), wird das
      Array unverändert (mit ursprünglichen NaNs) zurückgegeben, statt einen
      Fehler zu werfen.

    Returns
    -------
    np.ndarray, gleiche Länge wie Eingabe.
    """
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy(dtype=float)
    else:
        arr = np.asarray(arr, dtype=float).copy()

    n = len(arr)
    nan_mask = np.isnan(arr)

    if nan_mask.all():
        return arr

    if nan_mask.any():
        xp = np.where(~nan_mask)[0]
        arr[nan_mask] = np.interp(np.where(nan_mask)[0], xp, arr[xp])

    # Fensterlänge auf ungerade Zahl <= n reduzieren
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    # Polynomgrad auf < Fensterlänge begrenzen
    p = min(polyorder, w - 1)

    if w < 3 or p < 1 or w <= p:
        # Zu kurz für eine sinnvolle Glättung → unverändert (mit NaNs) zurückgeben
        out = arr.copy()
        out[nan_mask] = np.nan
        return out

    smoothed = savgol_filter(arr, window_length=w, polyorder=p)
    smoothed[nan_mask] = np.nan
    return smoothed
