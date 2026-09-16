# ══════════════════════════════════════════════════════════════════════════
#  Beat-to-Beat: Komplexe Amplitude am R-Peak + RR-Intervall
#
#  Inputs:  df_analysis     ← z. B. Ausgabe der Wrapper-Pipeline
#                              erwartet Spalten: Time, X, Y, Z, r, phi, theta
#           r_peak_times    ← detect_r_peaks()
#
#  Outputs: df_beat         ← beat-to-beat DataFrame mit:
#                              beat_id, t_rpeak, r_rpeak, ca_rpeak, RR_ms
#
#  Hinweis Migration: `plot_rpeak_ca()` aus dem Notebook-Export
#  (ComplexAmplitude_r_peak_edr.py) ist bewusst NICHT Teil dieses Moduls —
#  Plotting wandert in einem späteren Schritt in ein eigenes viz-Paket.
# ══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


def compute_rpeak_ca(df_analysis: pd.DataFrame,
                     r_peak_times: np.ndarray) -> pd.DataFrame:
    """
    Berechnet die komplexe Amplitude am R-Peak aus df_analysis.

    Kugelkoordinaten (physikalische Konvention):
        r     = |V⃗|              Vektorbetrag (bereits in df_analysis)
        phi   = arccos(z / r)   Polarwinkel   [0, π]
        theta = arctan2(y, x)   Azimutwinkel  [-π, π]

    Komplexe Amplitude (CA):
        CA = |phi + i·theta|

    Nächster Sample-Index wird per argmin(|t_arr - t_query|) bestimmt.

    Parameters
    ----------
    df_analysis   : pd.DataFrame  –  Spalten: Time, r, phi, theta (min.)
    r_peak_times  : array-like    –  R-Peak-Zeitpunkte in Sekunden

    Returns
    -------
    df_beat : pd.DataFrame
        beat_id   – fortlaufend ab 1
        t_rpeak   – R-Peak-Zeitpunkt [s]
        r_rpeak   – Vektorbetrag r am R-Peak [mV]
        ca_rpeak  – komplexe Amplitude |phi + i·theta| [rad]
        RR_ms     – RR-Intervall zum Vorgänger-Beat [ms]  (NaN für Beat 1)
    """
    t_arr   = df_analysis["Time"].to_numpy(dtype=float)
    r_arr   = df_analysis["r"].to_numpy(dtype=float)
    phi_arr = df_analysis["phi"].to_numpy(dtype=float)
    tht_arr = df_analysis["theta"].to_numpy(dtype=float)

    r_pt = np.asarray(r_peak_times, dtype=float)
    n    = len(r_pt)

    # Nächster Sample-Index pro R-Peak
    idx = np.array([int(np.argmin(np.abs(t_arr - t))) for t in r_pt])

    r_vals  = r_arr[idx]
    ca_vals = np.abs(phi_arr[idx] + 1j * tht_arr[idx])
    rr_ms   = np.concatenate([[np.nan], np.diff(r_pt) * 1000.0])

    df_beat = pd.DataFrame({
        "beat_id":  np.arange(1, n + 1),
        "t_rpeak":  r_pt,
        "r_rpeak":  r_vals,
        "ca_rpeak": ca_vals,
        "RR_ms":    rr_ms,
    })

    print(f"✅ {n} Beats  |  "
          f"CA  μ={ca_vals.mean():.3f}  σ={ca_vals.std():.3f}  |  "
          f"RR  μ={np.nanmean(rr_ms):.0f} ms  σ={np.nanstd(rr_ms):.0f} ms")
    return df_beat
