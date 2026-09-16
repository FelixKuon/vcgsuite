# ══════════════════════════════════════════════════════════════════════════
#  ROTATIONSBEREINIGTE PROJEKTION  (VLS)
#
#  Beat-für-Beat wird der XYZ-Vektor zum R-Peak-Zeitpunkt als lokale
#  Referenzachse genutzt. Jeder Trajektoriepunkt des Beats wird auf
#  diese Achse projiziert.
#  → Atemrotation wird rausgerechnet, da sich die Referenzachse
#    mit dem Herzdipol mitbewegt.
#
#  Hinweis Migration: Im Original (compute_vls.py) schrieb die
#  Ausführungszeile das Ergebnis als Seiteneffekt direkt in
#  `df_analysis['VLS']`. Die Funktion selbst gab bereits ein np.ndarray
#  zurück — hier wird nur die Seiteneffekt-Zuweisung entfernt: die
#  Zuweisung in die DataFrame-Spalte obliegt jetzt dem Aufrufer, z. B.:
#      df_analysis['VLS'] = compute_vls(df_analysis, r_peak_times)
# ══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


def compute_vls(df_signal: pd.DataFrame, r_peak_times: np.ndarray) -> np.ndarray:
    """
    Berechnet Beat-für-Beat die vorzeichenbehaftete Projektion der
    XYZ-Trajektorie auf den XYZ-Vektor am jeweiligen R-Peak.

    Parameters
    ----------
    df_signal     : pd.DataFrame  –  df_analysis (Time, X, Y, Z)
    r_peak_times  : np.ndarray    –  Zeitpunkte der R-Peaks [s],
                                     Output von detect_r_peaks()

    Returns
    -------
    vls : np.ndarray  –  Projektionssignal, gleiche Länge wie df_signal.
                         Die Zuweisung in eine DataFrame-Spalte (z. B.
                         df_analysis['VLS']) obliegt dem Aufrufer.
    """
    t_arr = df_signal['Time'].values
    x_arr = df_signal['X'].values
    y_arr = df_signal['Y'].values
    z_arr = df_signal['Z'].values
    vls   = np.zeros(len(df_signal))

    def _get_idx(t_query):
        return int(np.clip(np.searchsorted(t_arr, t_query), 0, len(t_arr) - 1))

    def _interp_xyz(t_query):
        j = int(np.clip(np.searchsorted(t_arr, t_query), 1, len(t_arr) - 1))
        a = (t_query - t_arr[j-1]) / (t_arr[j] - t_arr[j-1] + 1e-12)
        return np.array([
            x_arr[j-1] + a * (x_arr[j] - x_arr[j-1]),
            y_arr[j-1] + a * (y_arr[j] - y_arr[j-1]),
            z_arr[j-1] + a * (z_arr[j] - z_arr[j-1]),
        ])

    n_ok = n_skip = 0

    for i, t_rpeak in enumerate(r_peak_times):

        # Referenzvektor = XYZ-Zeiger zum R-Peak-Zeitpunkt
        v_ref  = _interp_xyz(t_rpeak)
        norm   = np.linalg.norm(v_ref)
        if norm < 1e-10:
            n_skip += 1
            continue
        v_ref_norm = v_ref / norm

        # Zeitgrenzen: aktueller R-Peak → nächster R-Peak
        t0   = float(t_rpeak)
        t1   = float(r_peak_times[i + 1]) if i + 1 < len(r_peak_times) else t_arr[-1]
        idx0 = _get_idx(t0)
        idx1 = min(_get_idx(t1) + 1, len(t_arr))
        idx_range = np.arange(idx0, idx1)

        traj         = np.stack([x_arr[idx_range],
                                 y_arr[idx_range],
                                 z_arr[idx_range]], axis=1)
        vls[idx_range] = traj @ v_ref_norm
        n_ok += 1

    print(f"VLS:  {n_ok} Beats projiziert  |  {n_skip} übersprungen")
    return vls
