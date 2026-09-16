# ══════════════════════════════════════════════════════════════════════════
#  Beat-to-Beat VCG-Rotation: r, komplexe Amplitude, RR
#
#  Hinweis Migration: `plot_beat_rotation()` aus dem Notebook-Export
#  (beat_to_beat_VCG_rotation.py) ist bewusst NICHT Teil dieses Moduls —
#  Plotting wandert in einem späteren Schritt in ein eigenes viz-Paket.
#  Hier bleibt nur die Compute-Logik.
#
#  `_nearest_spherical()` griff im Original implizit auf die Notebook-weiten
#  Variablen `t_arr`, `x_arr`, `y_arr`, `z_arr` (aus einer vorherigen Zelle)
#  zu. Da es dafür in einem Modul keine globalen Notebook-Variablen gibt,
#  nimmt `compute_beat_rotation()` jetzt zusätzlich `df_analysis` entgegen
#  und reicht die Arrays explizit an `_nearest_spherical()` durch.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ..signal_utils import savgol_smooth


def _nearest_spherical(t_query: float,
                       t_arr: np.ndarray, x_arr: np.ndarray,
                       y_arr: np.ndarray, z_arr: np.ndarray) -> tuple[float, float]:
    """
    Nächster Sample zu t_query (kein Interpolieren) → (r, |φ+iθ|).
    """
    j = int(np.argmin(np.abs(t_arr - t_query)))
    x, y, z = x_arr[j], y_arr[j], z_arr[j]
    r     = float(np.sqrt(x**2 + y**2 + z**2))
    theta = float(np.arctan2(y, x))
    phi   = float(np.arccos(np.clip(z / (r + 1e-10), -1, 1)))
    return r, float(np.abs(phi + 1j * theta))


def compute_beat_rotation(df_annot: pd.DataFrame, df_analysis: pd.DataFrame,
                          smooth_window: int | None = None,
                          smooth_poly: int = 2) -> pd.DataFrame:
    """
    Berechnet Beat-für-Beat: r, komplexe Amplitude (R_peak+ & R_turn), RR.

    Parameters
    ----------
    df_annot    : pd.DataFrame  –  df_annotations (aus `annotate_all_beats()`),
                                   muss 'beat_id', 't_R_peak+' und 't_R_turn'
                                   enthalten.
    df_analysis : pd.DataFrame  –  Vollständiges Signal-DataFrame, muss
                                   'Time', 'X', 'Y', 'Z' enthalten.
    smooth_window : int | None  –  Wenn gesetzt, werden zusätzlich per
                                   Savitzky-Golay geglättete Spalten
                                   (`<name>_smooth`) über
                                   `vcgsuite.signal_utils.savgol_smooth`
                                   ergänzt. None (Default) → keine Glättung.
    smooth_poly   : int         –  Polynomgrad für die Glättung (nur
                                   relevant wenn smooth_window gesetzt ist).

    Returns
    -------
    df_r : pd.DataFrame
    """
    t_arr = df_analysis['Time'].to_numpy()
    x_arr = df_analysis['X'].to_numpy()
    y_arr = df_analysis['Y'].to_numpy()
    z_arr = df_analysis['Z'].to_numpy()

    rows = []
    for _, row in df_annot.sort_values('beat_id').iterrows():
        t_rpk   = row.get('t_R_peak+', np.nan)
        t_rturn = row.get('t_R_turn',  np.nan)

        entry = {'beat_id': int(row['beat_id']), 't_R_peak+': float(t_rpk)}

        for name, t_q in [('Rpeak', t_rpk), ('Rturn', t_rturn)]:
            if not pd.isna(t_q):
                r, ca = _nearest_spherical(float(t_q), t_arr, x_arr, y_arr, z_arr)
            else:
                r, ca = np.nan, np.nan
            entry[f'{name}_r']  = r
            entry[f'{name}_ca'] = ca

        rows.append(entry)

    df_r = pd.DataFrame(rows)
    t_r  = df_r['t_R_peak+'].to_numpy(dtype=float)
    df_r['RR_ms'] = np.diff(t_r, prepend=np.nan) * 1000

    if smooth_window is not None:
        for col in ['Rpeak_r', 'Rturn_r', 'Rpeak_ca', 'Rturn_ca', 'RR_ms']:
            df_r[f'{col}_smooth'] = savgol_smooth(
                df_r[col].to_numpy(dtype=float), smooth_window, smooth_poly
            )

    return df_r
