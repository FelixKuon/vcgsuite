# ══════════════════════════════════════════════════════════════════════════
#  ANNOTATION ALLER BEATS
# ══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

from .hierarchical import annotate_beat_hierarchical

ALL_MARKERS = [
    'R_turn', 'S_on', 'Q_on', 'Q_off',
    'P_peak', 'P_on', 'P_off',
    'S_off', 'T_on', 'T_turn1', 'T_turn2', 'T_off',
]


def annotate_all_beats(df_analysis: pd.DataFrame,
                       r_peak_times: np.ndarray,
                       r_turn_times: np.ndarray) -> pd.DataFrame:
    """
    Annotiert alle Beats hierarchisch (siehe `annotate_beat_hierarchical`)
    und fasst die Marker-Zeitpunkte in einem DataFrame zusammen.

    Ersetzt die ursprüngliche Notebook-Zelle (annontation.py), die eine
    globale Variable `df_annotations` per Skript-Schleife erzeugte —
    stattdessen wird das Ergebnis hier als reiner Rückgabewert einer
    Funktion bereitgestellt.

    Parameters
    ----------
    df_analysis  : pd.DataFrame  –  Vollständiges Signal-DataFrame (Time, X, Y, Z, ...).
    r_peak_times : np.ndarray    –  R_peak+-Zeitpunkte [s], aus `detect_r_peaks()`.
    r_turn_times : np.ndarray    –  R_turn-Zeitpunkte [s], aus `detect_r_turn()`
                                    (gleiche Länge/Reihenfolge wie r_peak_times).

    Returns
    -------
    df_annotations : pd.DataFrame
        Eine Zeile pro Beat mit Spalten 'beat_id', 't_R_peak+' sowie
        't_<marker>' für jeden Marker in ALL_MARKERS.
    """
    rows = []
    for beat_id, rpt in enumerate(r_peak_times):
        r_turn_t = r_turn_times[beat_id] if beat_id < len(r_turn_times) else np.nan
        result   = annotate_beat_hierarchical(df_analysis, rpt, r_turn_t=r_turn_t)
        row      = {'beat_id': beat_id, 't_R_peak+': rpt}
        for marker in ALL_MARKERS:
            row[f't_{marker}'] = result.get(marker, np.nan)
        rows.append(row)

    df_annotations = pd.DataFrame(rows)

    print(f"Annotierte Beats: {len(df_annotations)}\n")
    print(f"{'Marker':12s}  {'gefunden':>8s}  {'Ø rel. Zeit':>12s}")
    print("-" * 40)
    for marker in ALL_MARKERS:
        col   = f't_{marker}'
        times = df_annotations[col].dropna()
        if len(times) == 0:
            print(f"  {marker:12s}  {'0':>7s}  –"); continue
        rel_ms = (times.values - df_annotations.loc[times.index, 't_R_peak+'].values) * 1000
        print(f"  {marker:12s}  {len(times):>7d}  {np.mean(rel_ms):>+10.1f} ms")

    return df_annotations
