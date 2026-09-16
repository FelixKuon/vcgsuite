import numpy as np
import pandas as pd


def detect_r_turn(df: pd.DataFrame, r_peak_times: np.ndarray,
                  search_lo_ms: float = 5.0, search_hi_ms: float = 13.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Sucht das erste lokale Minimum in A_abs nach jedem R_peak
    im Fenster [search_lo_ms, search_hi_ms] nach dem Peak.

    Returns
    -------
    r_turn_times  : np.ndarray  –  Zeitpunkte [s], NaN wenn nicht gefunden
    r_turn_values : np.ndarray  –  Amplitudenwerte
    """
    t   = df['Time'].values
    sig = df['A_abs'].values.astype(float)

    times, values = [], []
    for rpt in r_peak_times:
        mask = (t >= rpt + search_lo_ms / 1000.0) & \
               (t <= rpt + search_hi_ms / 1000.0)
        if mask.sum() < 2:
            times.append(np.nan); values.append(np.nan)
            continue
        seg = sig[mask]
        idx = np.argmin(seg)
        times.append(t[mask][idx])
        values.append(seg[idx])

    r_turn_times  = np.array(times)
    r_turn_values = np.array(values)
    deltas = (r_turn_times - r_peak_times) * 1000

    print(f"R_turn gefunden:   {(~np.isnan(r_turn_times)).sum()} / {len(r_peak_times)}")
    print(f"Ø Δ R_peak→R_turn: {np.nanmean(deltas):.1f} ms  (erwartet ~12 ms)")
    print(f"Min/Max Δ:         {np.nanmin(deltas):.1f} / {np.nanmax(deltas):.1f} ms")

    return r_turn_times, r_turn_values
