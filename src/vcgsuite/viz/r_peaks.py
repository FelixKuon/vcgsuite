# ══════════════════════════════════════════════════════════════════════════
#  R-PEAK DETEKTIONS-PLOT
#
#  Migriert aus plot_r_peak.py.
#
#  Hinweis Migration: Der Kommentar im Original ("Schwelle neu berechnen —
#  gleiche Logik wie in detect_r_peaks") ist NICHT (mehr) korrekt:
#  `vcgsuite.detection.r_peaks.detect_r_peaks()` verwendet inzwischen eine
#  adaptive, fensterbasierte Schwelle (gleitendes Perzentil über
#  `window_s`), keine einzelne globale `np.percentile(sig, 95)` mehr. Die
#  hier geplottete Schwellenlinie ist daher nur eine grobe, globale
#  Visualisierungs-Näherung der tatsächlich verwendeten adaptiven Schwelle
#  und NICHT identisch mit der Detektionsschwelle selbst.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_r_peak_detection(df_analysis: pd.DataFrame,
                          r_peak_times: np.ndarray,
                          r_peak_values: np.ndarray,
                          percentile: float = 95) -> go.Figure:
    """
    Plottet A_abs mit den detektierten R_peak+ Markern und einer groben
    (globalen) Schwellenlinie zur Orientierung.

    Parameters
    ----------
    df_analysis   : pd.DataFrame  –  muss Spalten 'Time', 'A_abs' enthalten.
    r_peak_times  : np.ndarray    –  Zeitpunkte [s], aus `detect_r_peaks()`.
    r_peak_values : np.ndarray    –  Amplitudenwerte, aus `detect_r_peaks()`.
    percentile    : float         –  Perzentil für die (rein visuelle,
                                    global-approximative) Schwellenlinie.

    Returns
    -------
    go.Figure
    """
    t = df_analysis["Time"].values
    sig = df_analysis["A_abs"].values.astype(float)

    thresh = np.percentile(sig, percentile)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=t, y=sig,
        mode="lines",
        name="A_abs",
        line=dict(color="#4f4f5e", width=0.8),
    ))

    fig.add_trace(go.Scattergl(
        x=r_peak_times, y=r_peak_values,
        mode="markers",
        name=f"R_peak+ ({len(r_peak_times)})",
        marker=dict(color="#ff5252", size=8, symbol="circle"),
    ))

    fig.add_hline(
        y=thresh,
        line=dict(color="#ffd700", width=1, dash="dash"),
        annotation_text=f"Schwelle ({thresh:.3f})",
        annotation_font_color="#ffd700",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0f11",
        plot_bgcolor="#0f0f11",
        title=dict(text="R_peak Detektion", font=dict(color="#e2e2e6")),
        xaxis=dict(title="Zeit (s)", color="#7a7a8c", gridcolor="#1e1e22"),
        yaxis=dict(title="A_abs", color="#7a7a8c", gridcolor="#1e1e22"),
        legend=dict(bgcolor="#1c1b19", font=dict(color="#e2e2e6")),
        height=350,
        hovermode="x unified",
    )

    return fig
