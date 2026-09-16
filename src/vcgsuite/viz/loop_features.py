# ══════════════════════════════════════════════════════════════════════════
#  LOOP-FEATURE ZEITREIHEN-PLOTS  (P-/QRS-/T-Loop)
#
#  Migriert aus plot_loop_features.py ("BLOCK B", "BLOCK B_T", "BLOCK B_QRS").
#
#  Das Original definierte `apply_savgol()` DREIMAL nahezu identisch (einmal
#  pro Block). Hier gibt es nur noch EINE generische Plot-Funktion
#  (`plot_loop_feature_timeseries`), die die SavGol-Glättung über
#  `vcgsuite.signal_utils.savgol_smooth` bezieht — keine eigene Kopie mehr.
#  Die drei Original-Blöcke sind als vorkonfigurierte Panel-Listen
#  (PANELS_P / PANELS_QRS / PANELS_T) + dünne Wrapper-Funktionen erhalten.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..signal_utils import savgol_smooth

BG = '#0f1117'
GRID = '#2a2a3a'
FONT = dict(family='Inter, sans-serif', color='#e0e0e0', size=11)

# Panel-Tupel: (Spaltenname, Farbe, Einheit, Beschreibung, Referenzlinie|None)
PANELS_P = [
    ('P_dur_ms', '#636efa', 'ms', 'P-Dauer', 110),
    ('P_rise_ms', '#ab63fa', 'ms', 'P_on → P_peak', None),
    ('PQ_ms', '#19d3f3', 'ms', 'PQ-Intervall', None),
    ('P_area', '#00cc96', 'a.u.', 'Loop-Fläche 3D', None),
    ('P_dipol_norm', '#ffa15a', 'a.u.', 'Dipolnorm ‖∫v dt‖', None),
    ('P_rho', '#ef553b', 'λ2/λ1', 'Rundheit ρ', 0.2),
    ('P_phi', '#19d3f3', 'λ3/Σλ', 'Planarität φ', None),
    ('P_asym', '#ff6692', '(LA−RA)', 'Asymmetrie', 0.0),
    ('P_round', '#b6e880', 'FWHM norm.', 'Roundness (FWHM)', None),
    ('theta_P_QRS', '#fdcb6e', '°', 'θ P↔QRS', None),
]

PANELS_QRS = [
    ('QRS_area', '#00cc96', 'a.u.', 'Loop-Fläche 3D', None),
    ('QRS_dipol_norm', '#ffa15a', 'a.u.', 'Dipolnorm ‖∫v dt‖', None),
    ('QRS_rho', '#ef553b', 'λ2/λ1', 'Rundheit ρ', 0.2),
    ('QRS_phi', '#19d3f3', 'λ3/Σλ', 'Planarität φ', None),
    ('QRS_asym', '#ff6692', 'sp/fr', 'Asymmetrie (Loop)', 0.0),
    ('G_QRS', '#636efa', 'sp/fr', 'G_QRS (spät/früh Fläche)', 1.0),
    ('QRS_max_speed', '#ab63fa', 'a.u./s', 'Max-Geschwindigkeit', None),
    ('QRS_mean_speed', '#e377c2', 'a.u./s', 'Mittl. Geschwindigkeit', None),
    ('theta_QRS_T_svd', '#fdcb6e', '°', 'θ QRS↔T (SVD-Achsen)', None),
    ('theta_QRS_P', '#bcbd22', '°', 'θ QRS↔P Achse', None),
    ('QRS_dur_ms', '#7f7f7f', 'ms', 'QRS-Dauer', 120),
    ('QRS_sym', '#b6e880', '0-1', 'QRS-Symmetrie (rise/total)', 0.5),
]

PANELS_T = [
    ('T_area', '#00cc96', 'a.u.', 'Loop-Fläche 3D', None),
    ('T_dipol_norm', '#ffa15a', 'a.u.', 'Dipolnorm ‖∫v dt‖', None),
    ('T_rho', '#ef553b', 'λ2/λ1', 'Rundheit ρ', 0.2),
    ('T_phi', '#19d3f3', 'λ3/Σλ', 'Planarität φ', None),
    ('T_asym_loop', '#ff6692', '(sp-fr)', 'Asymmetrie (Loop, spät−früh)', 0.0),
    ('T_round', '#b6e880', 'FWHM n.', 'Roundness (FWHM)', None),
    ('theta_P_T', '#fdcb6e', '°', 'θ P↔T Achse', None),
    ('theta_QT_deg', '#ab63fa', '°', 'θ QRS↔T (Winkel)', None),
    ('G_APD', '#636efa', 'sp/fr', 'G_APD (spät/früh Geschw.)', 1.0),
    ('T_mean_speed', '#e377c2', 'a.u./s', 'T-Mittl. Geschwindigkeit', None),
    ('T_asym', '#8c564b', 'sp/fr', 'T-Asymmetrie (Speed)', 1.0),
    ('T_width_ms', '#7f7f7f', 'ms', 'T-Breite', None),
    ('QTc_Bazett', '#bcbd22', 'ms', 'QTc Bazett', 440),
]


def plot_loop_feature_timeseries(df: pd.DataFrame, panels: list[tuple],
                                 id_col: str = 'beat_id',
                                 savgol_window: int = 3, savgol_poly: int = 1,
                                 title: str = 'Loop Features — Zeitreihe',
                                 row_height: int = 120) -> go.Figure:
    """
    Generische Zeitreihen-Darstellung von Loop-Features mit
    Savitzky-Golay-Glättungslinie (via `vcgsuite.signal_utils.savgol_smooth`).

    Parameters
    ----------
    df       : pd.DataFrame  –  df_p / df_qrs / df_t (o. ä.).
    panels   : list[tuple]   –  [(Spaltenname, Farbe, Einheit, Beschreibung,
                                 Referenzlinie|None), ...], siehe
                                PANELS_P/PANELS_QRS/PANELS_T oben.
    id_col   : str            –  X-Achsen-Spalte (Standard: 'beat_id').
    savgol_window, savgol_poly : int  –  Parameter für savgol_smooth().
    title    : str
    row_height : int          –  Höhe pro Panel in Pixeln.

    Returns
    -------
    go.Figure
    """
    panels = [p for p in panels if p[0] in df.columns]
    n = len(panels)
    bid = df[id_col]

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{desc}  [{unit}]' for _, _, unit, desc, _ in panels],
        vertical_spacing=0.025 if n <= 10 else 0.022,
    )

    for i, (col, color, unit, desc, ref) in enumerate(panels, start=1):
        vals = df[col]
        smoothed = savgol_smooth(vals, savgol_window, savgol_poly)

        # Rohdaten
        fig.add_trace(go.Scatter(
            x=bid, y=vals, mode='markers',
            marker=dict(color=color, size=4, opacity=0.4),
            name=desc, legendgroup=col, showlegend=True,
            hovertemplate=f'Beat %{{x}}<br>{desc}: %{{y:.4f}} {unit}<extra></extra>',
        ), row=i, col=1)

        # SavGol-Linie
        fig.add_trace(go.Scatter(
            x=bid, y=smoothed, mode='lines',
            line=dict(color=color, width=2),
            name=f'{desc} (SavGol w={savgol_window} p={savgol_poly})',
            legendgroup=col, showlegend=False,
            hovertemplate='SavGol: %{y:.4f}<extra></extra>',
        ), row=i, col=1)

        if ref is not None:
            fig.add_hline(y=ref,
                          line=dict(color='#888888', dash='dot', width=1),
                          annotation_text=str(ref),
                          annotation_font=dict(size=9, color='#888888'),
                          row=i, col=1)

        fig.update_yaxes(title_text=unit, row=i, col=1,
                         gridcolor=GRID, zerolinecolor=GRID,
                         title_font=dict(size=10))
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, row=i, col=1)

    fig.update_layout(
        height=row_height * n,
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        title=dict(
            text=f'{title}  |  SavGol window={savgol_window}, poly={savgol_poly}',
            font=dict(size=15, color='#ffffff')
        ),
        legend=dict(bgcolor='rgba(20,20,40,0.85)', bordercolor=GRID,
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=80, r=30, t=60, b=50),
    )
    fig.update_xaxes(title_text='Beat ID', row=n, col=1)
    fig.update_annotations(font=dict(color='#cccccc', size=11))
    return fig


def plot_p_loop_features(df_p: pd.DataFrame, savgol_window: int = 3,
                         savgol_poly: int = 1) -> go.Figure:
    """P-Loop-Feature-Zeitreihen (siehe BLOCK B im Original)."""
    return plot_loop_feature_timeseries(
        df_p, PANELS_P, savgol_window=savgol_window, savgol_poly=savgol_poly,
        title='P-Loop Features — Zeitreihe', row_height=130,
    )


def plot_qrs_loop_features(df_qrs: pd.DataFrame, savgol_window: int = 3,
                           savgol_poly: int = 1) -> go.Figure:
    """QRS-Loop-Feature-Zeitreihen (siehe BLOCK B_QRS im Original)."""
    return plot_loop_feature_timeseries(
        df_qrs, PANELS_QRS, savgol_window=savgol_window, savgol_poly=savgol_poly,
        title='QRS-Loop Features — Zeitreihe', row_height=120,
    )


def plot_t_loop_features(df_t: pd.DataFrame, savgol_window: int = 3,
                         savgol_poly: int = 1) -> go.Figure:
    """T-Loop-Feature-Zeitreihen (siehe BLOCK B_T im Original)."""
    return plot_loop_feature_timeseries(
        df_t, PANELS_T, savgol_window=savgol_window, savgol_poly=savgol_poly,
        title='T-Loop Features — Zeitreihe', row_height=120,
    )
