# ══════════════════════════════════════════════════════════════════════════
#  VCG-SIGNAL/3D-PLOT MIT ANNOTATIONEN
#
#  Migriert aus Plot_VCG_annotated.py (Zellen 9a-9d).
#
#  Hinweis Migration: Der Ausführungsteil am Ende des Originalskripts
#  ("Daten vorbereiten" + die beiden Beispielaufrufe von
#  `plot_signal_segments_and_markers()` / `plot_vcg_3d()` auf einem
#  Notebook-globalen df_analysis/df_annotations mit fest verdrahteten
#  Zeitfenstern T_START/T_END) ist entfernt — das sind reine
#  Beispielaufrufe, keine Bibliotheksfunktionalität. Alle Funktionen selbst
#  sind unverändert migriert und nehmen ihre Eingabe-DataFrames als
#  Parameter entgegen.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GT_MARKERS = [
    'Q_on', 'Q_off', 'R_peak+', 'R_turn',
    'S_on', 'S_off',
    'P_peak', 'P_on', 'P_off',
    'T_on', 'T_turn1', 'T_turn2', 'T_off',
]

QRS_MARKERS = ['Q_on', 'Q_off', 'R_peak+', 'R_turn', 'S_on', 'S_off']

MARKER_SEGMENT = {
    'P_on': 'P', 'P_peak': 'P', 'P_off': 'P',
    'Q_on': 'QRS', 'Q_off': 'QRS',
    'R_peak+': 'QRS', 'R_turn': 'QRS',
    'S_on': 'QRS', 'S_off': 'QRS',
    'T_on': 'T', 'T_turn1': 'T',
    'T_turn2': 'T', 'T_off': 'T',
}

SEGMENT_COLORS = {
    'P': '#1f77b4',    # Blau
    'QRS': '#d62728',   # Rot
    'T': '#2ca02c',    # Grün
    'PQ': '#a86fdf',   # Lila
    'ST': '#ff7f0e',   # Orange
}

MARKER_SIZES = {
    'R_peak+': 10, 'R_turn': 8,
    'P_peak': 8, 'T_turn1': 8, 'T_turn2': 8,
}  # default: 6

SEGMENT_EDGES = {
    'P': ('P_on', 'P_off'),
    'QRS': ('Q_on', 'S_off'),
    'T': ('T_on', 'T_off'),
    'PQ': ('P_off', 'Q_on'),
    'ST': ('S_off', 'T_on'),
}


def to_df_markers(detected: pd.DataFrame, beat_id: int | None = None,
                  markers: list[str] | None = None,
                  include_segments: list[str] | None = None) -> pd.DataFrame:
    """
    Wide-Format detected_df → df_markers für den Signalplot.

    Parameters
    ----------
    detected          : pd.DataFrame  –  df_annotations (Wide-Format)
    beat_id           : int|None      –  None → alle Beats
    markers           : list|None     –  None → alle GT_MARKERS
    include_segments  : list|None     –  z.B. ["QRS", "P"] → nur diese Segmente
    """
    rows_df = detected[detected['beat_id'] == beat_id] if beat_id is not None else detected

    allowed = set(markers or GT_MARKERS)
    if include_segments:
        seg_set = set(include_segments)
        allowed = {m for m in allowed if MARKER_SEGMENT.get(m, '') in seg_set}

    records = []
    for _, row in rows_df.iterrows():
        for m in GT_MARKERS:
            if m not in allowed:
                continue
            col = f't_{m}'
            if col not in row.index or pd.isna(row[col]):
                continue
            seg = MARKER_SEGMENT.get(m, 'QRS')
            records.append({
                'label': m,
                't': float(row[col]),
                'color': SEGMENT_COLORS.get(seg, '#888888'),
                'size': MARKER_SIZES.get(m, 6),
                'beat_id': int(row['beat_id']),
            })
    return pd.DataFrame(records)


def to_df_segments(detected: pd.DataFrame, beat_id: int | None = None,
                   segments: list[str] | None = None) -> pd.DataFrame:
    """
    Wide-Format detected_df → df_segments für den Signalplot.

    Parameters
    ----------
    detected  : pd.DataFrame  –  df_annotations (Wide-Format)
    beat_id   : int|None      –  None → alle Beats
    segments  : list|None     –  None → alle; z.B. ["QRS", "T"]
    """
    rows_df = detected[detected['beat_id'] == beat_id] if beat_id is not None else detected

    allowed_segs = set(segments or SEGMENT_EDGES.keys())
    records = []
    for _, row in rows_df.iterrows():
        for seg, (start_m, end_m) in SEGMENT_EDGES.items():
            if seg not in allowed_segs:
                continue
            ts = row.get(f't_{start_m}', np.nan)
            te = row.get(f't_{end_m}', np.nan)
            if pd.isna(ts) or pd.isna(te) or te <= ts:
                continue
            records.append({
                'label': seg,
                't_start': float(ts),
                't_end': float(te),
                'color': SEGMENT_COLORS.get(seg, '#888888'),
                'beat_id': int(row['beat_id']),
            })
    return pd.DataFrame(records)


def build_plot_data(detected: pd.DataFrame, beat_id: int | None = None,
                    mode: str = 'full'):
    """
    Erstellt df_markers und df_segments aus detected_df in einem Aufruf.

    Parameters
    ----------
    detected : pd.DataFrame  –  df_annotations
    beat_id  : int|None      –  None → alle Beats
    mode     : 'full'        –  alle Segmente
               'qrs'         –  nur QRS
               'p'           –  nur P-Welle
               't'           –  nur T-Welle
    """
    mode_map = {
        'full': (None, None),
        'qrs': (QRS_MARKERS, ['QRS']),
        'p': (['P_on', 'P_peak', 'P_off'], ['P']),
        't': (['T_on', 'T_turn1', 'T_turn2', 'T_off'], ['T']),
    }
    mk_list, seg_list = mode_map.get(mode, (None, None))

    df_mk = to_df_markers(detected, beat_id=beat_id, markers=mk_list, include_segments=seg_list)
    df_sg = to_df_segments(detected, beat_id=beat_id, segments=seg_list)

    print(f"df_markers:  {len(df_mk):>4d} Punkte   ({mode})")
    print(f"df_segments: {len(df_sg):>4d} Segmente ({mode})")
    return df_mk, df_sg


def select_time_window(df: pd.DataFrame, t_start: float | None = None,
                       t_end: float | None = None, time_col: str = 'Time') -> pd.DataFrame:
    """Gibt gefilterte Kopie von df für [t_start, t_end] zurück."""
    t = df[time_col].to_numpy()
    mask = np.ones(len(df), dtype=bool)
    if t_start is not None:
        mask &= (t >= t_start)
    if t_end is not None:
        mask &= (t <= t_end)
    return df.loc[mask].copy()


def _darken_hex(hex_color, factor: float = 0.75) -> str:
    """Dunkelt eine Hex-Farbe ab (factor < 1 = dunkler)."""
    h = str(hex_color).lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return '#{:02x}{:02x}{:02x}'.format(
        int(max(0, min(255, r * factor))),
        int(max(0, min(255, g * factor))),
        int(max(0, min(255, b * factor))),
    )


def plot_signal_segments_and_markers(
    df: pd.DataFrame, signal_col: str, df_segments: pd.DataFrame, df_markers: pd.DataFrame | None = None,
    time_col: str = 'Time', t_start: float | None = None, t_end: float | None = None,
    base_color: str = 'rgba(120,120,120,0.25)', base_width: int = 2,
    seg_width: int = 4, title: str | None = None,
) -> go.Figure:
    """
    Plottet ein Signal mit farbigen Segmentabschnitten und Markerpunkten.

    Parameters
    ----------
    df          : pd.DataFrame  –  Signaldaten (enthält time_col + signal_col)
    signal_col  : str           –  z.B. 'A_abs', 'V_abs', 'X'
    df_segments : pd.DataFrame  –  Ausgabe von to_df_segments()
    df_markers  : pd.DataFrame  –  Ausgabe von to_df_markers() (optional)
    t_start/end : float|None    –  Zeitfenster; None = gesamtes Signal
    """
    dfw = select_time_window(df, t_start, t_end, time_col).reset_index(drop=True)
    t = dfw[time_col].to_numpy()
    s = dfw[signal_col].to_numpy()

    fig = go.Figure()

    # ── Basislinie ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=s,
        mode='lines',
        line=dict(color=base_color, width=base_width),
        name=signal_col,
        hovertemplate=f'{signal_col}: %{{y:.6g}}<br>{time_col}: %{{x:.6g}}<extra></extra>',
        showlegend=False,
    ))

    # ── Farbige Segmentlinien ─────────────────────────────────────────────
    seen_labels = set()
    for _, seg in df_segments.iterrows():
        seg_df = select_time_window(dfw, float(seg['t_start']), float(seg['t_end']), time_col)
        if len(seg_df) < 2:
            continue
        label = str(seg.get('label', 'segment'))
        color = seg.get('color', '#888888')
        fig.add_trace(go.Scatter(
            x=seg_df[time_col], y=seg_df[signal_col],
            mode='lines',
            line=dict(color=color, width=seg_width),
            name=label,
            showlegend=(label not in seen_labels),
            hovertemplate=f'{label}<br>{signal_col}: %{{y:.6g}}<br>{time_col}: %{{x:.6g}}<extra></extra>',
        ))
        seen_labels.add(label)

    # ── Markerpunkte ──────────────────────────────────────────────────────
    if df_markers is not None and len(df_markers) > 0:
        dfm = df_markers.copy()
        if t_start is not None:
            dfm = dfm[dfm['t'] >= t_start]
        if t_end is not None:
            dfm = dfm[dfm['t'] <= t_end]

        if len(dfm) > 0:
            mk_t = dfm['t'].to_numpy(dtype=float)
            mk_idx = np.array([int(np.argmin(np.abs(t - tt))) for tt in mk_t])
            sizes = dfm['size'].to_numpy(dtype=float) * 1.8
            colors = np.array([_darken_hex(c, 0.75) for c in dfm['color'].to_numpy()])

            fig.add_trace(go.Scatter(
                x=t[mk_idx], y=s[mk_idx],
                mode='markers',
                marker=dict(size=sizes, color=colors,
                            line=dict(width=1, color='rgba(0,0,0,0.35)')),
                text=dfm['label'].to_numpy(),
                hovertemplate=(
                    '%{text}<br>'
                    f'{signal_col}: %{{y:.6g}}<br>'
                    f'{time_col}: %{{x:.6g}}<extra></extra>'
                ),
                name='Markers',
                showlegend=False,
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f0f11',
        plot_bgcolor='#0f0f11',
        title=dict(
            text=title or f'{signal_col}  [{t_start} – {t_end}]',
            font=dict(color='#e2e2e6')
        ),
        xaxis=dict(title=time_col, color='#7a7a8c', gridcolor='#1e1e22'),
        yaxis=dict(title=signal_col, color='#7a7a8c', gridcolor='#1e1e22'),
        legend=dict(bgcolor='#1c1b19', font=dict(color='#e2e2e6')),
        hovermode='x unified',
        height=400,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def plot_vcg_3d(
    df: pd.DataFrame, df_segments: pd.DataFrame, df_markers: pd.DataFrame | None = None,
    time_col: str = 'Time', x_col: str = 'X', y_col: str = 'Y', z_col: str = 'Z',
    t_start: float | None = None, t_end: float | None = None,
    base_color: str = 'rgba(120,120,120,0.25)', base_width: int = 3,
    seg_width: int = 7,
    title: str | None = None,
    camera_up: dict | None = None,
    camera_eye: dict | None = None,
    camera_center: dict | None = None,
    height: int = 750, width: int = 900,
) -> go.Figure:
    """
    3D VCG-Trajektorie mit farbigen Segmenten und Markerpunkten.

    Parameters
    ----------
    df          : pd.DataFrame  –  df_analysis
    df_segments : pd.DataFrame  –  Ausgabe von to_df_segments()
    df_markers  : pd.DataFrame  –  Ausgabe von to_df_markers() (optional)
    t_start/end : float|None    –  Zeitfenster; None = gesamtes Signal
    """
    camera_up = camera_up or dict(x=0, y=-1, z=0)
    camera_eye = camera_eye or dict(x=-0.4, y=0.1, z=2.0)
    camera_center = camera_center or dict(x=0, y=0, z=0)

    dfw = select_time_window(df, t_start, t_end, time_col).reset_index(drop=True)
    t = dfw[time_col].to_numpy()
    X = dfw[x_col].to_numpy()
    Y = dfw[y_col].to_numpy()
    Z = dfw[z_col].to_numpy()

    fig = go.Figure()

    # ── Basis-Trajektorie (grau) ──────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='lines',
        line=dict(color=base_color, width=base_width),
        name='Trajektorie',
        customdata=t.reshape(-1, 1),
        hovertemplate=(
            f'{time_col}: %{{customdata[0]:.4f}}s<br>'
            f'{x_col}: %{{x:.4f}}<br>'
            f'{y_col}: %{{y:.4f}}<br>'
            f'{z_col}: %{{z:.4f}}'
            '<extra></extra>'
        ),
        showlegend=True,
    ))

    # ── Farbige Segmente ──────────────────────────────────────────────────
    seen_labels = set()
    for _, seg in df_segments.iterrows():
        seg_df = select_time_window(dfw, float(seg['t_start']), float(seg['t_end']), time_col)
        if len(seg_df) < 2:
            continue
        label = str(seg.get('label', 'segment'))
        color = seg.get('color', '#888888')
        fig.add_trace(go.Scatter3d(
            x=seg_df[x_col], y=seg_df[y_col], z=seg_df[z_col],
            mode='lines',
            line=dict(color=color, width=seg_width),
            name=label,
            customdata=seg_df[[time_col]].to_numpy(),
            hovertemplate=(
                f'{label}<br>'
                f'{time_col}: %{{customdata[0]:.4f}}s<br>'
                f'{x_col}: %{{x:.4f}}<br>'
                f'{y_col}: %{{y:.4f}}<br>'
                f'{z_col}: %{{z:.4f}}'
                '<extra></extra>'
            ),
            showlegend=(label not in seen_labels),
        ))
        seen_labels.add(label)

    # ── Markerpunkte ──────────────────────────────────────────────────────
    if df_markers is not None and len(df_markers) > 0:
        dfm = df_markers.copy()
        if t_start is not None:
            dfm = dfm[dfm['t'] >= t_start]
        if t_end is not None:
            dfm = dfm[dfm['t'] <= t_end]

        for _, mk in dfm.iterrows():
            j = int(np.argmin(np.abs(t - float(mk['t']))))
            label = str(mk.get('label', 'marker'))
            color = _darken_hex(mk.get('color', '#888888'), 0.80)
            size = int(mk.get('size', 6)) + 2   # etwas größer für 3D

            fig.add_trace(go.Scatter3d(
                x=[X[j]], y=[Y[j]], z=[Z[j]],
                mode='markers+text',
                marker=dict(size=size, color=color,
                            line=dict(width=1, color='rgba(0,0,0,0.4)')),
                text=[label],
                textposition='top center',
                textfont=dict(size=9, color=color),
                name=label,
                customdata=[[t[j]]],
                hovertemplate=(
                    f'{label}<br>'
                    f'{time_col}: %{{customdata[0]:.4f}}s'
                    '<extra></extra>'
                ),
                showlegend=False,
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f0f11',
        height=height,
        width=width,
        margin=dict(l=10, r=10, t=60, b=10),
        title=dict(
            text=title or f'VCG 3D Trajektorie  [{t_start} – {t_end}]',
            font=dict(color='#e2e2e6'),
        ),
        legend=dict(bgcolor='#1c1b19', font=dict(color='#e2e2e6')),
        scene=dict(
            xaxis=dict(title=x_col, backgroundcolor='#0f0f11',
                       gridcolor='#1e1e22', zerolinecolor='#262523'),
            yaxis=dict(title=y_col, backgroundcolor='#0f0f11',
                       gridcolor='#1e1e22', zerolinecolor='#262523'),
            zaxis=dict(title=z_col, backgroundcolor='#0f0f11',
                       gridcolor='#1e1e22', zerolinecolor='#262523'),
            bgcolor='#0f0f11',
            aspectmode='data',
            dragmode='orbit',
            camera=dict(up=camera_up, eye=camera_eye, center=camera_center),
        ),
    )

    return fig
